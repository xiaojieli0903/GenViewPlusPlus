import re
import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
# from pytorch_lightning import seed_everything

from torch.utils.data import DataLoader, Dataset

from diffusers import DiffusionPipeline, StableUnCLIPImg2ImgPipeline, StableUnCLIPPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler

torch.set_grad_enabled(False)



def read_csv_with_fallback(file):
    df = pd.read_csv(file, nrows=1)
    has_header = not all(str(c).isdigit() for c in df.columns)

    if has_header:
        df = pd.read_csv(file)
    else:
        sample = pd.read_csv(file, header=None, nrows=1)
        num_columns = sample.shape[1]

        if num_columns == 2:
            df = pd.read_csv(file, header=None, names=['image', 'prompt'])
        elif num_columns == 3:
            col3_nonzero = sample[2][sample[2] != 0]
            if not col3_nonzero.empty:
                first_value = col3_nonzero.iloc[0]
                if first_value < 10:
                    df = pd.read_csv(file, header=None, names=['image', 'prompt', 'guidance_scale'])
                else:
                    df = pd.read_csv(file, header=None, names=['image', 'prompt', 'noise_level'])
            else:
                # If the third column is all 0, the default guidance_scale
                df = pd.read_csv(file, header=None, names=['image', 'prompt', 'guidance_scale'])
        elif num_columns == 4:
            df = pd.read_csv(file, header=None, names=['image', 'prompt', 'guidance_scale', 'noise_level'])
        else:
            raise ValueError(f"Unexpected number of columns: {num_columns}")
    has_guidance_scale = 'guidance_scale' in df.columns
    has_noise_level = 'noise_level' in df.columns
    return df, has_guidance_scale, has_noise_level


class PromptDataset(Dataset):
    """Build prompt loading dataset"""

    def __init__(self, file, start, n_skip, outdir, opt):
        self.file = file
        self.n_skip = n_skip

        with open(file, 'r') as f:
            first_line = f.readline().strip()
            columns = re.findall(r'(?:[^,"]+|"[^"]*")+', first_line)
            num_columns = len(columns)

        df, has_guidance_scale, has_noise_level = read_csv_with_fallback(file)
        self.has_guidance_scale = has_guidance_scale
        self.has_noise_level = has_noise_level

        self.data_list = df.to_dict('records') 
        
        self.data_list = [
            {**item, 'ids_name': int(item['image'].split('/')[-1].split('.')[0])} 
            for item in self.data_list
        ]
        self.ids = np.arange(len(self.data_list))
        print(f"total datas: {len(self.data_list)}")

        n_prompts_per_gpu = len(self.data_list) // n_skip + 1
        if start == n_skip - 1:
            self.ids = self.ids[n_prompts_per_gpu * start:]
            self.data_list = self.data_list[n_prompts_per_gpu * start:]
        else:
            self.ids = self.ids[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]
            self.data_list = self.data_list[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]
            
        # skip what has been generated, for resuming purpose
        self.outdir = outdir
        cur_id = self.skip_ids(opt)
        print(f"skipping {cur_id} datas!")

        self.data_list = self.data_list[cur_id:]
        self.ids = self.ids[cur_id:]
        
        print(f"remained datas: {len(self.data_list)}")
        
        self.num = len(self.data_list)

    def skip_ids(self, opt):
        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            split_size_folder = opt.split_size_folder
            split_size_image = opt.split_size_image
        else:
            split_size_folder = opt.split_size
            split_size_image = opt.split_size

        cur_id = 0
        for i, id in enumerate(self.ids):
            folder_level_1 = id // (split_size_folder * split_size_image)
            folder_level_2 = (id - folder_level_1 * split_size_folder * split_size_image) // split_size_image
            image_id = id - folder_level_1 * split_size_folder * split_size_image - folder_level_2 * split_size_image
            # file = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}", f"{image_id:05}.png")
            file = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}", f"{image_id:05}.jpg")
            if not os.path.isfile(file):
                break
            cur_id += 1
        return max(0, cur_id - 2)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        return self.data_list[item]


class ImageSaver(object):
    def __init__(self, outdir, opt):

        if opt.split:
            assert (opt.split_size > 0) or (opt.split_size_folder > 0 and opt.split_size_image > 0), \
                'splitting parameter wrong'

        self.outdir = outdir
        self.split = opt.split
        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            self.split_size_folder = opt.split_size_folder
            self.split_size_image = opt.split_size_image
        else:
            self.split_size_folder = opt.split_size
            self.split_size_image = opt.split_size
        self.save_size = opt.img_save_size
        self.last_folder_level_1 = -1
        self.last_folder_level_2 = -1
        os.makedirs(self.outdir, exist_ok=True)

        if self.split:
            self.cur_folder = None
        else:
            self.cur_folder = self.outdir

    def save(self, img, id, tag=None):
        id = int(id)
        if self.split:
            # compute folder id and image id
            folder_level_1 = id // (self.split_size_folder * self.split_size_image)
            folder_level_2 = (id - folder_level_1 * self.split_size_folder * self.split_size_image) // self.split_size_image
            image_id = id - folder_level_1 * self.split_size_folder * self.split_size_image - folder_level_2 * self.split_size_image
            if (self.cur_folder is None) or (self.last_folder_level_1 != folder_level_1) or \
                    (self.last_folder_level_2 != folder_level_2):
                self.cur_folder = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}")
                os.makedirs(self.cur_folder, exist_ok=True)
            self.last_folder_level_1 = folder_level_1
            self.last_folder_level_2 = folder_level_2
        else:
            image_id = id

        if tag is not None:
            # save_path = os.path.join(self.cur_folder, tag, f"{image_id:09}.png")
            # save_path = os.path.join(self.cur_folder, f"{image_id:09}_" + tag + ".png")
            save_path = os.path.join(self.cur_folder, f"{image_id:09}" + tag + ".jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            # save_path = os.path.join(self.cur_folder, f"{image_id:09}.png")
            save_path = os.path.join(self.cur_folder, f"{image_id:09}.jpg")
        img.save(save_path)

    def check(self, id, tag=None):
        id = int(id)
        if self.split:
            # compute folder id and image id
            folder_level_1 = id // (self.split_size_folder * self.split_size_image)
            folder_level_2 = (id - folder_level_1 * self.split_size_folder * self.split_size_image) // self.split_size_image
            image_id = id - folder_level_1 * self.split_size_folder * self.split_size_image - folder_level_2 * self.split_size_image
            if (self.cur_folder is None) or (self.last_folder_level_1 != folder_level_1) or \
                    (self.last_folder_level_2 != folder_level_2):
                self.cur_folder = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}")
                os.makedirs(self.cur_folder, exist_ok=True)
            self.last_folder_level_1 = folder_level_1
            self.last_folder_level_2 = folder_level_2
        else:
            image_id = id

        if tag is not None:
            save_path = os.path.join(self.cur_folder, f"{image_id:09}" + tag + ".jpg")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            # save_path = os.path.join(self.cur_folder, f"{image_id:09}.png")
            save_path = os.path.join(self.cur_folder, f"{image_id:09}.jpg")

        if os.path.isfile(save_path):
            return True
        else:
            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="data/output"
    )
    parser.add_argument(
        "--img_save_size",
        type=int,
        default=256,
        help="image saving size"
    )
    parser.add_argument(
        "--conditioned_mode",
        type=str,
        default='imgtxt',
        help="mode of the condition"
    )
    parser.add_argument(
        "--split",
        action='store_true',
        help="whether we split the data during saving (might further improve for many millions of images",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=1000,
        help="split size for saving images"
    )
    parser.add_argument(
        "--split_size_folder",
        type=int,
        default=1000,
        help="split size for number of folders inside each first level folder"
    )
    parser.add_argument(
        "--split_size_image",
        type=int,
        default=1000,
        help="split size for number of images inside each second level folder"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="how many prompts used in each batch"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--noise_level",
        type=int,
        default=100,
        help="noise level",
    )
    parser.add_argument(
        "--from_file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--root_path",
        type=str,
        help="root path of raw images",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the pre-trained model used for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    # distributed generation
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpus to use for generation",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="current gpu index",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1,
        help="number of nodes to use for generation",
    )
    parser.add_argument(
        "--node_idx",
        type=int,
        default=0,
        help="current node index",
    )
    opt = parser.parse_args()
    return opt


class StableUnCLIP:
    """
    Wrapper class for Stable UnCLIP image-to-image model.
    """
    def __init__(self, model_path, conditioned_mode='txt'):
        home_path = os.environ['HOME']
        default_img_model_path = f"{home_path}/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1-unclip/snapshots/e99f66a92bdcd1b0fb0d4b6a9b81b3b37d8bea44"
        default_txt_model_path = f"{home_path}/.cache/huggingface/hub/stable-diffusion-2-1"

        self.model_path = model_path
        if self.model_path is None:
            if conditioned_mode in ['img', 'imgtxt']:
                self.model_path = default_img_model_path
            elif conditioned_mode == 'txt':
                self.model_path = default_txt_model_path
                
        self.conditioned_mode = conditioned_mode

        # Load the appropriate model based on the conditioned_mode
        if conditioned_mode in ['img', 'imgtxt']:
            self.pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16, 
                revision="fp16"
            ).to("cuda")
        elif conditioned_mode in ['txt']:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16
            ).to("cuda")
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)


    def generate(self, img=None, prompt=None, time=1, steps=20, guidance_scale=2, noise_level=100):
        """
        Generate image-to-image transformations.
        """
        kwargs = dict(
            guidance_scale=guidance_scale,
            num_inference_steps=steps,
            noise_level=noise_level,
            num_images_per_prompt=time,
        )
        if self.conditioned_mode == 'imgtxt':
            output = self.pipe(img, prompt=prompt, **kwargs).images
        elif self.conditioned_mode == 'txt':
            output = self.pipe(prompt=prompt, **kwargs).images
        elif self.conditioned_mode == 'img':
            output = self.pipe(img, **kwargs).images
        return output


def main(opt):    
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    
    def make_tag(k: int, n_samples: int) -> str:
        return f"_sample{k}" if n_samples > 1 else ""

    # get the dataset and loader
    n_skip = opt.n_nodes * opt.n_gpus
    start = opt.node_idx * opt.n_gpus + opt.gpu_idx
    dataset = PromptDataset(file=opt.from_file, n_skip=n_skip, start=start, outdir=opt.outdir, opt=opt)
    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=1)
    print("images to generate:", len(dataset))
    
    # data saver
    gs_str = 'ada' if dataset.has_guidance_scale else str(opt.guidance_scale)
    nl_str = 'ada' if dataset.has_noise_level else str(opt.noise_level)
    folder_name =f'{opt.conditioned_mode}_guidance-{gs_str}_noise-{nl_str}_times{opt.n_nodes}_seed{opt.seed}'
    saver = ImageSaver(os.path.join(opt.outdir, folder_name), opt)

    # get the model
    generator = StableUnCLIP(opt.model_path, opt.conditioned_mode)
    
    for (i, data) in enumerate(data_loader): 
        prompts = data['prompt']
        imgs_paths = data['image']
        ids_name = data['ids_name']
        if opt.conditioned_mode.find('img') != -1:
            if not os.path.isfile(imgs_paths[0]):
                imgs = [Image.open(os.path.join(opt.root_path, img_path)).convert("RGB") for img_path in imgs_paths]
            else:
                imgs = [Image.open(img_path).convert("RGB") for img_path in imgs_paths]
        else:
            imgs = imgs_paths

        generated_images = False
        for j in range(len(ids_name)):
            for k in range(opt.n_samples):
                tag = make_tag(k, opt.n_samples)
                if not saver.check(ids_name[j], tag):
                    generated_images = True
                    break
                else:
                    print(f'Skipped {ids_name[j]}')
            if generated_images:
                break

        opt.guidance_scale, opt.noise_level = torch.tensor(opt.guidance_scale), torch.tensor(opt.noise_level)
        if generated_images:
            kwargs = dict(
                guidance_scale=data.get('guidance_scale', opt.guidance_scale).item(), 
                noise_level=data.get('noise_level', opt.noise_level).item()
            )
            print(f"image:{imgs_paths}, kwargs:{kwargs}")
            images = generator.generate(
                img=imgs, 
                prompt=prompts, 
                time=opt.n_samples, 
                steps=opt.steps, 
                **kwargs)

            # save images
            for j in range(len(ids_name)):
                for k in range(opt.n_samples):
                    img = images[j*opt.n_samples+k]
                    img = img.resize((opt.img_save_size, opt.img_save_size))
                    tag = make_tag(k, opt.n_samples)
                    saver.save(img, ids_name[j], tag)
        else:
            continue

if __name__ == "__main__":
    opt = parse_args()
    main(opt)
