import os
import csv
import time
import json
import torch
import random
import argparse
import numpy as np
import transformers
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed


system_content = """You are tasked with evaluating the visual complexity of a text prompt intended for image generation. Your goal is to assign a score from 1 to 4 that reflects how richly the prompt describes concrete visual elements. The more detailed and diverse the visual information, the higher the score should be.
Visual complexity is determined by identifying constraints that directly affect how an image would be generated. These include the specificity of objects mentioned (such as type, quantity, or material), descriptive visual features (such as color, texture, or lighting), spatial relationships (such as layout or perspective), dynamic elements (such as actions or interactions), and stylistic cues (such as artistic genres or cultural elements). A prompt that contains a wide range of such features is considered more complex than one that simply names general categories.
A prompt that only refers to broad object types without any additional visual information should be scored as 1. If it includes a small number of visual constraints—typically one to three—it should be scored as 2. A score of 3 is appropriate when the prompt contains a moderate level of detail, roughly four to six constraints. A prompt that includes more than six distinct and specific visual elements—such as combinations of materials, textures, color, spatial arrangement, and style—should be scored as 4.
When evaluating, do not include abstract or emotional language that does not translate directly into visual features. Focus only on concrete and visualizable information.
After analyzing the prompt, return a score from 1 to 4"""

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def score_on_text(model, text_prompt: str, explain: bool = False):
    ending = " along with a brief explanation of which visual elements contributed to the final decision." if explain \
        else "."
    evaluation_format = "\nReturn the result in the following format: [score: ?, explanation: ?]" if explain \
        else "\nReturn the result in the following format: [score: ?]"
        
    messages = [
        {
            "role": "system",
            "content": system_content + ending + evaluation_format 
        },
        {
            "role": "user", 
            "content": f"""Now output the score of visual constraints for the following text prompt:{text_prompt}."""
        }
    ]
    
    input_tensor = model["tokenizer"].apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    outputs = model["model"].generate(input_tensor.to(model["model"].device), max_new_tokens=1000)

    evalutaion_result = model["tokenizer"].decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True, skip_prompt=False)
    
    score = None
    explanation = None
    if explain:
        if "score:" in evalutaion_result and "explanation:" in evalutaion_result:
            score = evalutaion_result.split("score:")[1].split("explanation:")[0].strip().strip('[]')
            explanation = evalutaion_result.split("explanation:")[1].strip().strip('[]')
    else:
        if "score:" in evalutaion_result:
            score = evalutaion_result.split("score:")[1].strip().strip('[]')
    
    return score, explanation


def make_dir(output_dir):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)


def load_model(model_name, model_path=None):
    if model_name == "deepseek":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
        model.generation_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
        return {"model":model, "tokenizer":tokenizer}
    else:    # model_name == "llama"
        pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",
        )
        return pipeline
    

class CSVTextDataset(Dataset):
    def __init__(self, csv_file, output_csv, start=0, n_skip=1, has_header=False):
        self.data = []

        existing_keys = set()
        try:
            with open(output_csv, "r", encoding="utf-8") as out_file:
                reader = csv.reader(out_file, quotechar='"', skipinitialspace=True)
                if has_header:
                    next(reader, None)
                for row in reader:
                    if row and len(row) > 0:
                        existing_keys.add(row[0])
        except FileNotFoundError:
            print(f"{output_csv} does not exist, a new file will be created.")
            
        with open(csv_file, "r", encoding="utf-8") as infile:
            reader = csv.reader(infile, quotechar='"', skipinitialspace=True)
            if has_header:
                next(reader, None)
            for row in reader:                
                if not row or len(row) < 2:
                    print(f"Skipping invalid row: {row}")
                    continue
                if row[0] in existing_keys:
                    print(f"Skipping duplicate row: {row}")
                    continue
                self.data.append(row)

        total_samples = len(self.data)
        n_samples_per_gpu = (total_samples + n_skip - 1) // n_skip
        start_idx = n_samples_per_gpu * start
        end_idx = min(n_samples_per_gpu * (start + 1), total_samples)
        
        self.data = self.data[start_idx:end_idx]

        print(f"Total samples: {total_samples}")
        print(f"GPU {start} processing {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_csv", type=str, required=True, help="Path to input CSV file")
    parser.add_argument("--has_header", type=bool, default=True, help="Whether the input file contains a header row.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output CSV file")
    parser.add_argument("--explain", action="store_true", help="Whether to generate explanations")
    
    parser.add_argument("--model_name", type=str, required=True, help="Model name for processing")
    parser.add_argument("--model_path", type=str, default=os.path.expanduser("~/.cache/huggingface/hub/DeepSeek-V2-Lite-Chat"), 
                        help="Path to the pretrained language model.")

    parser.add_argument("--n_gpus", type=int, default=1, help="Total number of GPUs used")
    parser.add_argument("--gpu_rank", type=int, default=0, help="Current GPU index")
    
    parser.add_argument("--seed", type=int, default=0)
    
    return parser.parse_args()


def setup_seed(seed):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    set_seed(seed)
    
    print(f"Random seed set to {seed} (rank: {get_rank()})")


def is_score(value):
    return (isinstance(value, str) and value in {"1", "2", "3", "4"})

    
def main():
    opt = parse_args()
    setup_seed(opt.seed + get_rank())
    
    score_to_guidance = {"1": "8", "2": "6", "3": "4", "4": "2"}

    output_csv = opt.output_csv
    output_csv_with_explanation = output_csv[:-4] + "_withEX.csv"
    
    make_dir(os.path.dirname(output_csv))
    make_dir(os.path.dirname(output_csv_with_explanation))

    dataset = CSVTextDataset(opt.input_csv, output_csv, start=opt.gpu_rank, n_skip=opt.n_gpus, has_header=opt.has_header)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    
    returned_model = load_model(opt.model_name, model_path=opt.model_path)
    
    re_score_file = os.path.join(os.path.dirname(output_csv), "re-score.txt")

    start_time = time.time()
    with open(output_csv, "a", newline="") as file1, open(re_score_file, "a") as file3:
        writer1 = csv.writer(file1)

        writer2 = None
        if opt.explain:
            file2 = open(output_csv_with_explanation, "a", newline="")
            writer2 = csv.writer(file2)

        write_header1 = not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0
        if write_header1:
            writer1.writerow(['image', 'prompt', 'guidance_scale'])

        if opt.explain:
            write_header2 = not os.path.exists(output_csv_with_explanation) or os.path.getsize(output_csv_with_explanation) == 0
            if write_header2:
                writer2.writerow(['image', 'prompt', 'guidance_scale', 'explanation'])

        for row in tqdm(dataset, total=len(dataset), desc="Processing", unit="file"):
            image_name, text = row[0], row[1]
            score, times = "", 0
            while ((not is_score(score)) and times < 5):
                score, explanation = score_on_text(returned_model, text, opt.explain)
                if times:
                    print(f"re score:{text}")
                    file3.write(f"re score:{text}\n")
                    file3.flush()
                times += 1

            guidance_scale = score_to_guidance.get(score, score)
            writer1.writerow([image_name, text, guidance_scale])
            if opt.explain:
                writer2.writerow([image_name, text, guidance_scale, explanation])

            print(f"Processed: {image_name} → Score: {score} → Guidance Scale: {guidance_scale}")

        if opt.explain:
            file2.close()

    end_time = time.time()
    print(f"total time: {end_time - start_time:.2f} seconds")

    if os.path.exists(re_score_file) and os.path.getsize(re_score_file) == 0:
        try:
            os.remove(re_score_file)
        except Exception as e:
            print(f"Error when deleting 're-score.txt': {e}")

if __name__ =='__main__':
    main()
