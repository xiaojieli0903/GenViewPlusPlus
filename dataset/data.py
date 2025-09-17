import logging
import random
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import csv
import clip
from tqdm import tqdm
import torchvision.transforms as transforms

csv.field_size_limit(500 * 1024 * 1024)


class NCropTransform:
    """Create N crops from the same image"""
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        res = []
        for i in range(self.n):
            res.append(self.transform(x))
        return torch.cat(res, dim=0)


class SupconDataset(Dataset):
    def __init__(self, input_filename, transforms, 
                 num_views=1, root_list=None, 
                 syn_idx_list=None,
                 num_crop=1, tokenizer=None, 
                 root_suffix_list=None, real_images_path_suffix=None, 
                 syn_input_filename=None, syn_ratio=1.0,
                 path1=None, path2=None, sample_mode='default'):
        logging.debug(f'Loading csv data from {input_filename}.')
        assert num_crop >= 1, f'number of crops is less than 1: {num_crop}'
        self.images = []
        self.captions = []
        self.num_views = num_views
        self.tokenizer = tokenizer
        self.root_list = root_list
        self.syn_idx_list = syn_idx_list
        self.root_suffix_list = root_suffix_list
        self.real_images_path_suffix = real_images_path_suffix
        self.path_dict = {'path1':path1, 'path2':path2}
        self.sample_mode = sample_mode
        assert input_filename.endswith('.csv')
        with open(input_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith(('.png', '.jpg')):
                    self.images.append(image)  # relative dir
                    self.captions.append(prompt)
        
        self.syn_images = set()
        if syn_input_filename is not None:
            with open(syn_input_filename, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in tqdm(csv_reader):
                    image = row[0]
                    prompt = row[1]
                    if image.endswith(('.png', '.jpg')):
                        self.syn_images.add(image[:-4])
        
        #self.syn_sample_ratio = syn_ratio / (len(self.syn_images) / len(self.images)) if len(self.syn_images)!=0 else 0
        self.syn_sample_ratio = syn_ratio
        assert self.syn_sample_ratio <= 1.0
        
        if num_crop > 1:
            self.transforms = NCropTransform(transforms, num_crop)
        else:
            self.transforms = transforms
        logging.debug('Done loading data.')
        
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = []
        
        random_value = random.random()
        
        image_name = self.images[idx]
        if image_name.startswith("path1/") or image_name.startswith("path2/"):
            real_path = self.path_dict[image_name[:5]]
            image_name = image_name[6:]
        else:
            real_path = self.real_images_path_suffix[0]
            
        real_image_path = os.path.join(real_path, image_name)
        real_image_path = real_image_path[:-4] + self.real_images_path_suffix[1]
        if self.tokenizer is not None:
            pure_real_image = Image.open(real_image_path).convert('RGB')
            pure_real_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            pure_real_image = pure_real_transform(pure_real_image)
        
        if self.sample_mode == 'fixed+random':
            assert len(self.root_list) >= 2, "Need at least 2 roots for fixed+random sampling"
            selected_indices = [0, random.choice(range(1, len(self.root_list)))]
        elif self.sample_mode == 'random':
            assert len(self.root_list) >= 1, "Need at least 1 root for random sampling"
            selected_indices = [random.choice(range(0, len(self.root_list)))]
        else:
            selected_indices = list(range(self.num_views))
        
        for i in selected_indices:          
            current_path = os.path.join(self.root_list[i], image_name)
            current_path = current_path[:-4] + self.root_suffix_list[i]
            if i in self.syn_idx_list and image_name[:-4] in self.syn_images and random_value <= self.syn_sample_ratio:
                image_path = current_path
            else:
                image_path = real_image_path
                
            image = Image.open(image_path).convert('RGB')
            images.append(self.transforms(image))
        random.shuffle(images)
        # concat image on channel dim
        images = torch.cat(images, dim=0)
        if self.tokenizer is None:
            return images, idx
        else:
            # texts = self.tokenizer(str(self.captions[idx]))
            texts = clip.tokenize(self.captions[idx], truncate=True).squeeze().long()
            return images, texts, pure_real_image, idx
