import os
import os.path as osp
import numpy as np
import torch
import random
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from typing import Optional
from safetensors.torch import load_file, save_file
from concurrent.futures import ThreadPoolExecutor, as_completed
from einops import rearrange



class IN1kLatentDataset(Dataset):
    def __init__(self, root_dir, target_len=256, random='random'):
        super().__init__()
        self.RandomHorizontalFlipProb = 0.5
        self.root_dir = root_dir
        self.target_len = target_len
        self.random = random
        self.files = []
        files_1 = os.listdir(osp.join(root_dir, f'from_16_to_{target_len}'))
        files_2 = os.listdir(osp.join(root_dir, f'greater_than_{target_len}_resize'))
        files_3 = os.listdir(osp.join(root_dir, f'greater_than_{target_len}_crop'))
        files_23 = list(set(files_2) - set(files_3))    # files_3 in files_2
        self.files.extend([
            [osp.join(root_dir, f'from_16_to_{target_len}', file)] for file in files_1
        ])
        self.files.extend([
            [osp.join(root_dir, f'greater_than_{target_len}_resize', file)] for file in files_23
        ])
        self.files.extend([
            [
                osp.join(root_dir, f'greater_than_{target_len}_resize', file), 
                osp.join(root_dir, f'greater_than_{target_len}_crop', file)
            ] for file in files_3
        ])
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.random == 'random':
            path = random.choice(self.files[idx])
        elif self.random == 'resize':
            path = self.files[idx][0]  # only resize
        elif self.random == 'crop':
            path = self.files[idx][-1]  # only crop
        data = load_file(path)
        dtype = data['feature'].dtype
        
        feature = torch.zeros((self.target_len, 16), dtype=dtype)
        grid = torch.zeros((2, self.target_len), dtype=dtype)
        mask = torch.zeros((self.target_len), dtype=torch.uint8)
        size = torch.zeros(2, dtype=torch.int32)
        
        
        seq_len = data['grid'].shape[-1]
        if torch.rand(1) < self.RandomHorizontalFlipProb:
            feature[0: seq_len] = rearrange(data['feature'][0], 'h w c -> (h w) c')
        else:
            feature[0: seq_len] = rearrange(data['feature'][1], 'h w c -> (h w) c')
        grid[:, 0: seq_len] = data['grid']
        mask[0: seq_len] = 1
        size = data['size'][None, :]
        label = data['label']
        return dict(feature=feature, grid=grid, mask=mask, label=label, size=size)
        
       

# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60

# from https://github.com/Alpha-VLLM/LLaMA2-Accessory/blob/main/Large-DiT-ImageNet/train.py#L60
def get_train_sampler(dataset, global_batch_size, max_steps, resume_steps, seed):
    sample_indices = torch.empty([max_steps * global_batch_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[
            :sample_indices.size(0) - fill_ptr
        ]
        sample_indices[fill_ptr: fill_ptr + epoch_sample_indices.size(0)] = \
            epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_steps * global_batch_size : ].tolist()


   
class INLatentLoader():
    def __init__(self, train):
        super().__init__()

        self.train_config = train

        self.batch_size = self.train_config.loader.batch_size
        self.num_workers = self.train_config.loader.num_workers
        self.shuffle = self.train_config.loader.shuffle

        self.train_dataset = IN1kLatentDataset(
            self.train_config.data_path, self.train_config.target_len, self.train_config.random
        )
        
        
        self.test_dataset = None
        self.val_dataset = None

    def train_len(self):
        return len(self.train_dataset)

    def train_dataloader(self, global_batch_size, max_steps, resume_step, seed=42):
        sampler = get_train_sampler(
            self.train_dataset, global_batch_size, max_steps, resume_steps, seed
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return None

    def val_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
