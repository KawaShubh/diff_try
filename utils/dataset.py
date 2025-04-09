import torch
from PIL import Image
import os
import numpy as np

class UnpairedWCEDataset:
    def __init__(self, hr_dir, lr_dir, hr_size=1024, lr_size=280):
        self.hr_files = sorted(os.listdir(hr_dir))
        self.lr_files = sorted(os.listdir(lr_dir))
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.hr_size = (hr_size, hr_size)
        self.lr_size = (lr_size, lr_size)

    def __len__(self):
        return min(len(self.hr_files), len(self.lr_files))

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_files[idx])).convert("RGB")
        lr_img = Image.open(os.path.join(self.lr_dir, self.lr_files[idx])).convert("RGB")
        
        # Resize and normalize to [-1, 1]
        hr_img = hr_img.resize(self.hr_size)
        lr_img = lr_img.resize(self.lr_size)
        
        hr_tensor = (torch.tensor(np.array(hr_img)).float() / 127.5 - 1).permute(2, 0, 1)
        lr_tensor = (torch.tensor(np.array(lr_img)).float() / 127.5 - 1).permute(2, 0, 1)
        
        return {"lr": lr_tensor, "hr": hr_tensor}