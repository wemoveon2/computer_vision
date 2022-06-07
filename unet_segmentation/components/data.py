import torch
import pytorch_lightning as pl
import math
import glob
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class SegmentationDataset(Dataset):
  def __init__(self, image_path, mask_path, transforms):
    self.images = glob.glob(os.path.join(image_path, '*.jpg'))
    self.image_path = image_path
    self.mask_path = mask_path
    self.transforms = transforms

  def __len__(self):
    return len(self.images)
  
  def __getitem__(self, idx):
    img = np.array(Image.open(self.images[idx]).convert('RGB'))
    mask = np.array(Image.open(os.path.join(self.mask_path, os.path.basename(self.images[idx]).replace('.jpg', '.png')))) 
    mask[mask == 255.0] = 1.0  
    augmentations = self.transforms(image=img, mask=mask)
    image = augmentations["image"]
    mask = augmentations["mask"]
    mask = torch.unsqueeze(mask, 0)
    mask = mask.type(torch.float32)
    return image, mask

class SegmentationDataModule(pl.LightningDataModule):
    
    def __init__(self, image_path, mask_path, transform, train_size=0.90, batch_size: int = 7):
        super().__init__()
        self.image_path = image_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.transform = transform
        self.train_size = train_size
        
    def setup(self, stage = None):
        if stage in (None, 'fit'):
            ds = SegmentationDataset(self.image_path, self.mask_path, self.transform)
            train_size = math.floor(len(ds)*self.train_size)
            val_size = len(ds)-train_size
            train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
            self.train_dataset = train_ds
            self.val_dataset = val_ds
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.batch_size, num_workers=2, shuffle = True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size, num_workers=2, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.val_dataset, self.batch_size)
