import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
class SegmentationDataset(Dataset):
  def __init__(self, ds, transform = None):
    self.ds = ds
    self.transform = transform

  def __len__(self):
    return len(self.ds)
  
  def __getitem__(self, idx):
    image = self.ds.images[idx].numpy()
    mask = self.ds.masks[idx].numpy().astype(np.float32)
    mask = np.squeeze(mask, axis=2)
    mask[mask == 255.0] = 1.0
    image = Image.fromarray(image).convert('RGB')
    mask = Image.fromarray(mask.astype(np.uint8))
    image, mask = self.transform(image)
    mask = self.transform(mask)
    # if self.transform is not None:
    #     # augmentations = self.transform(images=image, mask=mask)
    #     # image = augmentations["image"]
    #     # mask = augmentations["mask"]
    #     image, = self.transform([image, mask])
        # mask = self.transform(mask)

    return image, mask

