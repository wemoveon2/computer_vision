import torch
import torchvision
import albumentations as A
import os 
import glob
from albumentations.pytorch import ToTensorV2
from PIL import Image

def save_images(model, loader, folder='val_img', device='cuda'):
    model.eval()
    if not os.path.isdir(folder):
        os.mkdir(folder)
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x).cuda())
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y, f"{folder}/mask_{idx}.png")  

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def merge_photos(src_folder: str='./val_img', dst_folder: str='./merged_val_img', remove_single: bool=True):
    files = glob.glob(src_folder+'/*.png')
    for i in range(int(len(files)/2)):
        pred_img = Image.open(f'{src_folder}/pred_{i}.png')
        mask_img = Image.open(f'{src_folder}/mask_{i}.png')
        get_concat_v(pred_img, mask_img).save(f'{dst_folder}/merged_pred_mask_{i}.png')
        if remove_single:
            os.remove(f'./val_img/pred_{i}.png')
            os.remove(f'./val_img/mask_{i}.png')

train_transform = A.Compose(
    [
        A.Resize(height=360, width=480),
        A.Rotate(limit=45, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        A.pytorch.ToTensorV2(),
    ]
)

inference_transform = A.Compose(
    [
        A.Resize(height=360, width=480),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        A.pytorch.ToTensorV2(),
    ]
)
