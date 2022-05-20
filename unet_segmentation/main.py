import hub 
import torch
import pytorch_lightning as pl
import torch.nn as nn
import torchvision.transforms as A
import components.model as cm
from torch.utils.data import DataLoader
from components import dataset
from albumentations.pytorch import ToTensorV2

if __name__ == '__main__':
    torch.cuda.empty_cache()
    ds = hub.load('hub://wemoveon/carvana_image_masking')
    train_transform = A.Compose(
        [
            A.Resize((160,240)),
            A.RandomRotation(35),
            A.RandomHorizontalFlip(p=0.5),
            A.RandomVerticalFlip(p=0.1),
            A.ToTensor(),
            A.Normalize(
                mean=(0),
                std=(1),

            )
        ],
    )
    ds = dataset.SegmentationDataset(ds, train_transform)
    dl = DataLoader(ds, batch_size=6, num_workers=6,shuffle=True)
    # dl = ds.pytorch(batch_size = 16, num_workers = 2, transform = {'images': train_transform , 'masks': train_transform }, shuffle = True)
    trainer = pl.Trainer(max_epochs=60, accelerator='auto', default_root_dir='.\\unet_segmentation\\checkpoints')
    model = cm.UNET(3,1)
    model = cm.LightModel(model, nn.BCEWithLogitsLoss())
    trainer.fit(model, dl)
    trainer.save_checkpoint("unet_segmentation.ckpt")