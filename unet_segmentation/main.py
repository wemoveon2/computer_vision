import pytorch_lightning as pl 
from components.model import UNET
from components.data import SegmentationDataModule
from components.utils import train_transform
from argparse import ArgumentParser


if __name__ == '__main__':
    torch.cuda.empty_cache()

    ds = SegmentationDataModule(
        image_path='../input/carvana-image-masking-png/train_images', mask_path= '../input/carvana-image-masking-png/train_masks', transform = train_transform
    )

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument('--max_epochs', default=3)
    parser.add_argument('--accelerator', default='gpu')
    parser.add_argument('--gpus', default=1)
    args = parser.parse_args()

    model = UNET()
    trainer = pl.Trainer.from_argparse_args(args, profiler='simple')
    trainer.fit(model, ds)
    trainer.save_checkpoint("unet_segmentation_2.ckpt")

# Read https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c


