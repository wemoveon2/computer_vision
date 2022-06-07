import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import pytorch_lightning as pl
from typing import List

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNET(pl.LightningModule):
    def __init__(self, in_channels: int=3, out_channels: int=1, features: List=[64,128,256,512]):
        super().__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        for feature in features:
            self.down.append(Block(in_channels, feature))
            in_channels=feature
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feature*2, feature, 2, 2)
            )
            self.up.append(
                Block(feature*2, feature) # x gets concat to 2xchannel
            )
        self.bottleneck = Block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        
        self.num_correct = 0
        self.num_pixels = 0
        self.dice_score = 0
    def forward(self, x):
        skip_connections = []
        for down in self.down:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up), 2):
            x = self.up[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1) # Concat along channels (b, c, h, w)
            x = self.up[idx+1](concat_skip)
        return self.final_conv(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        self.log('train_loss', loss, logger = True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss_fn(pred, y)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        self.num_correct += (pred == y).sum()
        self.num_pixels += torch.numel(pred)
        self.dice_score += (2 * (pred * y).sum()) / (
            (pred + y).sum() + 1e-8
        )
        self.log('val_loss', loss, prog_bar = True, logger = True)
        return {'loss': loss, 'len': len(self.trainer.val_dataloaders[0])}
    
    def validation_epoch_end(self, output):
        val_acc = float(f'{self.num_correct/self.num_pixels*100:.2f}')
        self.log('val_acc', val_acc, prog_bar = True, logger = True)
        dice_score = self.dice_score/len(output)
        self.log('dice_score', dice_score, prog_bar = True, logger = True)
        self.num_correct, self.num_pixels, self.dice_score = 0,0,0
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params = self.parameters(), lr = 1.5e-3, weight_decay = 0.3)
        return optimizer