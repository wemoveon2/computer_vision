import math
import torch
import torchmetrics
import torch.nn as nn
import torchvision.transforms.functional as TF
from transformers import get_cosine_schedule_with_warmup
import pytorch_lightning as pl

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias= False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64,128,256,512]):
        super(UNET, self).__init__()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        # If we input 161x161, pool floors the division to 80x80, this becomes 160x160 in the upsampling
        # this then results in us not being able to concat the skip connection
        # need to ensure all inputs are divisible by 16 or add general sol
        # cropping was used in the paper
        
        # UNET downsampling
        for feature in features:
            self.down.append(Block(in_channels, feature))
            in_channels=feature
        
        # UNET upsampling
        # Transposed convolutions
        # https://d2l.ai/chapter_computer-vision/transposed-conv.html
        # Can use a bilinear and then conv layer
        
        for feature in reversed(features):
            self.up.append(
                nn.ConvTranspose2d(feature*2, feature, 2, 2)
            )
            self.up.append(
                Block(feature*2, feature) # x gets concat to 2xchannel
            )
        self.bottleneck = Block(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)
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

def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class LightModel(pl.LightningModule):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.dice = []
        self._initialize_weights()
        # self.accuracy = torchmetrics.Accuracy()
    # def forward(self, x, y):
    #     pred = self.model(x)
    #     if y is not None:
    #         loss = self.loss_fn(pred, y)    
    #     return pred, loss
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss_fn(pred, y)
        dice = dice_score(pred, y)
        self.dice.append(dice)
        # self.accuracy(pred, y)
        self.log('train loss', loss, prog_bar = True, logger = True)
        # self.log('train_acc', self.accuracy)
        self.log('dice_score', dice)
        return {"loss":loss, 'predictions':pred, 'mask': batch[1]}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params = self.model.parameters(), lr = 1.5e-4, weight_decay = 0.3)
        # total_steps = 848
        # warmup_steps = math.floor(total_steps * 0.2)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        return [optimizer]#, [scheduler]

if __name__ == '__main__':
    model = UNET(3,1)
    model = LightModel(model, nn.BCEWithLogitsLoss())
    print(model.parameters())
