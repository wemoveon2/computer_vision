import torch
import torch.nn as nn

class block(nn.Module):
    
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()
        # Channels after each block is 4x what entered
        self.expansion = 4
        self.c1 = nn.Conv2d(in_channels, out_channels, 1, )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(out_channels, out_channels, 3, stride, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.c3 = nn.Conv2d(out_channels, out_channels*self.expansion, 1)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        
        x += identity
        x = self.relu(x)
        return x
    
class ResNet(nn.Module): # [3,4,6,3]
    def __init__(self, block, layers, image_channels, n_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.c1 = nn.Conv2d(image_channels, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(3,2,1)
        
        self.l1 = self._make_layer(block, layers[0], 64, 1)
        self.l2 = self._make_layer(block, layers[1], 128, 2)
        self.l3 = self._make_layer(block, layers[2], 256, 2)
        self.l4 = self._make_layer(block, layers[3], 512, 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, n_classes)
    
    def forward(self, x):
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x      

    def _make_layer(self, block, n_res, out_channels, stride):
        identity_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != out_channels * 4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, 1, stride),
                                                nn.BatchNorm2d(out_channels*4))
        
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels=out_channels*4
        for i in range(n_res-1):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)

def ResNet50(img_channels = 3, n_classes = 1000):
    return ResNet(block, [3,4,6,3], img_channels, n_classes)

def test():
    net = ResNet50(3, 1000)
    y = net(torch.randn(4, 3, 224, 224)).to("cuda")
    print(y.size())


test()