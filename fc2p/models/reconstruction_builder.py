import torch
import torch.nn as nn

# from model_utils import *
from .model_utils import *

class GlobalNet(nn.Module):

    def __init__(self, ch_in, ch_base):
        super(GlobalNet, self).__init__()

        self.head = conv_block(ch_in, ch_base // 2, ch_base // 2) # rf 5

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), # rf 10
            conv_block(ch_base // 2, ch_base, ch_base) # rf 14
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), # rf 28
            conv_block(ch_base, ch_base * 2, ch_base * 2) # rf 32
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), # rf 64
            conv_block(ch_base * 2, ch_base * 4, ch_base * 4) # rf 68
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_block(ch_base * 4, ch_base * 2, ch_base * 2),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_block(ch_base * 2, ch_base, ch_base)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_block(ch_base, ch_base // 2, ch_base // 2)
        )

        self.out = nn.Sequential(
            conv_block(ch_base// 2, ch_in, ch_in),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u3 = self.up3(d3)
        u2 = self.up2(u3)
        u1 = self.up1(u2)
        y = self.out(u1)
        return y
    
class GlobalNet_v2(nn.Module):

    def __init__(self, ch_in, ch_base):
        super(GlobalNet_v2, self).__init__()

        self.head = hor_block(ch_in, ch_base // 2) # rf 5

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2), # rf 10
            hor_block(ch_base // 2, ch_base) # rf 14
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2), # rf 28
            hor_block(ch_base, ch_base * 2) # rf 32
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2), # rf 64
            hor_block(ch_base * 2, ch_base * 4) # rf 68
        )

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            hor_block(ch_base * 4, ch_base * 2),
        )

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            hor_block(ch_base * 2, ch_base)
        )

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            hor_block(ch_base, ch_base // 2)
        )

        self.out = nn.Sequential(
            hor_block(ch_base// 2, ch_in),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        u3 = self.up3(d3)
        u2 = self.up2(u3)
        u1 = self.up1(u2)
        y = self.out(u1)
        return y
    
if __name__ == '__main__':
    inputs = torch.randn(4, 576, 64, 64)
    net = GlobalNet_v2(576, 256)
    outputs = net(inputs)
    print(outputs.shape)