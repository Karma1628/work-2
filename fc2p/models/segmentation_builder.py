import torch
import torch.nn as nn

# from model_utils import *
from .model_utils import *

class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
    def forward(self, x):
        b1,b2,b3,b4,b5,b6 = self.encoder_segment(x)
        output_segment = self.decoder_segment(b1,b2,b3,b4,b5,b6)
        if self.out_features:
            return output_segment, b2, b3, b4, b5, b6
        else:
            return output_segment

class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1), # rf 5
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1), # rf 7
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))  # rf 14
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1), # rf 16
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1), # rf 18
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2)) # rf 36
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1), # rf 38
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1), # rf 40
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2)) # rf 80 
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1), # rf 82 
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1), # rf 84
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2)) # rf 168
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1), # rf 170
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1), # 172
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp5 = nn.Sequential(nn.MaxPool2d(2)) # 344
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1), # 346
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1), # 348
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp3(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        mp4 = self.mp4(b4)
        b5 = self.block5(mp4)
        mp5 = self.mp5(b5)
        b6 = self.block6(mp5)
        return b1,b2,b3,b4,b5,b6

class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1,b2,b3,b4,b5,b6):
        up_b = self.up_b(b6)
        cat_b = torch.cat((up_b,b5),dim=1)
        db_b = self.db_b(cat_b)

        up1 = self.up1(db_b)
        cat1 = torch.cat((up1,b4),dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2,b3),dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3,b2),dim=1)
        db3 = self.db3(cat3)

        up4 = self.up4(db3)
        cat4 = torch.cat((up4,b1),dim=1)
        db4 = self.db4(cat4)

        out = self.fin_out(db4)
        return out
    
class Segmentor(nn.Module):

    def __init__(self, in_ch=576, out_ch=2, base_ch=256):
        super(Segmentor, self).__init__()

        self.head = conv_block(in_ch, base_ch // 2, base_ch // 2) # rf 5
        self.down1 = down(base_ch // 2, base_ch) # rf 14
        self.down2 = down(base_ch, base_ch * 2) # rf 32
        self.down3 = down(base_ch * 2, base_ch * 4) # 68

        self.up3 = up(base_ch * 4, base_ch * 2)
        self.up2 = up(base_ch * 2, base_ch)
        self.up1 = up(base_ch, base_ch // 2)

        self.out = nn.Conv2d(base_ch // 2, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x) # 16, 128, 64, 64
        d1 = self.down1(x) # 16, 256, 32, 32
        d2 = self.down2(d1) # 16, 512, 16, 16
        d3 = self.down3(d2) # 16, 1024, 8, 8
        u3 = self.up3(d3, d2) # 16, 512, 16, 16
        u2 = self.up2(u3, d1) # 16, 256, 32, 32
        u1 = self.up1(u2, x) # 16, 128, 64, 64
        out = self.out(u1)
        return out
    
if __name__ == '__main__':
    inputs = torch.randn(16, 576, 64, 64)
    # network = DiscriminativeSubNetwork(in_channels=576, out_channels=2)
    network = Segmentor()
    outputs = network(inputs)
    print(outputs.shape)