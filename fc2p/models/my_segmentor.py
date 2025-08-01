import torch
import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out, k=3, stride=1, padding=1):
        super(conv_block, self).__init__()
        if k == 3:
            self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride, padding=padding),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
        else:
             self.conv = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(ch_out),
                nn.ReLU(inplace=True),
            )
 
    def forward(self, x):
        return self.conv(x)
    
class up(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up, self).__init__()

        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_block(ch_in, ch_out, 3)
        )
        self.conv = conv_block(ch_out * 2, ch_out, 3)

    def forward(self, x1, x2):
        x1 = self.up_block(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)

class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False), 
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
            b, c, h, w = x.size()
            y = self.gap(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y.expand_as(x)

class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.mp2 = nn.MaxPool2d(2)
        self.mp4 = nn.MaxPool2d(4)

        self.conv64 = conv_block(96, 128, k=3)
        self.conv32 = conv_block(192 + 128, 256, k=3)
        self.conv16 = conv_block(384 + 256, 512, k=3)
        self.conv8 = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(512, 768, 3),
        )

    def forward(self, diff):
        # 96,64,64 192,64,64 384,64,64
        diff_64, diff_32, diff_16 = torch.split(diff, [96, 192, 384], dim=1)

        diff_64 = self.conv64(diff_64) # 128,64,64

        diff_32 = torch.cat((self.mp2(diff_64), self.mp2(diff_32)), dim=1)
        diff_32 = self.conv32(diff_32) # 256,32,32

        diff_16 = torch.cat((self.mp2(diff_32), self.mp4(diff_16)), dim=1)
        diff_16 = self.conv16(diff_16) # 512,16,16

        diff_8 = self.conv8(diff_16) # 768,8,8

        return diff_64, diff_32, diff_16, diff_8
    
class fusion_block(nn.Module):

    def __init__(self, in_channels):
        super(fusion_block, self).__init__()
        self.conv = conv_block(in_channels * 2, in_channels, 3)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.conv(x)
        return out
    
class AnomalySeg(nn.Module):

    def __init__(self, num_classes):
        super(AnomalySeg, self).__init__()
        self.encoder = Encoder()

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.fusion_in = fusion_block(672)
        self.conv_in = conv_block(672, 128, 3) # ! 128

        self.fusion8 = fusion_block(768)
        self.fusion16 = fusion_block(512)
        self.fusion32 = fusion_block(256)
        self.fusion64 = fusion_block(128)

        self.skip_conv_32 = conv_block(512+256, 256, 3)
        self.skip_conv_64 = conv_block(256+128+128, 128, 3) # ! 128

        self.decoder8 = up(768, 768)
        self.decoder16 = up(768, 512)
        self.decoder32 = up(512, 256)
        self.decoder64 = up(256, 128)

        self.pred = nn.Sequential(
            SE_Block(128),
            # nn.Conv2d(128, 64, 3, 1, 1),
            # nn.ReLU(inplace=True), # !
            # nn.BatchNorm2d(64),
            conv_block(128, 64),
            nn.Conv2d(64, num_classes, 3, 1, 1)
        )

    def forward(self, diff_0, diff_1):

        diff_in = self.fusion_in(diff_0, diff_1)
        diff_in = self.conv_in(diff_in)

        feat_0 = self.encoder(diff_0)
        feat_1 = self.encoder(diff_1)

        latent = self.fusion8(feat_0[3], feat_1[3]) # 768,8,8

        skip_16 = self.fusion16(feat_0[2], feat_1[2]) # 512,16,16

        skip_32 = torch.cat((self.up2(skip_16), self.fusion32(feat_0[1], feat_1[1])), dim=1)
        skip_32 = self.skip_conv_32(skip_32) # 256,32,32

        skip_64 = torch.cat([diff_in, self.up2(skip_32), self.fusion64(feat_0[0], feat_1[0])], dim=1)
        skip_64 = self.skip_conv_64(skip_64) # 128,64,64

        decoder16 = self.decoder16(latent, skip_16)
        decoder32 = self.decoder32(decoder16, skip_32)
        decoder64 = self.decoder64(decoder32, skip_64)

        out = self.pred(decoder64)

        return out

if __name__ == '__main__':
    x1 = torch.randn(4, 672, 64, 64)
    x2 = torch.randn(4, 672, 64, 64)
    # net = conv_block(576, 256, 256)
    # net = Encoder()
    net = AnomalySeg(2)
    outputs = net(x1, x2)
    for i in outputs:
        print(i.shape)
    # print(outputs.shape)

