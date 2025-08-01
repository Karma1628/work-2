import torch
import torch.nn as nn

import torch
import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class FeatureReconstructor(nn.Module):

    def __init__(self, ch_in, ch_base):
        super(FeatureReconstructor, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(ch_in, ch_base // 2),
            self._make_layer(ch_base // 2, ch_base, 'down'),
            self._make_layer(ch_base, ch_base * 2, 'down'),
            self._make_layer(ch_base * 2, ch_base * 4, 'down'),
            self._make_layer(ch_base * 4, ch_base * 4, 'down')
        )

        self.decoder = nn.Sequential(
            self._make_layer(ch_base * 4, ch_base * 4, 'up'),
            self._make_layer(ch_base * 4, ch_base * 2, 'up'),
            self._make_layer(ch_base * 2, ch_base, 'up'),
            self._make_layer(ch_base, ch_base // 2, 'up'),
            conv_block(ch_base// 2, ch_in),
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1)
        )

    def _make_layer(self, ch_in, ch_out, mode='down'):
        if mode == 'down':
            return nn.Sequential(
                nn.MaxPool2d(2),
                conv_block(ch_in, ch_out)
                )
        elif mode == 'up':
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                conv_block(ch_in, ch_out)
                )

    def forward(self, x):
        latent = self.encoder(x)
        recon = self.decoder(latent)
        return recon
    
    
if __name__ == '__main__':
    x = torch.randn(4, 1344, 64, 64).cuda()
    net = FeatureReconstructor(1344, 256).cuda()
    y = net(x)
    print(y.shape)