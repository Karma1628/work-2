import torch
import torch.nn as nn

class conv_block(nn.Module):

    def __init__(self, ch_in, ch_out, ch_mid=None):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_mid, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class down(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(down, self).__init__()

        self.down_block = nn.Sequential(
            nn.MaxPool2d(2),
            conv_block(ch_in, ch_out, ch_out)
        )
    
    def forward(self, x):
        return self.down_block(x)
    
class up(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(up, self).__init__()

        self.up_block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
        self.conv = conv_block(ch_out * 2, ch_out, ch_out)

    def forward(self, x1, x2):
        x1 = self.up_block(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)

class hor_block(nn.Module):

    def __init__(self, ch_in, ch_out):
        super(hor_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            gnconv(ch_out),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)
    
class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s
        # print('[gnconv]', order, 'order with dims=', self.dims, 'scale=%.4f'%self.scale)

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x) # 4, 1152, 64, 64
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1) # 4,36,64,64  4,1116,64,64

        dw_abc = self.dwconv(abc) * self.scale # 4, 1116, 64, 64

        dw_list = torch.split(dw_abc, self.dims, dim=1) # 36 72 144 288 576
        x = pwa * dw_list[0] # 4,36,64,64 

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x


if __name__ == '__main__':
    inputs = torch.randn(4, 576, 64, 64)
    # net = conv_block(576, 256, 256)
    net = gnconv(dim=576)
    outputs = net(inputs)
    print(outputs.shape)