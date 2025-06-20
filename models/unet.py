# Source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, norm_fn='batch'):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64, norm_fn = norm_fn))
        self.down1 = (Down(64, 128, norm_fn = norm_fn))
        self.down2 = (Down(128, 256, norm_fn = norm_fn))
        self.down3 = (Down(256, 512, norm_fn = norm_fn))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, norm_fn = norm_fn))
        self.up1 = (Up(1024, 512 // factor, bilinear, norm_fn = norm_fn))
        self.up2 = (Up(512, 256 // factor, bilinear, norm_fn = norm_fn))
        self.up3 = (Up(256, 128 // factor, bilinear, norm_fn = norm_fn))
        self.up4 = (Up(128, 64, bilinear, norm_fn = norm_fn))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, scale = 1.0, level = -1):     # x (bs, c, h, w)
        x1 = self.inc(x)      # x (bs, 64, h, w)
        x2 = self.down1(x1)   # x (bs, 128, h/2, w/2)
        x3 = self.down2(x2)   # x (bs, 256, h/4, w/4)
        x4 = self.down3(x3)   # x (bs, 512, h/8, w/8)
        x5 = self.down4(x4)   # x (bs, 1024, h/16, w/16)

        x = self.up1(x5, x4)  # x (bs, 512, h/8, w/8)
        x = self.up2(x, x3)   # x (bs, 256, h/4, w/4)
        x = self.up3(x, x2)   # x (bs, 128, h/2, w/2)
        x = self.up4(x, x1)   # x (bs, 64, h, w)
        logits = self.outc(x)
        return logits

class Up_bottle(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True, norm_fn = 'batch'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_fn=norm_fn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(mid_channels, out_channels, norm_fn=norm_fn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class MLP(nn.Module):
    '''
    Multilayer Perceptron.
  '''

    def __init__(self, input_dim, output_dim=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        '''Forward pass'''
        return self.layers(x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, norm_fn = 'batch'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # if norm_fn == 'batch':
        #     self.norm1 = nn.BatchNorm2d(mid_channels)
        #     self.norm2 = nn.BatchNorm2d(mid_channels)
        # elif norm_fn == 'group':
        #     self.norm1 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        #     self.norm2 = nn.GroupNorm(num_groups=8, num_channels=mid_channels)
        # elif norm_fn == 'instance':
        #     self.norm1 = nn.InstanceNorm2d(mid_channels)
        #     self.norm2 = nn.InstanceNorm2d(mid_channels)

        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
        #     self.norm1,
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     self.norm2,
        #     nn.ReLU(inplace=True)
        # )

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, norm_fn = 'batch'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_fn = norm_fn)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, norm_fn = 'batch'):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, norm_fn = norm_fn)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, norm_fn = norm_fn)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
