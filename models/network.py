import torch
from torch import nn
from .extractor import UnetExtractor, ResidualBlock

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, padding=padding, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            # nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, patch_size=32):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 1, kernel_size=3, stride=2),
        )

        self.out = nn.Linear(int(patch_size / 2) ** 2, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.model(x)
        x = x.reshape(batch_size, -1)
        x = self.out(x)

        return x


class Regresser2(nn.Module):
    def __init__(self, cfg, rgb_dim=3, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim

        self.img_encoder = UnetExtractor(in_channel=rgb_dim, encoder_dim=cfg.raft.encoder_dims)


        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(self.decoder_dims[0]+rgb_dim, self.head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.u_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
        )

    def forward(self, img):
        img_feat1, img_feat2, img_feat3 = self.img_encoder(img)

        up3 = self.decoder3(img_feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, img_feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, img_feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        u_out = self.u_head(out)

        return u_out
