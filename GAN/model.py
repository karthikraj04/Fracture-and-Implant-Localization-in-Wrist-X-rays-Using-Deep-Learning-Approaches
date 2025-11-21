import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        def down(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up(in_feat, out_feat, dropout=False):
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_feat, out_feat, 3, padding=1),
                nn.BatchNorm2d(out_feat),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.enc1 = down(in_channels, 64, normalize=False)
        self.enc2 = down(64, 128)
        self.enc3 = down(128, 256)
        self.enc4 = down(256, 512)
        self.enc5 = down(512, 512)
        self.enc6 = down(512, 512)
        self.enc7 = down(512, 512)
        self.enc8 = down(512, 512, normalize=False)

        self.middle = ResidualBlock(512)

        self.dec1 = up(512, 512, dropout=True)     # + enc7 → 1024
        self.dec2 = up(1024, 512, dropout=True)    # + enc6
        self.dec3 = up(1024, 512, dropout=True)    # + enc5
        self.dec4 = up(1024, 512)                  # + enc4
        self.dec5 = up(1024, 256)                  # + enc3
        self.dec6 = up(512, 128)                   # + enc2
        self.dec7 = up(256, 64)                    # + enc1
        self.dec8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        m = self.middle(e8)

        d1 = self.dec1(m)
        d2 = self.dec2(torch.cat([d1, e7], dim=1))
        d3 = self.dec3(torch.cat([d2, e6], dim=1))
        d4 = self.dec4(torch.cat([d3, e5], dim=1))
        d5 = self.dec5(torch.cat([d4, e4], dim=1))
        d6 = self.dec6(torch.cat([d5, e3], dim=1))
        d7 = self.dec7(torch.cat([d6, e2], dim=1))
        out = self.dec8(torch.cat([d7, e1], dim=1))
        return out

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=2):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, 4, padding=1)  # Output 1-channel patch matrix
        )

    def forward(self, img, masked_img):
        x = torch.cat((img, masked_img), dim=1)
        return self.model(x)
