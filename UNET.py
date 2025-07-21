import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class BasicUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=3):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = ConvBlock(128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)         # [B, 64, 512, 512]
        p1 = self.pool1(e1)       # [B, 64, 256, 256]

        e2 = self.enc2(p1)        # [B, 128, 256, 256]
        p2 = self.pool2(e2)       # [B, 128, 128, 128]

        m = self.middle(p2)       # [B, 256, 128, 128]

        u2 = self.up2(m)          # [B, 128, 256, 256]
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = self.up1(d2)         # [B, 64, 512, 512]
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        out = self.final(d1)      # [B, 3, 512, 512]
        return torch.sigmoid(out)
