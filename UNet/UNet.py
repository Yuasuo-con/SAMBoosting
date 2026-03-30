import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid_channels = in_channels // 2 if in_channels > out_channels else out_channels // 2
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + in_channels // 2 , out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))
class UNet(nn.Module):
    """
    U-Net model
    """
    def __init__(self, in_channels, out_channels, bilinear=False, features=[64, 128, 256, 512]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = Down(features[0], features[1])
        self.down2 = Down(features[1], features[2])
        self.down3 = Down(features[2], features[3])

        self.up1 = Up(features[3], features[2], bilinear)
        self.up2 = Up(features[2], features[1], bilinear)
        self.up3 = Up(features[1], features[0], bilinear)
        self.outc = nn.Conv2d(features[0], out_channels, kernel_size=1)
        # self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # x1: 64
        # x2: 128
        # x3: 256
        # x4: 512
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        # out = self.softmax(x)
        return x

if __name__ == '__main__':
    model = UNet(3, 1)
    x = torch.randn(1, 3, 512, 512)
    y = model(x)
    print(y.shape)