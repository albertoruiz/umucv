import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding='same', stride=1):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        c = F.relu(self.conv1(x))
        c = F.relu(self.conv2(c))
        p = F.max_pool2d(c, 2, 2)
        return c, p



class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, kernel_size=(3, 3), padding='same', stride=1):
        super(UpBlock, self).__init__()
        # Ajusta el padding para emular 'same'
        if padding == 'same':
            padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)  # Concatenar en el canal de características
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding='same', stride=1):
        super(Bottleneck, self).__init__()
        if padding == 'same':
            padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        filters = [16, 32, 64, 128, 256]

        # Down part of UNet
        self.down1 = DownBlock(in_channels, filters[0])
        self.down2 = DownBlock(filters[0], filters[1])
        self.down3 = DownBlock(filters[1], filters[2])
        self.down4 = DownBlock(filters[2], filters[3])

        # Bottleneck
        self.bottleneck = Bottleneck(filters[3], filters[4])

        # Up part of UNet
        self.up1 = UpBlock(filters[4], filters[3], filters[3])
        self.up2 = UpBlock(filters[3], filters[2], filters[2])
        self.up3 = UpBlock(filters[2], filters[1], filters[1])
        self.up4 = UpBlock(filters[1], filters[0], filters[0])

        # Final Convolution
        self.final_conv = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Downsample
        c1, p1 = self.down1(x)
        c2, p2 = self.down2(p1)
        c3, p3 = self.down3(p2)
        c4, p4 = self.down4(p3)

        # Bottleneck
        bn = self.bottleneck(p4)

        # Upsample + concat
        u1 = self.up1(bn, c4)
        u2 = self.up2(u1, c3)
        u3 = self.up3(u2, c2)
        u4 = self.up4(u3, c1)

        # Final layer
        out = self.final_conv(u4)
        return out

