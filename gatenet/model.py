from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutC(nn.Module):
    def __init__(self, in_ch: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.conv(x))


class GateNet(nn.Module):
    """
    GateNet from MonoRace paper:
    - U-Net style encoder-decoder
    - 5 output maps {y0..y4} at increasing resolutions
    - each map has 2 channels: foreground and instance boundary
    - deployment uses y4 only (highest resolution)
    """

    def __init__(self, in_channels: int = 3, f: int = 4, out_channels: int = 2) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        c1 = 64 // f
        c2 = 128 // f
        c3 = 256 // f
        c4 = 512 // f

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)
        self.down4 = Down(c4, c4)

        self.up1 = Up(in_ch=c4, skip_ch=c4, out_ch=c3)   # -> 256/f
        self.up2 = Up(in_ch=c3, skip_ch=c3, out_ch=c2)   # -> 128/f
        self.up3 = Up(in_ch=c2, skip_ch=c2, out_ch=c1)   # -> 64/f
        self.up4 = Up(in_ch=c1, skip_ch=c1, out_ch=c1)   # -> 64/f

        self.outc0 = OutC(c4, self.out_channels)  # y0 from deepest features
        self.outc1 = OutC(c3, self.out_channels)
        self.outc2 = OutC(c2, self.out_channels)
        self.outc3 = OutC(c1, self.out_channels)
        self.outc4 = OutC(c1, self.out_channels)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        y0 = self.outc0(x5)

        u1 = self.up1(x5, x4)
        y1 = self.outc1(u1)

        u2 = self.up2(u1, x3)
        y2 = self.outc2(u2)

        u3 = self.up3(u2, x2)
        y3 = self.outc3(u3)

        u4 = self.up4(u3, x1)
        y4 = self.outc4(u4)

        return y0, y1, y2, y3, y4

