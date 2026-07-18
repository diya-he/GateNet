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


class SeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SeparableDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SeparableConv(in_ch, out_ch),
            SeparableConv(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_conv_block(in_ch: int, out_ch: int, conv_kind: str = "standard") -> nn.Module:
    kind = str(conv_kind).strip().lower()
    if kind == "standard":
        return DoubleConv(in_ch, out_ch)
    if kind == "separable":
        return SeparableDoubleConv(in_ch, out_ch)
    raise ValueError("conv_kind must be 'standard' or 'separable'")


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, conv_kind: str = "standard") -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = make_conv_block(in_ch, out_ch, conv_kind=conv_kind)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, conv_kind: str = "standard", up_kind: str = "transpose") -> None:
        super().__init__()
        self.up_kind = str(up_kind).strip().lower()
        if self.up_kind == "transpose":
            self.up = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        elif self.up_kind == "bilinear":
            self.up = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        else:
            raise ValueError("up_kind must be 'transpose' or 'bilinear'")
        self.conv = make_conv_block(out_ch + skip_ch, out_ch, conv_kind=conv_kind)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.up_kind == "bilinear":
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = self.up(x)
        else:
            x = self.up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutC(nn.Module):
    def __init__(self, in_ch: int, out_channels: int, regression_channels: int = 0) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.regression_channels = max(0, int(regression_channels))
        if self.regression_channels > self.out_channels:
            raise ValueError("regression_channels cannot exceed out_channels")
        self.conv = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.regression_channels == 0:
            return torch.sigmoid(x)
        split = self.out_channels - self.regression_channels
        probs = torch.sigmoid(x[:, :split])
        regs = torch.tanh(x[:, split:])
        return torch.cat([probs, regs], dim=1)


class GateNet(nn.Module):
    """
    GateNet from MonoRace paper:
    - U-Net style encoder-decoder
    - 5 output maps {y0..y4} at increasing resolutions
    - output head can be widened for instance/category cues while keeping the
      lightweight backbone unchanged
    - deployment uses y4 only (highest resolution)
    """

    def __init__(
        self,
        in_channels: int = 3,
        f: int = 4,
        out_channels: int = 2,
        regression_channels: int = 0,
        conv_kind: str = "standard",
        up_kind: str = "transpose",
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.regression_channels = max(0, int(regression_channels))
        self.conv_kind = str(conv_kind).strip().lower()
        self.up_kind = str(up_kind).strip().lower()
        c1 = 64 // f
        c2 = 128 // f
        c3 = 256 // f
        c4 = 512 // f

        self.inc = make_conv_block(in_channels, c1, conv_kind=self.conv_kind)
        self.down1 = Down(c1, c2, conv_kind=self.conv_kind)
        self.down2 = Down(c2, c3, conv_kind=self.conv_kind)
        self.down3 = Down(c3, c4, conv_kind=self.conv_kind)
        self.down4 = Down(c4, c4, conv_kind=self.conv_kind)

        self.up1 = Up(in_ch=c4, skip_ch=c4, out_ch=c3, conv_kind=self.conv_kind, up_kind=self.up_kind)   # -> 256/f
        self.up2 = Up(in_ch=c3, skip_ch=c3, out_ch=c2, conv_kind=self.conv_kind, up_kind=self.up_kind)   # -> 128/f
        self.up3 = Up(in_ch=c2, skip_ch=c2, out_ch=c1, conv_kind=self.conv_kind, up_kind=self.up_kind)   # -> 64/f
        self.up4 = Up(in_ch=c1, skip_ch=c1, out_ch=c1, conv_kind=self.conv_kind, up_kind=self.up_kind)   # -> 64/f

        self.outc0 = OutC(c4, self.out_channels, self.regression_channels)  # y0 from deepest features
        self.outc1 = OutC(c3, self.out_channels, self.regression_channels)
        self.outc2 = OutC(c2, self.out_channels, self.regression_channels)
        self.outc3 = OutC(c1, self.out_channels, self.regression_channels)
        self.outc4 = OutC(c1, self.out_channels, self.regression_channels)

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
