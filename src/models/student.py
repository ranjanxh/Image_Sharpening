"""Student model: lightweight U-Net with Squeeze-and-Excitation blocks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import StudentConfig


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        reduced = max(1, channel // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def _conv_block(in_c: int, out_c: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
        nn.ReLU(inplace=True),
        SEBlock(out_c),
    )


class StudentModel(nn.Module):
    """4-level encoder/decoder U-Net with SE blocks in every conv block and
    skip connections; returns the sharpened output plus the e2/e4/bottleneck
    intermediate features used for feature-distillation loss."""

    def __init__(self, in_channels: int, out_channels: int, base_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc1 = _conv_block(in_channels, base_channels)
        self.enc2 = _conv_block(base_channels, base_channels * 2)
        self.enc3 = _conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = _conv_block(base_channels * 4, base_channels * 8)

        self.bottleneck = _conv_block(base_channels * 8, base_channels * 16)

        self.upconv4 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.dec4 = _conv_block(base_channels * 16, base_channels * 8)

        self.upconv3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = _conv_block(base_channels * 8, base_channels * 4)

        self.upconv2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = _conv_block(base_channels * 4, base_channels * 2)

        self.upconv1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _conv_block(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    @staticmethod
    def _align(upsampled: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = F.interpolate(upsampled, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return upsampled

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)

        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        b = self.bottleneck(p4)

        d4 = self._align(self.upconv4(b), e4)
        d4 = self.dec4(torch.cat((d4, e4), dim=1))

        d3 = self._align(self.upconv3(d4), e3)
        d3 = self.dec3(torch.cat((d3, e3), dim=1))

        d2 = self._align(self.upconv2(d3), e2)
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self._align(self.upconv1(d2), e1)
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        output = self.out_conv(d1)
        return output, e2, e4, b


def build_student(student_config: StudentConfig) -> StudentModel:
    return StudentModel(
        in_channels=student_config.in_channels,
        out_channels=student_config.out_channels,
        base_channels=student_config.base_channels,
    )
