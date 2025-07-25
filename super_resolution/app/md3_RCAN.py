# RCAN	채널 주의력 (attention) 사용, 최고 성능급	⭐⭐⭐⭐ 90,000+쌍 확보 가능

import torch
import torch.nn as nn

# Channel Attention Block
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return x * y


# Residual Channel Attention Block
class RCAB(nn.Module):
    def __init__(self, channel):
        super(RCAB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, padding=1),
            CALayer(channel)
        )

    def forward(self, x):
        res = self.body(x)
        return res + x


# Residual Group: 여러 RCAB로 구성
class ResidualGroup(nn.Module):
    def __init__(self, channel, num_rcab):
        super(ResidualGroup, self).__init__()
        blocks = [RCAB(channel) for _ in range(num_rcab)]
        blocks.append(nn.Conv2d(channel, channel, 3, padding=1))  # Skip 연결
        self.body = nn.Sequential(*blocks)

    def forward(self, x):
        res = self.body(x)
        return res + x


# 전체 RCAN 모델
class RCAN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=64, num_groups=10, num_rcab=20):
        super(RCAN, self).__init__()
        self.sub_mean = nn.Conv2d(3, 3, 1)  # 입력 정규화 여부 선택적 처리
        self.head = nn.Conv2d(3, num_channels, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualGroup(num_channels, num_rcab) for _ in range(num_groups)],
            nn.Conv2d(num_channels, num_channels, 3, padding=1)
        )

        # 업샘플링
        self.upscale = nn.Sequential(
            nn.Conv2d(num_channels, num_channels * (scale_factor ** 2), 3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.Conv2d(num_channels, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        out = self.upscale(res)
        return out
