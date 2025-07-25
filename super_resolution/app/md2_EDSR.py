# EDSR	성능 우수, Residual Block 사용	⭐⭐⭐ 18,000~45,000쌍 (36, 7, 72)

import torch
import torch.nn as nn

class MeanShift(nn.Conv2d):
    """이미지 정규화/비정규화에 사용되는 클래스 (DIV2K 용 기준 평균/표준편차 사용)"""
    def __init__(self, sign=-1):
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.bias.data = sign * torch.Tensor(rgb_mean)
        self.weight.requires_grad = False
        self.bias.requires_grad = False


class ResidualBlock(nn.Module):
    """EDSR의 기본 블록"""
    def __init__(self, num_feats):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feats, num_feats, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Upsampler(nn.Sequential):
    """PixelShuffle 방식 업샘플러"""
    def __init__(self, scale, n_feats):
        m = []
        if scale == 2 or scale == 3:
            m.append(nn.Conv2d(n_feats, n_feats * scale * scale, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(scale))
        elif scale == 4:
            m.append(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(2))
            m.append(nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1))
            m.append(nn.PixelShuffle(2))
        else:
            raise NotImplementedError
        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    def __init__(self, scale_factor=3, n_resblocks=16, n_feats=64):
        super(EDSR, self).__init__()
        self.sub_mean = MeanShift(sign=-1)
        self.add_mean = MeanShift(sign=1)

        # Head
        self.head = nn.Conv2d(3, n_feats, kernel_size=3, padding=1)

        # Body
        body = [ResidualBlock(n_feats) for _ in range(n_resblocks)]
        body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1))
        self.body = nn.Sequential(*body)

        # Tail
        self.tail = nn.Sequential(
            Upsampler(scale_factor, n_feats),
            nn.Conv2d(n_feats, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)
        return x
