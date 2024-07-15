#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/10 14:10
# @Author  : 作者名
# @File    : attention.py
# @Description  : 基础注意力相关代码
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AttentionConcat(nn.Module):
    # use attention before concat
    def __init__(self, dimension=1, channel=512, att=None):
        super().__init__()
        self.d = dimension
        self.c = channel
        self.att = att(self.c)

    def forward(self, x):
        return torch.cat([x[0], self.att(x[1])], self.d)



class AttentionConcat_DFPN_column_merge(nn.Module):
    # use attention before concat
    def __init__(self, dimension=1, channel=512, att=None):
        super().__init__()
        self.d = dimension
        self.c = channel
        self.att = att(self.c)


    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        target_height, target_width = x[0].shape[2], x[0].shape[3]
        attention_out = torch.cat([x[0], self.att(x[1])], self.d)
        x = x[2:]
        x.append(attention_out)
        x = [nn.functional.interpolate(tensor, size=(target_height, target_width), mode='bilinear', align_corners=False) if idx == 0 else tensor for idx, tensor in enumerate(x)]
        return torch.cat(x, self.d)



class ChannelAttention(nn.Module):
    # Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)  # O = (I - K + 2*P) / S + 1 所以110的卷积参数组合保持分辨率不变
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    # Spatial-attention module
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # O = (I - K + 2*P) / S + 1 分辨率保持不变
        self.act = nn.Sigmoid()

    def forward(self, x):
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    # Convolutional Block Attention Module 过CBAM的通道数和分辨率都是不变的
    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        return self.spatial_attention(self.channel_attention(x))


class SEBlock(nn.Module):
    def __init__(self, channels, ratio=4):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.global_pooling = self.max_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=channels, out_features=channels // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=channels // ratio, out_features=channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.global_pooling(x).view(b, c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x * v


# Efficient Channel Attention module
class ECA(nn.Module):
    def __init__(self, in_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        k = int(abs((math.log(in_channel, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1
        padding = kernel_size // 2
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.pool(x)
        out = out.view(x.size(0), 1, x.size(1))
        out = self.conv(out)
        out = out.view(x.size(0), x.size(1), 1, 1)
        return out * x
