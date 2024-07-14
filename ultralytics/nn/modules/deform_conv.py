import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class Deformconv2dModify(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, deformable_groups=1):
        super(Deformconv2dModify, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.deformable_groups = deformable_groups
        # Offset and mask convolution layers
        self.offset_conv = nn.Conv2d(
            in_channels,
            kernel_size * kernel_size * 2 * deformable_groups,
            kernel_size,
            stride,
            padding,
            dilation=dilation
        )
        # Initialize weights
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        self._deform_conv2d = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.default_act = nn.SiLU()  # default activation
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self._deform_conv2d(x, offset)
        return self.default_act(self.bn(x))
