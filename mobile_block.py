import re
from typing import Callable, Type, TypeVar

import torch
import torch.nn as nn

__all__ = [
    'BlockFactory',
    'InplaceReLU',
    'MobileBlock',
    'ShuffleBlock',
    'SkipBlock'
]


class ShuffleBlock(nn.Module):
    """
    shuffle channels
    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices, https://arxiv.org/abs/1707.01083
    """

    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel shuffle: [n, c, h, w] -> [n, g, c/g, h, w] -> [n, c/g, g, h, w] -> [n, c, h, w]
        n, c, h, w = x.size()
        g = self.groups
        return x.view(n, g, c // g, h, w).transpose(1, 2).contiguous().view(n, c, h, w)


class InplaceReLU(nn.ReLU):
    def __init__(self):
        super().__init__(inplace=True)


class MobileBlock(torch.nn.Module):
    """
    MobileNet-style base block
    MobileNetV2: Inverted Residuals and Linear Bottlenecks, https://arxiv.org/abs/1801.04381
    Pixel-wise (shuffle) -> Depth-wise -> Pixel-wise
    """

    def __init__(self,
                 input_size: int,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expansion: int = 1,
                 kernel: int = 3,
                 groups: int = 1,
                 batch_norm_2d: Type[torch.nn.BatchNorm2d] = torch.nn.BatchNorm2d,
                 relu: Callable[[], nn.Module] = InplaceReLU,
                 residual: bool = False
                 ):
        super().__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.expansion = expansion
        self.kernel = kernel
        self.groups = groups
        self.residual = residual and stride == 1 and in_channels == out_channels

        inner_channels = in_channels * expansion
        self.block = nn.Sequential(
            # pixel wise
            nn.Conv2d(in_channels=in_channels, out_channels=inner_channels, kernel_size=1, groups=groups, bias=False),
            batch_norm_2d(num_features=inner_channels),
            relu(),
            ShuffleBlock(groups=groups) if groups > 1 else nn.Sequential(),
            # depth wise
            nn.Conv2d(in_channels=inner_channels, out_channels=inner_channels, groups=inner_channels,
                      kernel_size=kernel, stride=stride, padding=kernel // 2, bias=False),
            batch_norm_2d(num_features=inner_channels),
            relu(),
            # pixel wise
            nn.Conv2d(in_channels=inner_channels, out_channels=out_channels, kernel_size=1, groups=groups, bias=False),
            batch_norm_2d(num_features=out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            return self.block(x) + x
        else:
            return self.block(x)

    @property
    def block_id(self) -> str:
        """
        unique block id [width_in_out_stride_expansion_kernel_groups]
        :return: block id in string
        """
        return '_'.join([
            f'w{self.input_size}',
            f'i{self.in_channels}',
            f'o{self.out_channels}',
            f's{self.stride}',
            f'e{self.expansion}',
            f'k{self.kernel}',
            f'g{self.groups}'
        ])

    @classmethod
    def factory(cls, block_id: str,
                batch_norm_2d: Type[torch.nn.BatchNorm2d] = torch.nn.BatchNorm2d,
                relu: Callable[[], nn.Module] = InplaceReLU,
                residual: bool = False):
        parse = re.findall(r'w(\d+)_i(\d+)_o(\d+)_s(\d+)_e(\d+)_k(\d+)_g(\d+)', block_id.lower())
        if not parse:
            raise ValueError(f'ParseError: {block_id}')
        input_size, in_channels, out_channels, stride, expansion, kernel, groups = (int(value) for value in parse[0])

        return cls(input_size=input_size, in_channels=in_channels, out_channels=out_channels,
                   stride=stride, expansion=expansion, kernel=kernel, groups=groups,
                   batch_norm_2d=batch_norm_2d, relu=relu, residual=residual)


class SkipBlock(nn.Module):
    def __init__(self,
                 input_size: int,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 batch_norm_2d: Type[nn.BatchNorm2d] = nn.BatchNorm2d):
        super().__init__()

        self.input_size = input_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if stride == 1 and in_channels == out_channels:
            # keep same tensor
            self.block = nn.Sequential()
        else:
            # down sample
            self.block = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                          kernel_size=stride, stride=stride, bias=False),
                batch_norm_2d(out_channels)
            )

    def forward(self, x):
        return self.block(x)


class BlockFactory:
    def __init__(self,
                 skip: bool = False,
                 expansion: int = 1,
                 kernel: int = 3,
                 groups: int = 1,
                 mobile_block_func=MobileBlock,
                 skip_block_func=SkipBlock
                 ):
        self.skip = skip
        self.expansion = expansion
        self.kernel = kernel
        self.groups = groups
        self.mobile_block_func = mobile_block_func
        self.skip_block_func = skip_block_func

    @property
    def block_name(self) -> str:
        if self.skip:
            return 'SkipBlock'
        else:
            return f'K{self.kernel}E{self.expansion}' + (f'G{self.groups}' if self.groups > 1 else '') + 'Block'

    def __call__(self, input_size: int, in_channels: int, out_channels: int, stride: int, residual: bool = False):
        if self.skip:
            return self.skip_block_func(
                input_size=input_size, in_channels=in_channels, out_channels=out_channels, stride=stride)
        else:
            return self.mobile_block_func(
                input_size=input_size, in_channels=in_channels, out_channels=out_channels,
                stride=stride, expansion=self.expansion, kernel=self.kernel, groups=self.groups, residual=residual)

    @classmethod
    def factory(cls, block_name: str,
                mobile_block_func=MobileBlock,
                skip_block_func=SkipBlock
                ) -> 'BlockFactory':
        if block_name == 'SkipBlock':
            return BlockFactory(skip=True, mobile_block_func=mobile_block_func, skip_block_func=skip_block_func)
        else:
            parse = re.findall(r'K(\d+)E(\d+)G?(\d+)?', block_name.upper())
            if not parse:
                raise ValueError(f'ParseError: {block_name}')
            kernel, expansion, groups = (int(value or '1') for value in parse[0])
            return BlockFactory(expansion=expansion, kernel=kernel, groups=groups,
                                mobile_block_func=mobile_block_func, skip_block_func=skip_block_func)
