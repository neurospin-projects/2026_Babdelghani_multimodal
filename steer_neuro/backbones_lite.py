"""Plain nn.Module copies of the ConvNet / ProjectionHead classes from
DeepLearning_Tracto/backbones.py, with the `lightning` dependency stripped out.

Both Champollion and ALMA checkpoints were trained with the LightningModule
wrappers (LitConvNet) from that file, but `import lightning` alone costs
~230s on this filesystem and buys nothing here: STEER only ever needs the
frozen encoder's plain forward pass, never the Lightning training wrappers.
state_dicts saved from `LitConvNet.encoder` / `LitConvNet.projection_head`
load into these classes unmodified (verified against both checkpoints).

Kept byte-for-byte equivalent to DeepLearning_Tracto/backbones.py's ConvNet /
ProjectionHead / Conv3dSame / ComputeOutputDim — do not let the two drift;
if backbones.py changes, port the change here too.
"""
from collections import OrderedDict
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def ComputeOutputDim(dimension, depth):
    """Compute the spatial output size after `depth` stride-2 downsampling steps."""
    if depth == 0:
        return dimension
    else:
        return ComputeOutputDim(dimension // 2 + dimension % 2, depth - 1)


class Conv3dSame(nn.Conv3d):
    """Conv3d with 'SAME' padding (output spatial size = ceil(input / stride))."""

    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((np.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: Tensor) -> Tensor:
        ih, iw, id_ = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=id_, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(x, [int(pad_d // 2), int(pad_d - pad_d // 2),
                          int(pad_w // 2), int(pad_w - pad_w // 2),
                          int(pad_h // 2), int(pad_h - pad_h // 2)])
        return F.conv3d(x, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvNet(nn.Module):
    """3D convolutional encoder for contrastive learning (see backbones.py for full docstring)."""

    def __init__(self, in_channels=1, encoder_depth=None, block_depth=None,
                 num_representation_features=None, linear=True,
                 adaptive_pooling=None, filters=None,
                 initial_kernel_size=None, initial_stride=None, max_pool=False,
                 drop_rate=None, memory_efficient=False, in_shape=None):
        super(ConvNet, self).__init__()

        self.num_representation_features = num_representation_features
        self.drop_rate        = drop_rate
        self.in_shape         = in_shape
        c, h, w, d            = in_shape
        self.encoder_depth    = encoder_depth
        self.filters          = filters
        self.block_depth      = block_depth
        self.initial_kernel_size = initial_kernel_size
        self.initial_stride   = initial_stride
        self.max_pool         = max_pool

        assert len(self.filters) >= encoder_depth, "Incomplete filters list given."

        if adaptive_pooling is None:
            h0 = math.ceil(h / initial_stride)
            w0 = math.ceil(w / initial_stride)
            d0 = math.ceil(d / initial_stride)
            self.z_dim_h = ComputeOutputDim(h0, self.encoder_depth)
            self.z_dim_w = ComputeOutputDim(w0, self.encoder_depth)
            self.z_dim_d = ComputeOutputDim(d0, self.encoder_depth)
            self.out_dim = self.z_dim_h * self.z_dim_w * self.z_dim_d
        else:
            self.out_dim = int(np.prod(adaptive_pooling[1]))

        modules_encoder = []
        layer_name  = ['', 'a', 'b', 'c', 'd', 'e']
        out_channels = None
        for step in range(encoder_depth):
            for depth in range(block_depth - 1):
                name        = layer_name[depth]
                in_ch       = 1 if (step == 0 and depth == 0) else out_channels
                kernel_size = initial_kernel_size if (step == 0 and depth == 0) else 3
                stride      = initial_stride if (step == 0 and depth == 0) else 1
                out_channels = filters[step]
                modules_encoder.append((f'conv{step}{name}',
                    nn.Conv3d(in_ch, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)))
                modules_encoder.append((f'norm{step}{name}',  nn.BatchNorm3d(out_channels)))
                modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
                if self.max_pool and step == 0 and depth == 0:
                    modules_encoder.append(('MaxPool', nn.MaxPool3d((2, 2, 2))))
                modules_encoder.append((f'DropOut{step}{name}', nn.Dropout3d(p=drop_rate)))
            name = layer_name[block_depth - 1]
            modules_encoder.append((f'conv{step}{name}',
                Conv3dSame(in_channels=out_channels, out_channels=out_channels,
                           kernel_size=(3, 3, 3), stride=(2, 2, 2),
                           groups=1, bias=True)))
            modules_encoder.append((f'norm{step}{name}',     nn.BatchNorm3d(out_channels)))
            modules_encoder.append((f'LeakyReLU{step}{name}', nn.LeakyReLU()))
            modules_encoder.append((f'DropOut{step}{name}',   nn.Dropout3d(p=drop_rate)))
            self.num_features = out_channels

        if adaptive_pooling is not None:
            if adaptive_pooling[0] == 'max':
                modules_encoder.append(
                    ('AdaptiveMaxPool', nn.AdaptiveMaxPool3d(output_size=adaptive_pooling[1])))
            elif adaptive_pooling[0] == 'average':
                modules_encoder.append(
                    ('AdaptiveAvgPool', nn.AdaptiveAvgPool3d(output_size=adaptive_pooling[1])))
            else:
                raise ValueError("adaptive_pooling mode must be 'max' or 'average'")

        modules_encoder.append(('Flatten', nn.Flatten()))
        if linear:
            modules_encoder.append(('Linear',
                nn.Linear(self.num_features * self.out_dim,
                          num_representation_features)))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

    def forward(self, x):
        return self.encoder(x)


class ProjectionHead(nn.Module):
    """Flexible MLP projection head."""

    def __init__(self, num_representation_features=256,
                 layers_shapes=[256, 10],
                 activation='relu',
                 drop_rate=0.0):
        super(ProjectionHead, self).__init__()
        self.num_representation_features = num_representation_features

        layers     = []
        input_size = layers_shapes[0]
        for i, dim_i in enumerate(layers_shapes[1:]):
            output_size = dim_i
            layers.append((f'Linear{i}', nn.Linear(input_size, output_size)))
            if i < len(layers_shapes) - 2:
                if activation == 'linear':
                    pass
                elif activation == 'relu':
                    layers.append((f'LeakyReLU{i}', nn.LeakyReLU()))
                elif activation == 'sigmoid':
                    layers.append((f'Sigmoid{i}', nn.Sigmoid()))
                else:
                    raise ValueError(
                        f"Unknown activation '{activation}'. "
                        "Choose 'linear', 'relu', or 'sigmoid'.")
                layers.append((f'DropOut{i}', nn.Dropout(p=drop_rate)))
            input_size = output_size

        self.layers = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.layers(x)
