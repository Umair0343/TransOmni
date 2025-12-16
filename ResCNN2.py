"""
TransOmni - 2D ResNet Backbone
Adapted from TransDoDNet's ResCNN2.py (3D) for 2D histopathology images

Key changes from TransDoDNet:
- All Conv3d -> Conv2d
- All BatchNorm3d/InstanceNorm3d -> BatchNorm2d/InstanceNorm2d
- All MaxPool3d -> MaxPool2d
- All avg_pool3d -> avg_pool2d
- Kernel sizes: (3,3,3) -> (3,3), (2,2,2) -> (2,2)
- Strides: (2,2,2) -> (2,2), (1,1,1) -> (1,1)
- Input channels: 1 (CT) -> 3 (RGB)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


class Conv2d_wd(nn.Conv2d):
    """Conv2d with weight standardization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), 
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=False):
        super(Conv2d_wd, self).__init__(in_channels, out_channels, kernel_size, 
                                        stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, kernel_size, stride=(1, 1), padding=(0, 0), 
            dilation=(1, 1), bias=False, weight_std=False):
    """3x3 convolution with padding."""
    if weight_std:
        return Conv2d_wd(in_planes, out_planes, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, 
                        stride=stride, padding=padding, dilation=dilation, bias=bias)


def downsample_basic_block(x, planes, stride):
    """Downsample block for residual connections."""
    out = F.avg_pool2d(x, kernel_size=1, stride=stride)
    zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2), out.size(3))
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads.cuda()], dim=1))
    return out


def Norm_layer(norm_cfg, inplanes):
    """Create normalization layer based on config."""
    if norm_cfg == 'BN':
        out = nn.BatchNorm2d(inplanes)
    elif norm_cfg == 'SyncBN':
        out = nn.SyncBatchNorm(inplanes)
    elif norm_cfg == 'GN':
        out = nn.GroupNorm(16, inplanes)
    elif norm_cfg == 'IN':
        out = nn.InstanceNorm2d(inplanes, affine=True)
    return out


def Activation_layer(activation_cfg, inplace=True):
    """Create activation layer based on config."""
    if activation_cfg == 'relu':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)
    return out


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, 
                 stride=(1, 1), downsample=None, weight_std=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=3, stride=stride, 
                            padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, stride=(1, 1), 
                            padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for deeper ResNets."""
    expansion = 4

    def __init__(self, inplanes, planes, norm_cfg, activation_cfg, 
                 stride=(1, 1), downsample=None, weight_std=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, planes)
        self.conv2 = conv3x3(planes, planes, kernel_size=3, stride=stride, 
                            padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, planes)
        self.conv3 = conv3x3(planes, planes * 4, kernel_size=1, bias=False, weight_std=weight_std)
        self.norm3 = Norm_layer(norm_cfg, planes * 4)
        self.nonlin = Activation_layer(activation_cfg, inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.nonlin(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.nonlin(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.nonlin(out)

        return out


class ResNet(nn.Module):
    """2D ResNet backbone for feature extraction.
    
    Adapted from 3D version for 2D histopathology images.
    """

    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
        200: (Bottleneck, (3, 24, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,  # RGB images instead of single-channel CT
                 shortcut_type='B',
                 norm_cfg='IN',
                 activation_cfg='relu',
                 weight_std=False):
        super(ResNet, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        block, layers = self.arch_settings[depth]
        self.inplanes = 32
        
        # Initial convolutions - adjusted stride for 2D images
        self.conv1 = conv3x3(in_channels, 32, kernel_size=3, stride=(2, 2), 
                            padding=1, bias=False, weight_std=weight_std)
        self.norm1 = Norm_layer(norm_cfg, 32)
        self.nonlin1 = Activation_layer(activation_cfg, inplace=True)
        self.conv2 = conv3x3(32, 32, kernel_size=3, stride=(1, 1), 
                            padding=1, bias=False, weight_std=weight_std)
        self.norm2 = Norm_layer(norm_cfg, 32)
        self.nonlin2 = Activation_layer(activation_cfg, inplace=True)
        
        # Max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet stages
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, 
                                       stride=(2, 2), norm_cfg=norm_cfg, 
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, 
                                       stride=(2, 2), norm_cfg=norm_cfg, 
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, 
                                       stride=(2, 2), norm_cfg=norm_cfg, 
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layer4 = self._make_layer(block, 320, layers[3], shortcut_type, 
                                       stride=(2, 2), norm_cfg=norm_cfg, 
                                       activation_cfg=activation_cfg, weight_std=weight_std)
        self.layers = []

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=(1, 1), 
                    norm_cfg='BN', activation_cfg='relu', weight_std=False):
        downsample = None
        if stride != (1, 1) or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    conv3x3(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, 
                           stride=stride, downsample=downsample, weight_std=weight_std))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_cfg, activation_cfg, 
                               weight_std=weight_std))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2d_wd)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        self.layers = []
        x = self.nonlin1(self.norm1(self.conv1(x)))
        x = self.nonlin2(self.norm2(self.conv2(x)))
        self.layers.append(x)
        
        x = self.layer1(x)
        self.layers.append(x)
        x = self.layer2(x)
        self.layers.append(x)
        x = self.layer3(x)
        self.layers.append(x)
        x = self.layer4(x)
        self.layers.append(x)

        return x

    def get_layers(self):
        return self.layers
