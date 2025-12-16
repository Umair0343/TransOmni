"""
TransOmni - 2D TransDoDNet Model
Adapted from TransDoDNet's TransDoDNet.py for 2D histopathology images

Key changes from TransDoDNet:
- All Conv3d -> Conv2d
- All ConvTranspose3d -> ConvTranspose2d
- All 3D normalization -> 2D normalization
- Upsample: trilinear -> bilinear
- Dynamic heads use 2D convolutions
- Output shape: [B, N_queries, 2, H, W] instead of [B, N_queries, 2, D, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import ResCNN2
from models.deformable_transformer import build_deformable_transformer
from models.position_encoding import build_position_encoding


def _expand(tensor, length: int):
    """Expand tensor for dynamic head processing.
    
    Args:
        tensor: Input tensor [B, C, H, W]
        length: Number of queries to expand to
        
    Returns:
        Expanded tensor [B*length, C, H, W]
    """
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class Conv2d_wd(nn.Conv2d):
    """Conv2d with weight standardization."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1), 
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
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
            dilation=(1, 1), groups=1, bias=True, weight_std=False):
    """3x3 convolution with padding."""
    if weight_std:
        return Conv2d_wd(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, groups=groups, bias=bias)
    else:
        return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, groups=groups, bias=bias)


def Norm_layer(norm_cfg, inplanes):
    """Create normalization layer."""
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
    """Create activation layer."""
    if activation_cfg == 'relu':
        out = nn.ReLU(inplace=inplace)
    elif activation_cfg == 'LeakyReLU':
        out = nn.LeakyReLU(negative_slope=1e-2, inplace=inplace)
    return out


class ResCNN_DeformTR(nn.Module):
    """2D ResCNN with Deformable Transformer for multi-task segmentation.
    
    This model combines a CNN encoder-decoder with a deformable transformer
    for learning task-specific dynamic heads.
    """
    
    def __init__(self, args, norm_cfg='IN', activation_cfg='relu', num_classes=None, 
                 weight_std=False, res_depth=None, dyn_head_dep_wid=[3, 8]):
        super(ResCNN_DeformTR, self).__init__()

        self.args = args
        self.args.activation = activation_cfg
        self.num_classes = num_classes
        self.dyn_head_dep_wid = dyn_head_dep_wid
        
        if res_depth >= 50:
            expansion = 4
        else:
            expansion = 1

        # Calculate number of dynamic parameters
        num_dyn_params = (dyn_head_dep_wid[1] * dyn_head_dep_wid[1] + dyn_head_dep_wid[1]) * (dyn_head_dep_wid[0] - 1) + \
                        (dyn_head_dep_wid[1] * 2 + 2) * 1
        print(f"###Total dyn params {num_dyn_params}###")

        # 2D Upsample (bilinear instead of trilinear)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Memory convolution from transformer
        if self.args.add_memory >= 1:
            if self.args.num_feature_levels >= 1:
                self.memory_conv3_l0 = nn.Sequential(
                    conv3x3(args.hidden_dim, 256, kernel_size=1, bias=False, weight_std=weight_std),
                    Norm_layer(norm_cfg, 256),
                    Activation_layer(activation_cfg, inplace=True),
                )

        # CNN bottleneck
        self.cnn_bottle = nn.Sequential(
            conv3x3(320 * expansion, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        # Skip connection convolutions
        self.shortcut_conv3 = nn.Sequential(
            conv3x3(256 * expansion, 256, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 256),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv2 = nn.Sequential(
            conv3x3(128 * expansion, 128, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 128),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv1 = nn.Sequential(
            conv3x3(64 * expansion, 64, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 64),
            Activation_layer(activation_cfg, inplace=True),
        )

        self.shortcut_conv0 = nn.Sequential(
            conv3x3(32, 32, kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, 32),
            Activation_layer(activation_cfg, inplace=True),
        )

        # 2D Transpose convolutions for decoder
        self.transposeconv_stage3 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, bias=False)
        self.transposeconv_stage0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, bias=False)

        # Decoder stages (using 2D BasicBlocks)
        self.stage3_de = ResCNN2.BasicBlock(256, 256, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage2_de = ResCNN2.BasicBlock(128, 128, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage1_de = ResCNN2.BasicBlock(64, 64, norm_cfg, activation_cfg, weight_std=weight_std)
        self.stage0_de = ResCNN2.BasicBlock(32, 32, norm_cfg, activation_cfg, weight_std=weight_std)

        # Pre-classification convolution
        self.precls_conv = nn.Sequential(
            conv3x3(32, dyn_head_dep_wid[1], kernel_size=1, bias=False, weight_std=weight_std),
            Norm_layer(norm_cfg, dyn_head_dep_wid[1]),
            Activation_layer(activation_cfg, inplace=True),
        )

        # 2D ResNet backbone
        self.backbone = ResCNN2.ResNet(
            depth=res_depth, 
            shortcut_type='B', 
            norm_cfg=norm_cfg,
            activation_cfg=activation_cfg,
            weight_std=weight_std
        )
        self.backbone_layers = self.backbone.get_layers()

        # Deformable Transformer
        if self.args.using_transformer:
            self.transformer = build_deformable_transformer(args)
            self.position_embedding = build_position_encoding(args)
            
            # Controller generates dynamic head parameters
            self.controller = nn.Sequential(
                nn.Linear(args.hidden_dim, args.hidden_dim),
                Activation_layer(activation_cfg, inplace=True),
                nn.Linear(args.hidden_dim, num_dyn_params)
            )
            
            # Input projection for transformer
            input_proj_list = []
            backbone_num_channels = [64 * expansion, 128 * expansion, 256 * expansion, 320 * expansion]
            backbone_num_channels = backbone_num_channels[-args.num_feature_levels:]
            for in_channels in backbone_num_channels:
                input_proj_list.append(nn.Sequential(
                    conv3x3(in_channels, args.hidden_dim, kernel_size=1, bias=True, weight_std=weight_std),
                    Norm_layer(norm_cfg, args.hidden_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
            
            # Query embeddings (one per task/tissue type)
            self.query_embed = nn.Embedding(self.args.num_queries, args.hidden_dim * 2)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2d_wd, nn.ConvTranspose2d)):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def parse_dynamic_params(self, params, channels, weight_nums, bias_nums):
        """Parse dynamic parameters for convolution layers.
        
        Args:
            params: Predicted parameters [num_instances, total_params]
            channels: Number of channels in intermediate layers
            weight_nums: List of weight parameter counts per layer
            bias_nums: List of bias parameter counts per layer
            
        Returns:
            Tuple of (weight_splits, bias_splits)
        """
        assert params.dim() == 2
        assert len(weight_nums) == len(bias_nums)
        assert params.size(1) == sum(weight_nums) + sum(bias_nums)

        num_insts = params.size(0)
        num_layers = len(weight_nums)

        params_splits = list(torch.split_with_sizes(
            params, weight_nums + bias_nums, dim=1
        ))

        weight_splits = params_splits[:num_layers]
        bias_splits = params_splits[num_layers:]

        for l in range(num_layers):
            if l < num_layers - 1:
                # 2D reshape: [num_insts * channels, in_channels, 1, 1]
                weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
            else:
                # Final layer: 2 classes
                weight_splits[l] = weight_splits[l].reshape(num_insts * 2, -1, 1, 1)
                bias_splits[l] = bias_splits[l].reshape(num_insts * 2)

        return weight_splits, bias_splits

    def heads_forward(self, features, weights, biases, num_insts):
        """Forward pass through dynamic heads.
        
        Args:
            features: Input features [1, num_insts * C, H, W]
            weights: List of weight tensors
            biases: List of bias tensors
            num_insts: Number of instances (batch * queries)
            
        Returns:
            Output tensor
        """
        assert features.dim() == 4  # 2D: [1, C, H, W]
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            # 2D convolution instead of 3D
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def forward(self, inputs_x, task_id):
        """Forward pass of TransDoDNet.
        
        Args:
            inputs_x: Input images [B, 3, H, W]
            task_id: Task IDs for each sample in batch
            
        Returns:
            Segmentation output [B, num_queries, 2, H, W]
        """
        bs, c, h, w = inputs_x.shape

        # Backbone feature extraction
        _ = self.backbone(inputs_x)
        layers = self.backbone.get_layers()

        # Transformer processing
        srcs = []
        masks = []
        pos = []
        for l, feat in enumerate(layers[-self.args.num_feature_levels:]):
            src = feat
            srcs.append(self.input_proj[l](src))
            # Create mask (all False = all valid) for 2D
            masks.append(torch.zeros(src.shape[0], src.shape[2], src.shape[3], 
                        dtype=torch.bool, device=src.device))
            pos.append(self.position_embedding(src).to(src.dtype))
            del feat

        hs, _, _, _, _, memory = self.transformer(srcs, masks, pos, self.query_embed.weight)
        params = self.controller(hs[-1].flatten(0, 1))

        # Decoder path
        if self.args.add_memory == 0:
            x = self.cnn_bottle(layers[-1])
        elif self.args.add_memory == 1:
            x = self.memory_conv3_l0(memory[-1])
        elif self.args.add_memory == 2:
            x = self.memory_conv3_l0(memory[-1]) + self.cnn_bottle(layers[-1])
        else:
            print("Error: no pre-defined add_memory mode!")

        # Decoder stages with skip connections
        skip3 = self.shortcut_conv3(layers[-2])
        x = self.transposeconv_stage3(x)
        x = x + skip3
        x = self.stage3_de(x)

        skip2 = self.shortcut_conv2(layers[-3])
        x = self.transposeconv_stage2(x)
        x = x + skip2
        x = self.stage2_de(x)

        x = self.transposeconv_stage1(x)
        skip1 = self.shortcut_conv1(layers[-4])
        x = x + skip1
        x = self.stage1_de(x)

        x = self.transposeconv_stage0(x)
        skip0 = self.shortcut_conv0(layers[-5])
        x = x + skip0
        x = self.stage0_de(x)

        # Dynamic head computation
        head_inputs = self.precls_conv(x)
        head_inputs = _expand(head_inputs, self.args.num_queries)
        N, _, H, W = head_inputs.size()  # 2D: [B*num_queries, C, H, W]
        head_inputs = head_inputs.reshape(1, -1, H, W)  # [1, B*num_queries*C, H, W]

        # Prepare weight and bias counts
        weight_nums, bias_nums = [], []
        for i in range(self.dyn_head_dep_wid[0] - 1):
            weight_nums.append(self.dyn_head_dep_wid[1] * self.dyn_head_dep_wid[1])
            bias_nums.append(self.dyn_head_dep_wid[1])
        weight_nums.append(self.dyn_head_dep_wid[1] * 2)
        bias_nums.append(2)

        weights, biases = self.parse_dynamic_params(params, self.dyn_head_dep_wid[1], weight_nums, bias_nums)

        # Dynamic head forward
        seg_out = self.heads_forward(head_inputs, weights, biases, N)
        seg_out = self.upsample(seg_out)
        
        # Reshape output: [B, num_queries, num_classes, H, W]
        seg_out = seg_out.view(bs, self.args.num_queries, self.num_classes, 
                              seg_out.shape[-2], seg_out.shape[-1])

        return seg_out


class TransDoDNet(nn.Module):
    """2D TransDoDNet for multi-task histopathology segmentation.
    
    This is the main model class that wraps ResCNN_DeformTR with
    additional configuration options.
    """
    
    def __init__(self, args, norm_cfg='IN', activation_cfg='relu', num_classes=None,
                 weight_std=False, deep_supervision=False, res_depth=None, 
                 dyn_head_dep_wid=[3, 8]):
        super().__init__()
        self.do_ds = False
        self.ResCNN_DeformTR = ResCNN_DeformTR(
            args, norm_cfg, activation_cfg, num_classes, 
            weight_std, res_depth, dyn_head_dep_wid
        )
        
        if weight_std == False:
            self.conv_op = nn.Conv2d
        else:
            self.conv_op = Conv2d_wd
            
        if norm_cfg == 'BN':
            self.norm_op = nn.BatchNorm2d
        if norm_cfg == 'GN':
            self.norm_op = nn.GroupNorm
        if norm_cfg == 'IN':
            self.norm_op = nn.InstanceNorm2d
            
        self.num_classes = num_classes
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

    def forward(self, x, task_id):
        """Forward pass.
        
        Args:
            x: Input images [B, 3, H, W]
            task_id: Task IDs for each sample
            
        Returns:
            Segmentation output [B, num_queries, num_classes, H, W]
        """
        seg_output = self.ResCNN_DeformTR(x, task_id)
        return seg_output
