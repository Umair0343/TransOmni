"""
TransOmni - 2D Multi-Scale Deformable Attention Module
Adapted from TransDoDNet's ms_deform_attn.py for 2D images

Key changes from TransDoDNet:
- 2D spatial shapes (H, W) instead of 3D (D, H, W)
- Sampling locations in 2D instead of 3D
- Pure PyTorch implementation (no CUDA kernels required)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


def ms_deform_attn_core_pytorch_2D(value, spatial_shapes, sampling_locations, attention_weights):
    """Multi-scale deformable attention core for 2D.
    
    Args:
        value: [B * n_heads, C // n_heads, sum(H_i * W_i)]
        spatial_shapes: [n_levels, 2] (H, W per level)
        sampling_locations: [B, Len_q, n_heads, n_levels, n_points, 2]
        attention_weights: [B, Len_q, n_heads, n_levels, n_points]
        
    Returns:
        Output tensor [B, Len_q, C]
    """
    N, Len_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
    C = value.shape[1]
    
    # Split value based on spatial shapes
    value_list = value.split([H * W for H, W in spatial_shapes], dim=2)
    
    # Convert sampling locations from [0, 1] to [-1, 1] for grid_sample
    sampling_grids = 2 * sampling_locations - 1
    
    sampling_value_list = []
    for lid, (H, W) in enumerate(spatial_shapes):
        # Reshape value for this level: [B*n_heads, C, H, W]
        value_l = value_list[lid].reshape(N * n_heads, C, H, W)
        
        # Get sampling locations for this level: [B, Len_q, n_heads, n_points, 2]
        sampling_grid_l = sampling_grids[:, :, :, lid, :, :]
        
        # Reshape for grid_sample: [B*n_heads, Len_q, n_points, 2]
        sampling_grid_l = sampling_grid_l.permute(0, 2, 1, 3, 4).reshape(
            N * n_heads, Len_q, n_points, 2
        )
        
        # Sample: [B*n_heads, C, Len_q, n_points]
        sampling_value_l = F.grid_sample(
            value_l, 
            sampling_grid_l,
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=False
        )
        
        sampling_value_list.append(sampling_value_l)
    
    # Reshape attention weights: [B, n_heads, Len_q, n_levels, n_points]
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        N * n_heads, 1, Len_q, n_levels * n_points
    )
    
    # Stack sampled values: [B*n_heads, C, Len_q, n_levels*n_points]
    sampling_values = torch.stack(sampling_value_list, dim=-1).reshape(
        N * n_heads, C, Len_q, n_levels * n_points
    )
    
    # Apply attention weights
    output = (sampling_values * attention_weights).sum(-1)  # [B*n_heads, C, Len_q]
    
    # Reshape output: [B, Len_q, C * n_heads]
    output = output.reshape(N, n_heads, C, Len_q).permute(0, 3, 1, 2).reshape(N, Len_q, -1)
    
    return output


class MSDeformAttn(nn.Module):
    """Multi-Scale Deformable Attention Module for 2D.
    
    This module performs deformable attention across multiple feature scales,
    learning to sample from relevant locations rather than fixed grid positions.
    """
    
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """Initialize MS Deformable Attention.
        
        Args:
            d_model: Feature dimension
            n_levels: Number of feature levels (scales)
            n_heads: Number of attention heads
            n_points: Number of sampling points per attention head per level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f'd_model ({d_model}) must be divisible by n_heads ({n_heads})')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        # Sampling offsets: predict (x, y) offset for each point
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # Attention weights for each point
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()
        
        # For visualization
        self.atte_w = None
        self.samp_location = None

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        
        # Initialize sampling offsets in a grid pattern around center
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.n_heads, 1, 1, 2
        ).repeat(1, self.n_levels, self.n_points, 1)
        
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, 
                input_level_start_index, input_padding_mask=None):
        """Forward pass of MS Deformable Attention.
        
        Args:
            query: Query features [B, Len_q, C]
            reference_points: Reference points [B, Len_q, n_levels, 2] normalized to [0, 1]
            input_flatten: Flattened multi-scale features [B, sum(H_i*W_i), C]
            input_spatial_shapes: Spatial shapes per level [n_levels, 2]
            input_level_start_index: Start index for each level [n_levels]
            input_padding_mask: Optional padding mask
            
        Returns:
            Output features [B, Len_q, C]
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in
        
        # Project values
        value = self.value_proj(input_flatten)
        
        # Apply padding mask if provided
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        
        # Reshape value for attention: [B, Len_in, n_heads, C // n_heads]
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        # Predict sampling offsets: [B, Len_q, n_heads, n_levels, n_points, 2]
        sampling_offsets = self.sampling_offsets(query).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points, 2
        )
        
        # Predict attention weights: [B, Len_q, n_heads, n_levels * n_points]
        attention_weights = self.attention_weights(query).view(
            N, Len_q, self.n_heads, self.n_levels * self.n_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            N, Len_q, self.n_heads, self.n_levels, self.n_points
        )
        
        # Add offsets to reference points
        # reference_points: [B, Len_q, n_levels, 2]
        # offset normalizer based on spatial shapes
        offset_normalizer = torch.stack([
            input_spatial_shapes[..., 1],  # W
            input_spatial_shapes[..., 0]   # H
        ], -1).float()  # [n_levels, 2]
        
        sampling_locations = reference_points[:, :, None, :, None, :] + \
            sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        
        # Reshape value for core function: [B * n_heads, C // n_heads, Len_in]
        value = value.permute(0, 2, 3, 1).reshape(
            N * self.n_heads, self.d_model // self.n_heads, Len_in
        )
        
        # Core deformable attention
        output = ms_deform_attn_core_pytorch_2D(
            value, 
            input_spatial_shapes.tolist() if isinstance(input_spatial_shapes, torch.Tensor) 
                else input_spatial_shapes,
            sampling_locations, 
            attention_weights
        )
        
        # Store for visualization
        self.atte_w = attention_weights
        self.samp_location = sampling_locations
        
        # Output projection
        output = self.output_proj(output)
        
        return output
