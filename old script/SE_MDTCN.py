# -*- coding: utf-8 -*-
"""
Created on Sat Oct 18 21:05:14 2025

@author: Fujie
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from typing import Tuple, Union, Optional

#%% Modules

def param_orthogonal_weight_norm(module: nn.Module,
                                 name: str = 'weight',
                                 dim: int = 0) -> nn.Module:
    """
    Apply orthogonal_ initialization to module.weight first, then wrap it with
    parametrizations.weight_norm.

    Args:
        module (nn.Module): The neural network module to operate on.
        name (str, optional): The name of the parameter to orthogonally initialize.
            Defaults to 'weight'.
        dim (int, optional): The dimension used for weight_norm. Defaults to 0.

    Returns:
        nn.Module: The module after orthogonal initialization and weight normalization.
    """
    # 1) Orthogonally initialize the underlying tensor
    # Get the specified parameter from the module (default: 'weight').
    # If it exists and is a Tensor, apply orthogonal initialization.
    w = getattr(module, name, None)
    if isinstance(w, torch.Tensor):
        nn.init.orthogonal_(w)  # Apply orthogonal initialization to the parameter
    elif hasattr(w, 'data') and isinstance(w.data, torch.Tensor):
        nn.init.orthogonal_(w.data)  # Apply orthogonal init to .data (for wrapped objects)

    # 2) Wrap with parametrizations.weight_norm
    # Apply weight_norm to the module parameter and return the processed module.
    return weight_norm(module, name=name, dim=dim)



class ElectrodeNormalization(nn.Module):
    """
    Perform cross-channel normalization at each time step using LayerNorm and einops.

    This module normalizes the input tensor along the channel dimension for every
    time step, which is equivalent to independently normalizing all channels at each
    time index along the temporal axis.

    Args:
        C (int): Number of channels in the input tensor (i.e., feature dimension per time step).
        eps (float, optional): A small constant added during normalization to avoid division by zero.
            Defaults to 1e-5.
        elementwise_affine (bool, optional): Whether to apply a learnable affine transform after
            normalization. Defaults to True.
        bias (bool, optional): Whether to use a bias term in the affine transform. Defaults to True.
    """

    def __init__(self, C: int, eps: float = 1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        # Create a LayerNorm layer with normalized_shape=C, meaning normalization is applied
        # over the last dimension (the channel dimension).
        self.ln = nn.LayerNorm(normalized_shape=C, eps=eps,
                               elementwise_affine=elementwise_affine, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is batch size,
                C is the number of channels, and T is the number of time steps.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T), same as the input shape,
                with the channel dimension normalized.
        """
        # Rearrange input from (B, C, T) to (B, T, C), swapping time and channel axes
        x_t = rearrange(x, 'b c t -> b t c')

        # Normalize the C channels for each time step
        x_t = self.ln(x_t)

        # Rearrange back from (B, T, C) to (B, C, T)
        return rearrange(x_t, 'b t c -> b c t')



class DWCausalConv1d(nn.Conv1d):
    """
    Depthwise causal convolution layer (Depthwise Causal Convolution).

    This layer implements causal convolution, meaning the convolution operation uses
    only the current time step and previous time steps. It follows the idea of
    depthwise separable convolution: each input channel has its own independent
    convolution kernel, and no bias term is used.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels in the output tensor.
        kernel_size (int or tuple): Size of the convolution kernel.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
        **kwargs: Other arguments passed to `nn.Conv1d`.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        # Compute the amount of left padding
        pad = (kernel_size - 1) * dilation
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=in_channels,  # Depthwise convolution: one kernel per input channel
            bias=False  # Do not use bias
        )
        self._left_pad = pad

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L), where B is batch size,
                C_in is the number of input channels, and L is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L), where C_out is the number
                of output channels and L is the sequence length, after causal convolution
                and trimming the extra padding.
        """
        # Call the parent forward method to perform convolution
        y = super().forward(x)
        # If left padding exists, remove the padded part
        if self._left_pad > 0:
            y = y[:, :, :-self._left_pad]
        return y



class ResidualBlock(nn.Module):
    """
    Residual block that combines depthwise causal convolutions and normalization layers.

    This block consists of two depthwise causal convolution layers, an activation function
    (ELU), normalization layers (ElectrodeNormalization), and a residual connection.
    After convolution, normalization, activation, and dropout, the result is added to the
    residual input to form a skip connection.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int or tuple): Size of the convolution kernel.
        dilation (int): Dilation rate of the convolution.
        dropout (float): Dropout probability used to drop some neurons.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # First depthwise causal convolution layer
        self.conv1 = param_orthogonal_weight_norm(
            DWCausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            name='weight', dim=0
        )

        # Second depthwise causal convolution layer
        self.conv2 = param_orthogonal_weight_norm(
            DWCausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            name='weight', dim=0
        )

        # First normalization layer
        self.norm1 = ElectrodeNormalization(C=in_channels, elementwise_affine=True, bias=False)

        # Second normalization layer
        self.norm2 = ElectrodeNormalization(C=in_channels, elementwise_affine=True, bias=False)

        # Activation function: ELU
        self.act = nn.ELU()

        # Dropout layer
        self.drop = nn.Dropout1d(dropout)

        # Residual connection (use 1x1 conv to match channels if in/out differ)
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L), where B is batch size,
                C_in is the number of input channels, and L is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L), where C_out is the number
                of output channels and L is the sequence length, after convolution,
                normalization, activation, and residual addition.
        """
        # Pass input through the residual branch
        res = self.down(x)

        # First conv + norm + activation + dropout
        x = self.drop(self.act(self.norm1(self.conv1(x))))
        # Second conv + norm + activation + dropout
        x = self.drop(self.act(self.norm2(self.conv2(x))))

        # Return output with residual added
        return self.act(x + res)



class DWTCNLayer(nn.Module):
    """
    Depthwise Separable Causal Convolution Network layer.

    This layer is composed of multiple `ResidualBlock`s. Each `ResidualBlock` contains
    depthwise causal convolutions, normalization, activation, and a residual connection.
    Dilated convolutions are used to extract features across multiple temporal scales.

    Args:
        input_channels (int): Number of channels in the input tensor.
        n_layers (int): Number of layers, i.e., the number of `ResidualBlock`s.
        kernel_size (int or tuple): Size of the convolution kernel.
        dropout (float): Dropout probability used to drop some neurons.
        dilations (list[int] or None, optional): Dilation sequence for each layer. If None,
            a sequence [1, 2, 4, 8, ...] is generated automatically and must have length
            >= `n_layers`. If a list is provided, dilations are taken from it in order.
    """

    def __init__(self, input_channels, n_layers, kernel_size, dropout, dilations=None):
        super().__init__()
        # 1) Choose the dilation schedule
        if dilations is None:
            dilations = [2**i for i in range(n_layers)]
        else:
            assert len(dilations) >= n_layers, "Length of dilations must be >= n_layers"

        # 2) Build layers according to the schedule
        layers = []
        for i in range(n_layers):
            layers.append(ResidualBlock(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout
            ))

        # Stack all residual blocks sequentially
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L), where B is batch size,
                C_in is the number of input channels, and L is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_in, L), same as the input shape,
                after passing through multiple residual blocks.
        """
        return self.network(x)



class DWTCNBlock(nn.Module):
    """
    Supports generating multiple branches per architecture and orthogonally initializing
    branches of the same structure.

    This module can create multiple branches, where each branch is a network composed of
    `DWTCNLayer`s. Each branch can be configured with its own parameters (e.g., number of
    layers, dilation schedule, etc.). Orthogonal initialization is applied between branches.

    Args:
        input_channels (int): Number of input channels.
        branch_params (list[dict]): A list of parameter dictionaries for each branch setup.
            Each dictionary should contain:
            - 'n_branches' (int, optional): Number of branches for this configuration.
              Defaults to 1.
            - 'n_layers' (int): Number of layers (`ResidualBlock`s) in each `DWTCNLayer`.
            - 'kernel_size' (int or tuple): Size of the convolution kernel.
            - 'dropout' (float): Dropout probability.
            - 'dilations' (list[int] or None, optional): Dilation sequence for each layer
              (length >= n_layers). If None, a dilation sequence is generated automatically.
    """

    def __init__(self, input_channels, branch_params):
        super().__init__()
        self.branches = nn.ModuleList()
        for p in branch_params:
            n_b = p.get('n_branches', 1)  # Get the number of branches (default: 1)
            for _ in range(n_b):
                branch = DWTCNLayer(
                    input_channels=input_channels,
                    n_layers=p['n_layers'],
                    kernel_size=p['kernel_size'],
                    dropout=p['dropout'],
                    dilations=p.get('dilations', None)  # New: allow passing a dilation schedule
                )
                self.branches.append(branch)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, L), where B is batch size,
                C_in is the number of input channels, and L is the sequence length.

        Returns:
            torch.Tensor: Output tensor of shape (B, C_out, L), where C_out is the sum of
                output channels from all branches, and L is the sequence length.
        """
        # Forward each branch and collect outputs; each element is a [B, C, T] tensor
        outs = [b(x) for b in self.branches]

        # Concatenate all branch outputs along the channel dimension -> [B, C * total_branches, T]
        return torch.cat(outs, dim=1)



class SEChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention.

    This module implements the SE channel attention mechanism, which uses global
    statistics to adaptively reweight the importance of each channel.
    Paper: https://arxiv.org/abs/1709.01507

    Procedure:
        1) Apply global variance pooling over the temporal dimension T to get (B, C).
        2) Pass through two fully connected layers with a nonlinearity to reduce and then
           restore dimensionality; apply a Sigmoid to obtain channel attention (B, C).
        3) Multiply the channel attention back to the original input to obtain the
           recalibrated output (B, C, T).

    Args:
        channels (int): Number of channels in the input tensor.
        reduction (int, optional): Reduction ratio controlling the hidden dimension,
            where `mid = max(channels // reduction, 1)`. Defaults to 2.
    """

    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ELU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is batch size,
                C is the number of channels, and T is the number of time steps.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T), with channels adjusted by
                the learned channel attention.
        """
        # x: [B, C, T]
        B, C, T = x.shape

        # 1) Global variance pooling -> (B, C)
        y = torch.var(x, dim=-1, unbiased=False)

        # 2) Excitation via fully connected layers -> (B, C)
        y = self.fc(y)

        # 3) Multiply the channel attention back to the original input
        y = y.view(B, C, 1)  # Reshape to (B, C, 1) for broadcasting

        # Channel attention
        attn = y
        return attn



class SE_SpatialFilter(nn.Module):
    """
    Squeeze-and-Excitation (SE) spatial filter.

    This module combines an SE channel-attention mechanism with a spatial filtering
    operation. It computes the spatial variability of input features and, together with
    channel attention, produces an adaptive spatial filtering factor. This factor is used
    to modulate the response along the spatial dimension (i.e., the temporal dimension).

    Procedure:
        1) Compute the variance of input features to measure spatial variability per channel.
        2) Compute a spatial amplitude ratio from the variability statistics.
        3) Use SE channel attention to generate channel-wise weights.
        4) Compute the spatial filter value and multiply it with the input to obtain the
           spatially filtered output.

    Args:
        channels (int): Number of channels in the input tensor.
        reduction (int, optional): Reduction ratio for channel attention. Defaults to 2.
        spatial_filter_factor (float, optional): Initial value of the spatial filter factor.
            Defaults to 2.0.
        spatial_filter_factor_learnable (bool, optional): Whether the spatial filter factor
            is learnable. Defaults to False.
    """

    def __init__(self, channels: int, reduction: int = 2,
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False):
        super().__init__()

        # Initialize the SE channel attention mechanism
        self.se_attn = SEChannelAttention(channels=channels, reduction=reduction)

        self.spatial_amp_ratio = None

        # Whether the spatial filter factor is learnable
        if spatial_filter_factor_learnable:
            self.spatial_filter_factor = nn.Parameter(
                torch.tensor(spatial_filter_factor, dtype=torch.float32)
            )
        else:
            self.spatial_filter_factor = spatial_filter_factor

        self.spatial_filter_value = None

    def cal_amp_ratio(self, x: torch.Tensor):
        """
        Compute the spatial amplitude ratio used to adjust the spatial response.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is batch size,
                C is the number of channels, and T is the number of time steps.
        """
        with torch.no_grad():
            # Reshape input and compute variance over the temporal dimension T
            x_temp = rearrange(x, 'b c t -> c (b t)')
            x_var = torch.var(x_temp, dim=-1, unbiased=False)

            # Compute the ratio between maximum and minimum variance
            x_var_max = torch.max(x_var)
            x_var_min = torch.min(x_var)

            # Ratio of max to min (with epsilon for stability)
            spatial_amp_ratio = torch.sqrt(x_var_max / (x_var_min + 1e-9))

            # Smooth the spatial amplitude ratio
            if self.spatial_amp_ratio is None:
                self.spatial_amp_ratio = spatial_amp_ratio
            else:
                self.spatial_amp_ratio = (self.spatial_amp_ratio + spatial_amp_ratio) / 2

    def get_spatial_filter_value(self):
        """
        Get the current spatial filter value.

        Returns:
            torch.Tensor: The current spatial filter value.
        """
        return self.spatial_filter_value.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T), where B is batch size,
                C is the number of channels, and T is the number of time steps.

        Returns:
            torch.Tensor: Output tensor of shape (B, C, T), the input adjusted by the
                spatial filter.
        """
        # Compute the spatial amplitude ratio
        self.cal_amp_ratio(x)

        # Compute the spatial filter value and multiply it with the input
        self.spatial_filter_value = 1 + self.spatial_filter_factor * self.spatial_amp_ratio * self.se_attn(x)
        y = x * self.spatial_filter_value

        return y



class FeatureProjector(nn.Module):
    """
    Feature projector that maps input features into a higher-dimensional space.

    This module applies a linear transformation to the input tensor: it first normalizes
    with LayerNorm, then expands the feature dimension with a fully connected layer,
    followed by a GELU nonlinearity and Dropout. The processed features are returned.

    Args:
        input_channels (int): Number of channels (feature dimension) of the input tensor.
        fc_expand_dim (int): Expansion factor that determines the hidden dimension size,
            where `hidden_dim = input_channels * fc_expand_dim`.
        dropout (float, optional): Dropout probability. Defaults to 0.2.
    """

    def __init__(self, input_channels, fc_expand_dim, dropout=0.2):
        super().__init__()
        hidden_dim = input_channels * fc_expand_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(
                normalized_shape=input_channels,
                elementwise_affine=True,
                bias=True
            ),
            nn.Linear(input_channels, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Identity()  # Keep shape unchanged; pass through the result
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C), where B is batch size and
                C is the number of input channels.

        Returns:
            torch.Tensor: Output tensor of shape (B, hidden_dim), where
                hidden_dim = input_channels * fc_expand_dim.
        """
        return self.projector(x)



#%% Complete Model  

class SE_MDTCN(nn.Module):
    """
    An Inception-style multi-branch TCN backbone with channel (spatial) and temporal attention
    (both scaling factors can optionally be learnable, with configurable initial values/ranges).

    This network combines a multi-branch Temporal Convolutional Network (TCN) with channel and
    temporal attention mechanisms. The input is first flattened and normalized, then multiple
    TCN branches extract temporal features. For the spatial dimension, a Squeeze-and-Excitation
    (SE) based module produces adaptive attention weights, followed by classification.

    Input:
        x (B, chs, els, T): Input tensor, where B is batch size, chs is channels per electrode,
                            els is number of electrodes, and T is the number of time steps.

    Output:
        logits (B, num_classes): Classification logits.

    Args:
        input_channels (int): Number of channels per electrode.
        input_electrodes (int): Number of electrodes.
        input_times (int): Number of time steps.
        num_classes (int): Number of classes.
        temporal_branch_params (list): A list of per-branch parameter dicts (e.g., #layers, kernel size).
        spatial_filter_mode (str or None, optional): Whether to use spatial attention (default: 'SE').
            If None, spatial attention is disabled.
        spatial_filter_factor (float, optional): Initial value of the spatial filter factor. Defaults to 2.0.
        spatial_filter_factor_learnable (bool, optional): Whether the spatial filter factor is learnable.
            Defaults to False.
        fc_expand_dim (int, optional): Feature expansion factor. Defaults to 4.
        fc_dropout (float, optional): Dropout probability for the FC/projection head. Defaults to 0.0.
    """

    def __init__(self,
                 input_channels: int,
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):

        super().__init__()

        # Save input arguments
        self.input_channels = input_channels
        self.input_electrodes = input_electrodes
        self.input_times = input_times
        self.num_classes = num_classes
        self.temporal_branch_params = temporal_branch_params
        self.spatial_filter_mode = spatial_filter_mode
        self.spatial_filter_factor = spatial_filter_factor
        self.spatial_filter_factor_learnable = spatial_filter_factor_learnable
        self.fc_expand_dim = fc_expand_dim
        self.fc_dropout = fc_dropout

        # Total channels after flattening: (channels per electrode) * (number of electrodes)
        self.input_channels_all = self.input_channels * self.input_electrodes

        # Preprocessing: cross-channel normalization at each time step
        self.electrode_normalization = ElectrodeNormalization(
            C=self.input_channels_all,
            elementwise_affine=True,
            bias=True
        )

        # ===== Channel attention (SE) and factor re-parameterization =====
        if (self.spatial_filter_mode or '').lower() == 'se':
            self.spatial_filter = SE_SpatialFilter(
                channels=self.input_channels_all,
                reduction=2,
                spatial_filter_factor=self.spatial_filter_factor,
                spatial_filter_factor_learnable=self.spatial_filter_factor_learnable
            )
        else:
            print('default no electrode attn')
            self.spatial_filter = nn.Identity()

        # Multi-branch TCN backbone
        self.temporal_filter = DWTCNBlock(
            input_channels=self.input_channels_all,
            branch_params=self.temporal_branch_params
        )

        # Global pooling over the temporal dimension -> (B, C_all * n_branches, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)

        # Classification head
        # FC input dim: C_all * number_of_branches
        self.fc_input_channels = self.input_channels_all * len(self.temporal_filter.branches)
        self.fc_output_channels = self.fc_input_channels * fc_expand_dim
        self.fea_proj = FeatureProjector(
            input_channels=self.fc_input_channels,
            fc_expand_dim=fc_expand_dim,
            dropout=fc_dropout
        )
        self.classifier = nn.Linear(self.fc_output_channels, num_classes, bias=True)

        # Checkpoints (useful for debugging / inspection)
        self.ckpt_1 = nn.Identity()
        self.ckpt_2 = nn.Identity()
        self.ckpt_3 = nn.Identity()
        self.ckpt_4 = nn.Identity()

    def get_spatial_filter_value(self):
        """
        Get the spatial filter value.

        If a spatial filter is used, return its current value; otherwise return None.
        """
        if self.spatial_filter != nn.Identity():
            return self.spatial_filter.get_spatial_filter_value()
        else:
            print('non spatial filter')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x (torch.Tensor): Input tensor of shape (B, chs, els, T), where B is batch size,
                chs is channels per electrode, els is number of electrodes, and T is time steps.

        Output:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        # Input x: (B, chs, els, T) -> (B, C_all, T)
        if len(x.shape) == 4:
            x = rearrange(x, 'b chs els t -> b (chs els) t')

        # Cross-channel normalization (per time step)
        x = self.electrode_normalization(x)

        # Additional temporal mean removal (zero-mean per channel within each sample)
        x = x - x.mean(dim=-1, keepdim=True)

        # Checkpoint 1
        x = self.ckpt_1(x)

        # Spatial filtering / attention
        x = self.spatial_filter(x)

        # Checkpoint 2
        x = self.ckpt_2(x)

        # TCN feature extraction
        x = self.temporal_filter(x)

        # Checkpoint 3
        x = self.ckpt_3(x)

        # Global temporal pooling
        x = self.avg_pool(x).squeeze(-1)  # shape: (B, C_all * n_branches)

        # Feature projection
        x = self.fea_proj(x)

        # Checkpoint 4
        x = self.ckpt_4(x)

        # Classification head
        x = self.classifier(x)

        return x


#%% Ablation experiments

# 1. without ElectrodeNormalization
class SE_MDTCN_woElectrodeNormalization(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.electrode_normalization = nn.Identity()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)     

# 2. without SpatialFilter
class SE_MDTCN_woSE_SpatialFilter(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.spatial_filter = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)          

# 3. Replacement DWTCN with power
class Power(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.pow(2)
    
    
class SE_MDTCN_woSE_DWTCNBlock(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.temporal_filter=Power()
        self.model.fc_input_channels = self.model.input_channels_all 
        self.model.fc_output_channels = self.model.fc_input_channels * fc_expand_dim
        self.model.fea_proj = FeatureProjector(input_channels=self.model.fc_input_channels,
                                                 fc_expand_dim=fc_expand_dim,
                                                 dropout=fc_dropout)
        self.model.classifier = nn.Linear(self.model.fc_output_channels, num_classes, bias=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)     

# 4. Replacement DWTCN with ELU
class SE_MDTCN_woSE_DWTCNBlock2(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.temporal_filter=nn.ELU(inplace=True)
        self.model.fc_input_channels = self.model.input_channels_all 
        self.model.fc_output_channels = self.model.fc_input_channels * fc_expand_dim
        self.model.fea_proj = FeatureProjector(input_channels=self.model.fc_input_channels,
                                                 fc_expand_dim=fc_expand_dim,
                                                 dropout=fc_dropout)
        self.model.classifier = nn.Linear(self.model.fc_output_channels, num_classes, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)     
 
# 5. Replacement of ElectrodeNormalization with within-channel zscore
class ZScoreNormalization(nn.Module):
    """
    ZScoreNormalization module.

    ZScoreNormalization standardizes the input tensor using Z-score normalization.
    For each channel (C) and electrode (E), it normalizes the signal along the merged
    B*T dimension (batch and time combined).
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Z-score normalization for each channel (C) and electrode (E) in the input
        tensor [B, C, E, T], along the B*T dimension.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, E, T], where B is batch size,
                C is the number of channels, E is the number of electrodes, and T is the
                time length.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as the input [B, C, E, T].
        """
        batch_size, num_channels, num_electrodes, time_steps = x.shape

        # Reshape data: [B, C, E, T] -> [C, E, B*T] by merging B and T for computation
        x_reshaped = x.view(num_channels, num_electrodes, batch_size * time_steps)  # [C, E, B*T]

        # Compute mean and standard deviation over the B*T dimension
        mean = x_reshaped.mean(dim=2, keepdim=True)  # [C, E, 1]
        std = x_reshaped.std(dim=2, keepdim=True)    # [C, E, 1]

        # Add a small epsilon to avoid division by zero
        epsilon = 1e-10
        x_normalized = (x_reshaped - mean) / (std + epsilon)

        # Restore the original shape: [B, C, E, T]
        x_normalized = x_normalized.view(batch_size, num_channels, num_electrodes, time_steps)

        return x_normalized


class SE_MDTCN_wZScoreNormalization(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        
        self.zscore= ZScoreNormalization() 
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.electrode_normalization =  nn.Identity()
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x=self.zscore(x)
        return self.model(x)   


# 6. Replacement of SE_SpatialFilter with Spatial filter in EEGNET
class DepthwiseConv2D(nn.Conv2d):  # Depthwise conv layer inheriting from nn.Conv2d
    """
    Depthwise convolution (Depthwise Conv2D).

    This layer fixes `groups=in_channels`, implementing "one independent convolution per
    input channel". The stride is fixed to 1. Padding can be 'valid' (i.e., 0), 'same',
    or an explicit int/tuple. Optionally, an L2 max-norm constraint can be applied to
    each filter over the spatial dimensions (kH*kW) before the forward pass.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels; must be an integer multiple of
            `in_channels` (depth multiplier).
        kernel_size (Tuple[int, int]): Kernel size (kH, kW).
        max_norm (Optional[float]): If not None, apply an L2 max-norm constraint to each
            filter before forward.
        padding (Union[str, int, Tuple[int, int]]): Padding mode: 'valid', 'same', or an
            explicit padding value.
        bias (bool): Whether to use a bias term.

    Attributes:
        max_norm (Optional[float]): L2 max-norm threshold (None disables it).
        weight (torch.nn.Parameter): Convolution kernel weights inherited from the parent
            class, with shape [C_out, 1, kH, kW] in the depthwise case.
        bias (Optional[torch.nn.Parameter]): Bias parameter if `bias=True`.
    """

    def __init__(self,
                 in_channels: int,  # Number of input channels
                 out_channels: int,  # Number of output channels (must be a multiple of in_channels)
                 kernel_size: Tuple[int, int],  # Kernel size
                 max_norm: Optional[float] = None,  # L2 max-norm threshold (optional)
                 padding: Union[str, int, Tuple[int, int]] = 'valid',  # Padding mode
                 bias: bool = False):  # Whether to use bias
        # out_channels must be an integer multiple of in_channels
        if out_channels % in_channels != 0:  # Validate depth multiplier divisibility
            raise ValueError(
                f"out_channels ({out_channels}) must be an integer multiple of in_channels ({in_channels})."
            )

        # Normalize/standardize padding argument
        if isinstance(padding, str):  # If a string is provided, normalize to lowercase first
            p = padding.lower()
            if p == 'valid':  # 'valid' is equivalent to zero padding
                padding_arg = 0
            elif p == 'same':  # 'same' is supported natively in newer PyTorch versions
                padding_arg = 'same'
            else:
                raise ValueError("padding only supports 'valid' or 'same', or an explicit int/tuple")
        else:
            padding_arg = padding  # If int/tuple, use directly

        # Store max-norm threshold (or None)
        self.max_norm = float(max_norm) if max_norm is not None else None

        super().__init__(in_channels=in_channels,   # Call parent constructor: set input channels
                         out_channels=out_channels,  # Set output channels
                         kernel_size=kernel_size,    # Kernel size
                         stride=1,                   # Depthwise conv uses stride=1
                         padding=padding_arg,        # Use normalized padding
                         dilation=1,                 # Default dilation=1
                         groups=in_channels,         # Key: groups=in_channels enables depthwise conv
                         bias=bias)                  # Whether to use bias

    @property
    def depth_multiplier(self) -> int:  # Read-only property: return depth multiplier
        return self.out_channels // self.in_channels  # out_channels / in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass
        # Apply an L2 max-norm constraint to each filter (C_out) over (kH*kW)
        if self.max_norm is not None:  # If max-norm is enabled
            with torch.no_grad():  # Do not track gradients
                w = self.weight.data                         # [C_out, 1, kH, kW]
                w_flat = w.view(self.out_channels, -1)       # [C_out, kH*kW] flatten spatial dims
                w_flat = torch.renorm(w_flat, p=2, dim=1, maxnorm=self.max_norm)  # Clip L2 norm
                w.copy_(w_flat.view_as(w))  # Write back to original weight shape
        return super().forward(x)  # Use parent implementation to perform the convolution


class DWSpatialFilter2D(nn.Module):  # Spatial filter module using a depthwise Conv2D

    def __init__(self,
                 in_channels: int,   # Number of input channels
                 out_channels: int,  # Number of output channels (must be a multiple of in_channels)
                 kernel_size: Tuple[int, int]):  #
        super().__init__()
        self.dwconv2D = DepthwiseConv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            max_norm=1.0,
            padding='valid',
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass
        x = x.unsqueeze(dim=1)
        x = self.dwconv2D(x)
        x = x.squeeze(dim=2)

        return x

    

class SE_MDTCN_wDWConvSpatialFilter2D(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.model.spatial_filter = DWSpatialFilter2D(in_channels=1 ,
                                                    out_channels=self.model.input_channels_all,
                                                    kernel_size= (self.model.input_channels_all, 1))
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)     


# 7. Addition of TemporalFilter in EEGNET
class TemporalFilter2D(nn.Module):  # Temporal Conv2D-based filtering module
    def __init__(self,
                 in_channels: int,   # Number of input channels
                 out_channels: int,  # Number of output channels (e.g., F1)
                 kernel_size: Tuple[int, int]):  #
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=in_channels,    # Temporal convolution: along the time dimension
            out_channels=out_channels,  # Output channels (F1)
            kernel_size=kernel_size,    # Kernel size (e.g., fs//2)
            padding='same',             # 'same' keeps the temporal length unchanged
            bias=False,                 # Do not use bias
            groups=1,                   # Standard convolution (not grouped)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass
        x = x.unsqueeze(dim=1)
        x = self.conv2d(x)
        x = x.squeeze(dim=1)

        return x


class SE_MDTCN_wTemporalFilter2D(nn.Module):

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,
                 num_classes: int,
                 temporal_branch_params: list,
                 spatial_filter_mode: str | None = 'SE',
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False,
                 fc_expand_dim: int = 4,
                 fc_dropout: float = 0.0):
        
        super().__init__()
        self.model=SE_MDTCN(input_channels=input_channels, 
                             input_electrodes=input_electrodes,
                             input_times=input_times,
                             num_classes=num_classes,
                             temporal_branch_params=temporal_branch_params,
                             spatial_filter_mode=spatial_filter_mode,
                             spatial_filter_factor=spatial_filter_factor,
                             spatial_filter_factor_learnable=spatial_filter_factor_learnable,
                             fc_expand_dim=fc_expand_dim,
                             fc_dropout=fc_dropout)
        
        self.original_spatial_filter= self.model.spatial_filter
        
        
        self.model.spatial_filter = nn.Sequential(
                                        TemporalFilter2D(in_channels=1,
                                                         out_channels=1,
                                                         kernel_size=(1, 127),),
                                        self.original_spatial_filter
                                        )
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)     

    
#%% Test codes
   
if __name__ == "__main__":
    batch_size=8
    num_channel= 1
    num_electrode=64
    num_time=512
    EEG_dummy = torch.randn(batch_size, num_channel, num_electrode, num_time)     

    # Define different parameters for three branch configurations
    branch_params = [
        {'n_layers': 2, 'kernel_size': 3, 'dropout': 0.1, 'n_branches': 3, 'dilations': [1,2,3]},
        {'n_layers': 3, 'kernel_size': 5, 'dropout': 0.2, 'n_branches': 2},
        {'n_layers': 4, 'kernel_size': 7, 'dropout': 0.15, 'n_branches': 1},
    ]
    
    model =SE_MDTCN(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN:{model_shape}")      
    
    model =SE_MDTCN_woElectrodeNormalization(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_woElectrodeNormalization:{model_shape}")    

    
    model =SE_MDTCN_woSE_SpatialFilter(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_woSE_SpatialFilter:{model_shape}")   
    
    model =SE_MDTCN_woSE_DWTCNBlock(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_woSE_DWTCNBlock:{model_shape}")   
    
    model =SE_MDTCN_wZScoreNormalization(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_wZScoreNormalization:{model_shape}")   
    
    
    model =SE_MDTCN_wDWConvSpatialFilter2D(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_wDWConvSpatialFilter2D:{model_shape}")   
    
    
    model =SE_MDTCN_wTemporalFilter2D(input_channels = num_channel, 
                            input_electrodes = num_electrode, 
                            input_times = num_time,
                            num_classes=5, 
                            temporal_branch_params=branch_params, 
                            spatial_filter_mode='SE',
                            spatial_filter_factor=2,
                            spatial_filter_factor_learnable=True,
                            fc_expand_dim=4,
                            )
    model_shape=model(EEG_dummy).shape
    print(f"SE_MDTCN_wTemporalFilter2D:{model_shape}")   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    