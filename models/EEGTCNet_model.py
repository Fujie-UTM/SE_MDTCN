# -*- coding: utf-8 -*-
"""Defines the EEGTCNet model and its supporting layers.

This module is modified from:
https://github.com/iis-eth-zurich/eeg-tcnet

Reference:
    [1] T. M. Ingolfsson, M. Hersche, X. Wang, N. Kobayashi,
        L. Cavigelli, and L. Benini, "EEG-TCNet: An Accurate Temporal
        Convolutional Network for Embedded Motor-Imagery Brain-Machine
        Interfaces," in 2020 IEEE International Conference on Systems,
        Man, and Cybernetics (SMC), 11-14 Oct. 2020, pp. 2958-2965,
        doi: 10.1109/SMC42975.2020.9283028.

The module contains the EEGTCNet architecture, including 2D convolution,
depthwise separable convolution, dilated causal convolution, TCN blocks,
activation functions, pooling layers, dropout, and fully connected layers.

Last modification: 2026-06-21
Author: Fujie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce

#%% 1. Basic blocks.
def max_norm_rows_(w: torch.Tensor, max_norm: float, eps: float = 1e-8):
    """Applies an in-place L2 max-norm constraint to each row of a tensor.

    Args:
        w (torch.Tensor): Weight tensor whose first dimension is treated as
            rows. All remaining dimensions are flattened before computing
            row-wise L2 norms.
        max_norm (float): Maximum allowed L2 norm for each row.
        eps (float, optional): Minimum norm value used to avoid division by
            zero. Defaults to 1e-8.

    Returns:
        None: The input tensor is modified in place.
    """
    with torch.no_grad():
        flat = w.view(w.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        desired = norms.clamp(max=max_norm)
        flat.mul_(desired / norms)

class DepthwiseConv2D(nn.Conv2d):  # Depthwise convolution layer inherited from nn.Conv2d.
    """Depthwise 2D convolution layer with optional L2 max-norm constraint.

    This layer fixes ``groups`` to ``in_channels`` so that each input channel is
    convolved independently. The stride is fixed to 1. Padding can be ``'valid'``
    (equivalent to 0), ``'same'``, an integer, or a pair of integers. When
    ``max_norm`` is provided, an L2 max-norm constraint is applied to each filter
    before the forward convolution.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels. This must be an integer
            multiple of ``in_channels``.
        kernel_size (Tuple[int, int]): Convolution kernel size as ``(kH, kW)``.
        max_norm (Optional[float]): Maximum L2 norm for each output filter. If
            ``None``, no max-norm constraint is applied.
        padding (Union[str, int, Tuple[int, int]]): Padding mode or explicit
            padding value. Supported strings are ``'valid'`` and ``'same'``.
        bias (bool): Whether to include a learnable bias term.

    Attributes:
        max_norm (Optional[float]): L2 max-norm threshold, or ``None`` when the
            constraint is disabled.
        weight (torch.nn.Parameter): Convolution weights inherited from
            ``nn.Conv2d`` with shape ``[C_out, 1, kH, kW]`` for depthwise
            convolution.
        bias (Optional[torch.nn.Parameter]): Optional convolution bias.
    """

    def __init__(self,
                 in_channels: int, 
                 out_channels: int,  
                 kernel_size: Tuple[int, int], 
                 max_norm: Optional[float] = None,  
                 padding: Union[str, int, Tuple[int, int]] = 'valid',  
                 bias: bool = False): 
        # Ensure that out_channels defines an integer depth multiplier.
        if out_channels % in_channels != 0:  # Validate the depth multiplier divisibility requirement.
            raise ValueError(
                f"out_channels ({out_channels}) must be an integer multiple of in_channels ({in_channels})."
            )

        # Normalize the padding argument before passing it to nn.Conv2d.
        if isinstance(padding, str): 
            p = padding.lower()  
            if p == 'valid':  # 
                padding_arg = 0  
            elif p == 'same':  
                padding_arg = 'same'  
            else:
                raise ValueError("padding only supports 'valid' or 'same', or an explicit integer/2-tuple value.")
        else:
            padding_arg = padding 

        self.max_norm = float(max_norm) if max_norm is not None else None  

        super().__init__(in_channels=in_channels,  
                         out_channels=out_channels,  
                         kernel_size=kernel_size, 
                         stride=1,  
                         padding=padding_arg, 
                         dilation=1, 
                         groups=in_channels,  
                         bias=bias)  

    @property
    def depth_multiplier(self) -> int:  # Read-only property that returns the depth multiplier.
        """Returns the number of output filters per input channel.

        Returns:
            int: The ratio ``out_channels // in_channels``.
        """
        return self.out_channels // self.in_channels  # Compute output channels divided by input channels.

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Runs depthwise convolution on the input tensor.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C_in, H, W]``.

        Returns:
            torch.Tensor: Output tensor produced by the depthwise convolution.
        """
        if self.max_norm is not None:
            max_norm_rows_(self.weight, self.max_norm)
        return super().forward(x)  # Use the parent implementation to compute the convolution.
    


class SeparableConv2D(nn.Module):  # Depthwise separable convolution module: depthwise followed by pointwise.
    """Depthwise separable 2D convolution module.

    The module first applies depthwise convolution, where each channel is
    convolved independently and the channel count is preserved. It then applies
    a 1x1 pointwise convolution to project the tensor to the target number of
    output channels. Both steps use stride 1, and the depthwise convolution uses
    ``'same'`` padding.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels produced by the pointwise
            convolution.
        kernel_size (tuple): Depthwise convolution kernel size as ``(kH, kW)``.

    Attributes:
        conv (nn.Sequential): Sequential container that wraps the depthwise and
            pointwise convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size) -> None:  # Constructor.
        """Initializes the depthwise separable convolution module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (tuple): Depthwise convolution kernel size as
                ``(kH, kW)``.
        """
        super().__init__() 
        self.conv = nn.Sequential( 
            nn.Conv2d(in_channels=in_channels,  
                      out_channels=in_channels,  
                      kernel_size=kernel_size,  
                      stride=1,  
                      groups=in_channels, 
                      padding='same',  
                      bias=False), 
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels,  
                      kernel_size=1,  # Use a 1x1 convolution.
                      stride=1,  
                      padding='same',  
                      bias=False), 
        )

    def forward(self, x):  # Forward pass.
        """Runs the separable convolution module.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C_in, H, W]``.

        Returns:
            torch.Tensor: Output tensor with shape ``[N, C_out, H, W]`` when
            ``'same'`` padding preserves spatial dimensions.
        """
        return self.conv(x) 




# TemporalConvNet input format is [batch_size, feature channels, 1d time series data].

# TCNet applies a TCN block to EEG tensors by transforming each electrode channel.
# TCNet input format is [batch_size, feature channels, electrode channels, time].
class CausalConv1d(nn.Conv1d):
    """Causal 1D convolution implemented with padded Conv1d and right trimming.

    The layer applies ``Conv1d`` with padding equal to ``(kernel_size - 1) *
    dilation`` and then removes the extra right-side values in ``forward``. This
    is equivalent to ``Conv1d(padding=(k - 1) * dilation)`` followed by a chomp
    operation of the same length.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        """Initializes a causal 1D convolution layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the 1D convolution kernel.
            stride (int, optional): Convolution stride. Only ``1`` is supported.
                Defaults to 1.
            dilation (int, optional): Dilation factor for the convolution.
                Defaults to 1.
            bias (bool, optional): Whether to include a learnable bias term.
                Defaults to False.

        Raises:
            ValueError: If ``stride`` is not 1.
        """
        # Only stride=1 is supported, which is the common TCN setting. To support other strides, adjust trimming in forward.
        if stride != 1:
            raise ValueError("CausalConv1d currently only supports stride=1.")
        pad = (int(kernel_size) - 1) * int(dilation)
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=bias
        )
        self._left_pad = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies causal 1D convolution to the input sequence.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C_in, T]``.

        Returns:
            torch.Tensor: Output tensor with shape ``[N, C_out, T]``.
        """
        y = super().forward(x)           # Apply regular convolution with padding.
        if self._left_pad > 0:
            y = y[:, :, :-self._left_pad]  # Remove the extra right-side values to preserve length and causality.
        return y.contiguous()



class ResidualBlock(nn.Module):
    """Residual block used in the temporal convolutional network.

    The block contains two causal 1D convolution stages, each followed by batch
    normalization, ELU activation, and dropout. A 1x1 convolution is used for
    the residual branch when the input and output channel counts differ.
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        """Initializes a residual TCN block.

        Args:
            n_inputs (int): Number of input channels.
            n_outputs (int): Number of output channels.
            kernel_size (int): Size of the causal convolution kernel.
            stride (int): Stride argument kept for interface compatibility.
            dilation (int): Dilation factor for the causal convolutions.
            padding (int): Expected causal padding value, equal to
                ``(kernel_size - 1) * dilation``.
            dropout (float, optional): Dropout probability used after each ELU
                activation. Defaults to 0.5.

        Raises:
            AssertionError: If ``padding`` does not match the causal convolution
            requirement.
        """
        super().__init__()
        # The padding argument follows (kernel_size - 1) * dilation and is only checked here; CausalConv1d computes it internally.
        assert padding == (kernel_size - 1) * dilation, "Causal convolution requires padding = (k-1)*dilation"

        self.net = nn.Sequential(
            CausalConv1d(n_inputs,  n_outputs, kernel_size, stride=1, dilation=dilation, bias=False),
            nn.BatchNorm1d(n_outputs),
            nn.ELU(),
            nn.Dropout(dropout),

            CausalConv1d(n_outputs, n_outputs, kernel_size, stride=1, dilation=dilation, bias=False),
            nn.BatchNorm1d(n_outputs),
            nn.ELU(),
            nn.Dropout(dropout),
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=False) if n_inputs != n_outputs else None
        self.elu = nn.ELU()

        self._initialize_weights()

    def _initialize_weights(self):
        """Initializes convolution and batch-normalization parameters.

        Returns:
            None: Module parameters are initialized in place.
        """
        nn.init.kaiming_uniform_(self.net[0].weight)  # Weight of the first CausalConv1d layer.
        nn.init.kaiming_uniform_(self.net[4].weight)  # Weight of the second CausalConv1d layer.
        nn.init.ones_(self.net[1].weight); nn.init.zeros_(self.net[1].bias)
        nn.init.ones_(self.net[5].weight); nn.init.zeros_(self.net[5].bias)
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight)

    def forward(self, x):
        """Runs the residual block.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C_in, T]``.

        Returns:
            torch.Tensor: Output tensor with shape ``[N, C_out, T]`` after the
            residual addition and ELU activation.
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)



class TCNLayer(nn.Module):
    """Temporal convolutional network composed of residual causal blocks."""

    def __init__(self, in_channels, out_channels_list, kernel_size=2, dropout=0.3):
        """Initializes a TemporalConvNet layer.

        Args:
            in_channels (int): Number of input channels.
            out_channels_list (list of int): Output channel count for each TCN
                level.
            kernel_size (int, optional): Size of the 1D convolution kernel.
                Defaults to 2.
            dropout (float, optional): Dropout probability. Defaults to 0.3.
        """
        super().__init__()
        layers = []
        num_levels = len(out_channels_list)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = in_channels if i == 0 else out_channels_list[i - 1]
            out_channels = out_channels_list[i]
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, 
                                        stride=1, dilation=dilation_size,
                                        padding=(kernel_size - 1) * dilation_size, 
                                        dropout=dropout))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Runs the temporal convolutional network.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C_in, T]``.

        Returns:
            torch.Tensor: Output tensor produced by the stacked residual blocks.
        """
        return self.network(x)



class TCNet(nn.Module):
    """Applies a TCN layer to 3D temporal data or 4D EEG feature maps."""

    def __init__(self, in_channels, out_channels_list, kernel_size, dropout=0.3):
        """Initializes the TCNet wrapper.

        Args:
            in_channels (int): Number of input feature channels.
            out_channels_list (list of int): Output channel count for each TCN
                level.
            kernel_size (int): Size of the 1D convolution kernel.
            dropout (float, optional): Dropout probability used in TCN residual
                blocks. Defaults to 0.3.
        """
        super().__init__()
        self.tcn_block = TCNLayer(
            in_channels=in_channels,
            out_channels_list=out_channels_list,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs TCNet on a 3D sequence or a 4D EEG tensor.

        Args:
            x (torch.Tensor): Input tensor with shape ``[N, C, T]`` or
                ``[N, C, E, T]``, where ``E`` is the electrode dimension.

        Returns:
            torch.Tensor: For 3D input, returns ``[N, C_out, T]``. For 4D
            input, returns ``[N, C_out, E, T]``.

        Raises:
            ValueError: If ``x`` is neither 3D nor 4D.
        """
        if x.dim() == 4:  # [N, C, E, T] -> efficient batched processing.
            N, C, E, T = x.shape
            x = x.permute(0, 2, 1, 3).reshape(N * E, C, T)   # (N*E, C, T)
            y = self.tcn_block(x)                            # (N*E, C_out, T)
            C_out = y.shape[1]
            y = y.reshape(N, E, C_out, T).permute(0, 2, 1, 3)  # -> [N, C_out, E, T]
            return y
        elif x.dim() == 3:  # [N, C, T]
            return self.tcn_block(x)
        else:
            raise ValueError(f"TCNet expects 3D or 4D input, got {x.dim()}D.")




class Dense(nn.Linear):  # Fully connected layer inherited from nn.Linear with optional L2 max-norm.
    """Linear layer with optional row-wise L2 max-norm constraint.

    If the input tensor has more than two dimensions, it is flattened to
    ``[N, in_features]`` before being passed to the linear layer. Before the
    forward linear transformation, the layer optionally applies an L2 max-norm
    constraint to each output unit's weight vector.

    Args:
        in_channels (int): Input feature dimension, equivalent to
            ``in_features``.
        out_channels (int): Output feature dimension, equivalent to
            ``out_features``.
        max_norm (Optional[float]): Maximum L2 norm for each row of the weight
            matrix. If ``None``, the constraint is disabled.
        bias (bool): Whether to include a learnable bias term.

    Attributes:
        max_norm (Optional[float]): L2 max-norm threshold, or ``None`` when the
            constraint is disabled.
        weight (torch.nn.Parameter): Linear weight matrix with shape
            ``[out_features, in_features]``.
        bias (Optional[torch.nn.Parameter]): Optional bias vector with shape
            ``[out_features]``.
    """
    def __init__(self,
                 in_channels: int,  
                 out_channels: int,  
                 max_norm: Optional[float] = 0.25,  
                 bias: bool = True) -> None:  
        super().__init__(in_features=in_channels, 
                         out_features=out_channels,  
                         bias=bias) 
        self.max_norm = float(max_norm) if max_norm is not None else None  # Store the threshold.


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        """Applies optional flattening, max-norm, and a linear transformation.

        Args:
            x (torch.Tensor): Input tensor whose first dimension is the batch
                dimension.

        Returns:
            torch.Tensor: Linear layer output with shape ``[N, out_features]``.
        """
        # Automatically flatten the input to [N, in_features].
        if x.dim() > 2:  # Flatten all dimensions except the batch dimension when the input is multidimensional.
            x = x.view(x.size(0), -1)

        if self.max_norm is not None:
            max_norm_rows_(self.weight, self.max_norm)

        # Use the parent implementation to perform the linear transformation.
        return super().forward(x) 

#%%

class EEGTCNetBackbone(nn.Module):
    """Backbone feature extractor for the EEGTCNet architecture.

    The backbone applies temporal convolution, depthwise spatial convolution,
    depthwise separable convolution, and a TCN module to extract EEG features
    before the final classifier.

    Args:
        input_channels (int): Number of channels in the input tensor.
        input_electrodes (int): Number of EEG electrodes.
        input_times (int): Number of input time samples.
        fs (int): Sampling frequency.
        F1 (int): Number of filters in the first convolution block.
        D (int): Depth multiplier for the depthwise convolution block.
        F2 (int): Number of filters in the separable convolution block.
        KE (int): Temporal convolution kernel size.
        L (int): Number of TCN layers.
        FT (int): Number of TCN feature channels.
        KT (int): TCN convolution kernel size.
        pe_drop (float): Dropout probability for the spatial convolution blocks.
        pt_drop (float): Dropout probability for the temporal convolution blocks.

    Attributes:
        input_channels (int): Number of channels in the input tensor.
        input_electrodes (int): Number of EEG electrodes.
        input_times (int): Number of input time samples.
        fs (int): Sampling frequency.
        F1 (int): Number of filters in the first convolution block.
        D (int): Depth multiplier for the depthwise convolution block.
        F2 (int): Number of filters in the separable convolution block.
        KE (int): Temporal convolution kernel size.
        L (int): Number of TCN layers.
        FT (int): Number of TCN feature channels.
        KT (int): TCN convolution kernel size.
        pe_drop (float): Spatial-block dropout probability.
        pt_drop (float): Temporal-block dropout probability.
        layer1 (nn.Sequential): Initial temporal convolution and batch
            normalization block.
        layer2 (nn.Sequential): Depthwise spatial convolution block.
        layer3 (nn.Sequential): Depthwise separable convolution block.
        tcn (TCNet): Temporal convolutional network wrapper.
    """

    def __init__(self, 
                 input_channels = 1,  
                 input_electrodes = 64,  
                 input_times = 512,  
                 fs=256,  
                 F1=8, 
                 D=2, 
                 F2=16, 
                 KE=32, 
                 L=2, 
                 FT=12, 
                 KT=4, 
                 pe_drop=0.2, 
                 pt_drop=0.3,
                 ):
        """Initializes the EEGTCNet backbone.

        Args:
            input_channels (int, optional): Number of channels in the input
                tensor. Defaults to 1.
            input_electrodes (int, optional): Number of EEG electrodes. Defaults
                to 64.
            input_times (int, optional): Number of input time samples. Defaults
                to 512.
            fs (int, optional): Sampling frequency. Defaults to 256.
            F1 (int, optional): Number of filters in the first convolution
                block. Defaults to 8.
            D (int, optional): Depth multiplier for depthwise convolution.
                Defaults to 2.
            F2 (int, optional): Number of filters in the separable convolution
                block. Defaults to 16.
            KE (int, optional): Temporal convolution kernel size. Defaults to
                32.
            L (int, optional): Number of TCN layers. Defaults to 2.
            FT (int, optional): Number of TCN feature channels. Defaults to 12.
            KT (int, optional): TCN convolution kernel size. Defaults to 4.
            pe_drop (float, optional): Dropout probability for spatial blocks.
                Defaults to 0.2.
            pt_drop (float, optional): Dropout probability for temporal blocks.
                Defaults to 0.3.
        """

        super().__init__()
        
        self.input_channels=input_channels  
        self.input_electrodes=input_electrodes  
        self.input_times=input_times 
        self.fs=fs  
        self.F1=F1 
        self.D=D
        self.F2=F2
        self.KE=KE
        self.L=L
        self.FT=FT
        self.KT=KT
        self.pe_drop= pe_drop
        self.pt_drop=pt_drop
        
        self.layer1 = nn.Sequential( 
            nn.Conv2d(in_channels=self.input_channels,  
                      out_channels=self.F1,  
                      kernel_size=(1, KE), 
                      padding='same', 
                      bias=False,  
                      groups=1,), 
            nn.BatchNorm2d(self.F1),  
            )
        
        self.layer2= nn.Sequential(
            DepthwiseConv2D(in_channels=self.F1,  
                            out_channels=(self.D * self.F1),  
                            kernel_size=(self.input_electrodes, 1),  
                            max_norm=1.0), 
            
            nn.BatchNorm2d(self.D*self.F1),  
            nn.ELU(inplace=True), 
            nn.AvgPool2d(kernel_size=(1, 8)), 
            nn.Dropout(p=self.pe_drop)  
            )
        
        self.layer3=nn.Sequential(
            SeparableConv2D(in_channels=(self.D * self.F1),
                            out_channels=self.F2,
                            kernel_size=(1, 16)),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=self.pe_drop),
            )
        
        self.tcn = TCNet(
            in_channels=self.F2,
            out_channels_list=[self.FT] * self.L,
            kernel_size=self.KT,
            dropout=self.pt_drop,
        )

    def forward(self, x):
        """Extracts EEG features with the backbone.

        Args:
            x (torch.Tensor): Input EEG tensor with shape
                ``[B, input_channels, input_electrodes, input_times]``.

        Returns:
            torch.Tensor: Extracted feature tensor with shape ``[B, FT]``.
        """
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x = self.tcn(x)
        x = x[:, :, -1, -1]         # [B, FT].
        return x

    
#%% 2. EEGTCNet model.

class EEGTCNet(nn.Module):
    """EEGTCNet model with a backbone and dense classifier.

    Args:
        input_channels (int): Number of channels in the input tensor.
        input_electrodes (int): Number of EEG electrodes.
        input_times (int): Number of input time samples.
        fc_in_channels (int): Input feature dimension of the classifier.
        num_classes (int): Number of target classes.
        fs (int): Sampling frequency.
        F1 (int): Number of filters in the first convolution block.
        D (int): Depth multiplier for the depthwise convolution block.
        F2 (int): Number of filters in the separable convolution block.
        KE (int): Temporal convolution kernel size.
        L (int): Number of TCN layers.
        FT (int): Number of TCN feature channels.
        KT (int): TCN convolution kernel size.
        pe_drop (float): Dropout probability for spatial convolution blocks.
        pt_drop (float): Dropout probability for temporal convolution blocks.

    Attributes:
        backbone (EEGTCNetBackbone): Backbone network used to extract EEG
            features.
        classifier (Dense): Dense classification head that maps extracted
            features to class logits.
    """

    def __init__(self, 
                 input_channels = 1,  
                 input_electrodes = 64,  
                 input_times = 641,  
                 fc_in_channels=12,  
                 num_classes=5,
                 fs=256, 
                 F1=8, 
                 D=2, 
                 F2=16, 
                 KE=32, 
                 L=2, 
                 FT=12, 
                 KT=4, 
                 pe_drop=0.2, 
                 pt_drop=0.3):
        """Initializes the EEGTCNet classifier model.

        Args:
            input_channels (int, optional): Number of channels in the input
                tensor. Defaults to 1.
            input_electrodes (int, optional): Number of EEG electrodes. Defaults
                to 64.
            input_times (int, optional): Number of input time samples. Defaults
                to 641.
            fc_in_channels (int, optional): Input feature dimension of the
                classifier. Defaults to 12.
            num_classes (int, optional): Number of target classes. Defaults to
                5.
            fs (int, optional): Sampling frequency. Defaults to 256.
            F1 (int, optional): Number of filters in the first convolution
                block. Defaults to 8.
            D (int, optional): Depth multiplier for depthwise convolution.
                Defaults to 2.
            F2 (int, optional): Number of filters in the separable convolution
                block. Defaults to 16.
            KE (int, optional): Temporal convolution kernel size. Defaults to
                32.
            L (int, optional): Number of TCN layers. Defaults to 2.
            FT (int, optional): Number of TCN feature channels. Defaults to 12.
            KT (int, optional): TCN convolution kernel size. Defaults to 4.
            pe_drop (float, optional): Dropout probability for spatial blocks.
                Defaults to 0.2.
            pt_drop (float, optional): Dropout probability for temporal blocks.
                Defaults to 0.3.
        """
                 
                
        super().__init__()
        self.backbone=EEGTCNetBackbone( input_channels=input_channels,  
                                        input_electrodes=input_electrodes,  
                                        input_times=input_times,  
                                        fs=fs,  
                                        F1=F1, 
                                        D=D,
                                        F2=F2,
                                        KE=KE,
                                        L=L,
                                        FT=FT,
                                        KT=KT,
                                        pe_drop= pe_drop,
                                        pt_drop=pt_drop,)
            
        self.classifier=Dense(in_channels=fc_in_channels,  
                              out_channels=num_classes, 
                              max_norm=0.25)  # Optional L2 max-norm constraint threshold.

    def forward(self, x):
        """Runs the full EEGTCNet model.

        Args:
            x (torch.Tensor): Input EEG tensor with shape
                ``[B, input_channels, input_electrodes, input_times]``.

        Returns:
            torch.Tensor: Class logits with shape ``[B, num_classes]``.
        """
        x=self.backbone(x) 
        x=self.classifier(x)  
        return x  

#%%%
if __name__ == "__main__":

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = EEGTCNet(
        input_channels= 1, 
        input_electrodes=64,    
        input_times=641,
        fc_in_channels=12,
        fs=256,
        F1=8, 
        D=2, 
        F2=16, 
        KE=32, 
        L=2,
        FT=12,
        KT=4,
        pe_drop=0.2,
        pt_drop=0.3).to(device)
    
    x = torch.randn(8, 1, 64, 641, device=device)
    logits = model(x)
    print('logits shape:', logits.shape)  # [8, 5]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    








