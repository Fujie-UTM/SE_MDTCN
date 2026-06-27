# -*- coding: utf-8 -*-
"""
PyTorch implementation of the HS-STDCN model.

This file implements the hybrid-scale spatial-temporal dilated convolution
network (HS-STDCN) architecture from the paper for EEG imagined speech
recognition. The code includes hybrid-scale temporal convolution, depthwise
spatial convolution, dilated convolution residual blocks, a fully connected
classification layer with a max-norm constraint, and a simple random-input
test entry point.

References:
    [1] F. Li et al., "Decoding imagined speech from EEG signals using
    hybrid-scale spatial-temporal dilated convolution network,"
    Journal of Neural Engineering, vol. 18, no. 4, p. 0460c4,
    2021/08/11 2021, doi: 10.1088/1741-2552/ac13c0.
    
Last modification: 2026-06-21
@author: Fujie    
"""


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional
import torch.optim as optim
#%%

def max_norm_rows_(w: torch.Tensor, max_norm: float, eps: float = 1e-8):
    """Applies an L2 max-norm constraint to each row of a tensor.

    This function treats dimension 0 of the input tensor `w` as the row
    dimension and flattens all remaining dimensions into the feature dimension
    of each row. If the L2 norm of a row is greater than `max_norm`, that row
    is scaled down proportionally; otherwise, it is left unchanged. The trailing
    underscore in the function name indicates that the input tensor is modified
    in place.

    Args:
        w (torch.Tensor): Weight tensor to constrain. Dimension 0 is treated as
            the row dimension, and all remaining dimensions are flattened into
            the feature dimension of each row.
        max_norm (float): Maximum allowed L2 norm for each row.
        eps (float): Lower bound for row norms to avoid division by zero.
            Defaults to 1e-8.

    Returns:
        None: This function modifies `w` in place and does not return a new
        tensor.
    """
    # Applies the in-place weight constraint without tracking gradients so that this operation does not enter the autograd graph.
    with torch.no_grad():
        # Keeps dimension 0 as the row dimension and flattens all following dimensions into the column dimension.
        flat = w.view(w.size(0), -1)
        # Computes the L2 norm of each row and uses eps as a lower bound to avoid division by zero.
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        # Limits the target norms so that they do not exceed max_norm.
        desired = norms.clamp(max=max_norm)
        # Scales each row by desired / norms so that its norm does not exceed max_norm.
        flat.mul_(desired / norms)


# 1. Hybrid-scale temporal convolution module.
class HybridTemporalConv(nn.Module):
    """Hybrid-scale temporal convolution module.

    This module builds one `Conv2d + BatchNorm2d` branch for each given temporal
    kernel length. Each branch performs convolution only along the temporal
    dimension, and the outputs from all scale branches are added elementwise to
    fuse EEG temporal features at different time scales.

    Attributes:
        conv_list (nn.ModuleList): List of temporal convolution branches. Each
            branch contains a two-dimensional convolution layer and a
            two-dimensional batch normalization layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size_list=[7, 19, 31, 63]):
        """Initializes the hybrid-scale temporal convolution module.

        Args:
            in_channels (int): Number of input channels. For EEG input with
                shape `[B, 1, C, T]`, this is typically 1.
            out_channels (int): Number of output channels for each temporal
                scale convolution branch, corresponding to F1 in the paper.
            kernel_size_list (list[int]): List of convolution kernel lengths for
                different temporal scales. Defaults to `[7, 19, 31, 63]`.
        """
        super().__init__()
        # Builds one convolution branch for each temporal kernel length and uses same padding to preserve the temporal length.
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding='same'),
                nn.BatchNorm2d(out_channels)
            )
            for k in kernel_size_list
        ])

    def forward(self, x):
        """Runs the forward pass of hybrid-scale temporal convolution.

        Args:
            x (torch.Tensor): Input EEG tensor with shape `[B, 1, C, T]`, where
                `B` is the batch size, `C` is the number of electrodes, and `T`
                is the number of temporal samples.

        Returns:
            torch.Tensor: Temporal feature tensor fused from multiple time
            scales, with shape `[B, F1, C, T]`.
        """
        # The input x has shape [B, 1, C, T].
        # Passes x through each temporal convolution branch, stacks all branch outputs, and sums along the new dimension.
        out = torch.sum(torch.stack([conv(x) for conv in self.conv_list]), dim=0)
        # out = sum(conv(x) for conv in self.conv_list)  # Shape: [B, F1, C, T]
        return out


# 2. Spatial convolution module (depthwise separable convolution).
class DepthwiseSpatialConv(nn.Module):
    """Depthwise spatial convolution module.

    This module uses a two-dimensional convolution with `groups=in_channels` to
    perform spatial convolution independently on each input channel. The kernel
    height equals the number of EEG electrodes, so the spatial convolution scans
    along the electrode dimension and compresses it from `input_electrodes` to
    1.

    Attributes:
        conv (nn.Conv2d): Depthwise spatial convolution layer with kernel size
            `(input_electrodes, 1)`.
    """

    def __init__(self, in_channels, out_channels, input_electrodes):
        """Initializes the depthwise spatial convolution module.

        Args:
            in_channels (int): Number of input feature channels, typically F1
                from the previous layer.
            out_channels (int): Number of output feature channels. In the
                current model, this is typically `F1 * 2`, corresponding to a
                depth multiplier of 2.
            input_electrodes (int): Number of EEG electrodes, used as the height
                of the spatial convolution kernel.
        """
        super(DepthwiseSpatialConv, self).__init__()
        # Uses groups=in_channels to implement channel-wise spatial convolution with kernel size (input_electrodes, 1).
        self.conv = nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size=(input_electrodes, 1),
                              groups=in_channels,
                              padding='valid',
                              bias=False)
    def forward(self, x):
        """Runs the forward pass of depthwise spatial convolution.

        Args:
            x (torch.Tensor): Input tensor with shape `[B, F1, C, T]`.

        Returns:
            torch.Tensor: Tensor after spatial convolution, with shape
            `[B, out_channels, 1, T]`.
        """
        # The input x has shape [B, F1, C, T].
        out = self.conv(x)  # Output shape: [B, out_channels, 1, T].

        return out


# 3. Dilated convolution block for one-dimensional temporal sequences.
class DilatedBlock(nn.Module):
    """One-dimensional dilated convolution residual block.

    This module contains two one-dimensional dilated convolution layers whose
    dilation rates are specified by `dilation_list`. Each dilated convolution is
    followed by batch normalization, ELU activation, and Dropout. The module
    also contains a residual connection: if the number of input channels differs
    from the number of output channels, a 1x1 convolution is used to align the
    channels in the residual branch.

    Attributes:
        dilated_conv1 (nn.Conv1d): First one-dimensional dilated convolution
            layer.
        dilated_conv2 (nn.Conv1d): Second one-dimensional dilated convolution
            layer.
        down (nn.Module): Channel alignment layer for the residual branch; an
            identity mapping when the channel counts are the same.
        bn1 (nn.BatchNorm1d): Batch normalization layer after the first dilated
            convolution.
        bn2 (nn.BatchNorm1d): Batch normalization layer after the second dilated
            convolution.
        act (nn.ELU): ELU activation function.
        drop (nn.Dropout): Dropout layer.
    """

    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilation_list=[1, 2],
                 p_dropout=0.3):
        """Initializes the one-dimensional dilated convolution residual block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Length of the one-dimensional dilated convolution
                kernel. Defaults to 3.
            dilation_list (list[int]): List of dilation rates for the two
                dilated convolution layers. Defaults to `[1, 2]`.
            p_dropout (float): Dropout probability. Defaults to 0.3.
        """
        super().__init__()

        # First dilated convolution layer, using dilation_list[0] as the dilation rate and same padding to preserve temporal length.
        self.dilated_conv1 = nn.Conv1d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size = kernel_size,
                                       dilation = dilation_list[0],
                                       padding = 'same')

        # Second dilated convolution layer, with both input and output channels equal to out_channels.
        self.dilated_conv2 = nn.Conv1d(in_channels=out_channels,
                                       out_channels=out_channels, 
                                       kernel_size = kernel_size, 
                                       dilation = dilation_list[1],
                                       padding ='same')

        # Uses a 1x1 convolution to match channels in the residual branch if input and output channel counts differ.
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        # One-dimensional batch normalization layers corresponding to the two dilated convolution layers.
        self.bn1=nn.BatchNorm1d(num_features=out_channels)
        self.bn2=nn.BatchNorm1d(num_features=out_channels)

        # Activation function and Dropout regularization layer.
        self.act=nn.ELU()
        self.drop = nn.Dropout(p=p_dropout)

    def forward(self, x):
        """Runs the forward pass of the dilated convolution residual block.

        Args:
            x (torch.Tensor): Input one-dimensional sequence features with shape
                `[B, in_channels, T]`.

        Returns:
            torch.Tensor: Output one-dimensional sequence features with shape
            `[B, out_channels, T]`.
        """
        # Computes the residual branch; down performs channel projection if the channel counts differ.
        res = self.down(x)      
        # First dilated convolution: convolution, batch normalization, ELU activation, and Dropout.
        x=self.drop(self.act(self.bn1(self.dilated_conv1(x))))
        # Second dilated convolution: convolution, batch normalization, ELU activation, and Dropout.
        x=self.drop(self.act(self.bn2(self.dilated_conv2(x))))
        # Adds the main branch output and the residual branch.
        x= x+res
        return x


class HS_STDCNBackbone(nn.Module):
    """Feature extraction backbone of HS-STDCN.

    The backbone sequentially contains hybrid-scale temporal convolution,
    depthwise spatial convolution, batch normalization, ELU, average pooling,
    Dropout, two one-dimensional dilated convolution residual blocks, and a
    flattening layer. This module outputs deep spatial-temporal feature vectors
    for the classifier.

    Attributes:
        hybrid_temporal (HybridTemporalConv): Hybrid-scale temporal convolution
            module.
        spatial_out_channels (int): Number of output channels after spatial
            convolution, set to `F1 * 2`.
        depthwise_spatial (DepthwiseSpatialConv): Depthwise spatial convolution
            module.
        process (nn.Sequential): Processing sequence after spatial convolution,
            including batch normalization, ELU, average pooling, and Dropout.
        dilated_block1 (DilatedBlock): First dilated convolution residual block.
        dilated_block2 (DilatedBlock): Second dilated convolution residual block.
        flatten (nn.Flatten): Flattening layer.
    """

    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,    
                 temporal_ks_list=[7, 19, 31, 63],
                 avg_pool_ks=16,
                 F1=8, 
                 F2=16,
                 dilation_ks=3,
                 dilation_list=[1, 2],
                 dropout_spatial=0.2,
                 dropout_dilated=0.3):
        """Initializes the HS-STDCN feature extraction backbone.

        Args:
            input_channels (int): Number of channels in the input tensor,
                typically 1.
            input_electrodes (int): Number of EEG electrodes.
            input_times (int): Number of EEG temporal samples. This parameter is
                retained to record the input configuration.
            temporal_ks_list (list[int]): List of hybrid-scale temporal
                convolution kernel lengths.
            avg_pool_ks (int): Average pooling kernel length along the temporal
                dimension.
            F1 (int): Number of output channels for each hybrid-scale temporal
                convolution branch.
            F2 (int): Number of output channels for the dilated convolution
                blocks.
            dilation_ks (int): Dilated convolution kernel length.
            dilation_list (list[int]): Dilation rates for the two convolution
                layers in a dilated convolution block.
            dropout_spatial (float): Dropout probability after the spatial
                convolution processing stage.
            dropout_dilated (float): Dropout probability inside the dilated
                convolution blocks.
        """
        super().__init__()

        # Hybrid-scale temporal convolution: input shape [B, 1, C, T], output shape [B, F1, C, T].
        self.hybrid_temporal = HybridTemporalConv(in_channels=input_channels,
                                                  out_channels=F1,
                                                  kernel_size_list=temporal_ks_list)

        # Number of output channels after spatial convolution, corresponding to a depth multiplier of 2.
        self.spatial_out_channels = F1 * 2

        # Depthwise spatial convolution: performs convolution along the electrode dimension and reduces the electrode dimension to 1.
        self.depthwise_spatial = DepthwiseSpatialConv(in_channels=F1, 
                                                      out_channels=self.spatial_out_channels,
                                                      input_electrodes=input_electrodes)
        # Standard processing flow after spatial convolution: batch normalization, ELU activation, temporal average pooling, and Dropout.
        self.process= nn.Sequential(
            nn.BatchNorm2d(num_features= self.spatial_out_channels),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_ks)),
            nn.Dropout(dropout_spatial),
            )
        # First dilated convolution residual block, mapping spatial convolution output channels to F2.
        self.dilated_block1=DilatedBlock(in_channels= self.spatial_out_channels, 
                                         out_channels=F2, 
                                         kernel_size=dilation_ks, 
                                         dilation_list=dilation_list,
                                         p_dropout=dropout_dilated)

        # Second dilated convolution residual block, with both input and output channels equal to F2.
        self.dilated_block2=DilatedBlock(in_channels=F2, 
                                         out_channels=F2, 
                                         kernel_size=dilation_ks, 
                                         dilation_list=dilation_list,
                                         p_dropout=dropout_dilated)
        # Flattens the final one-dimensional sequence features into the classifier input vector.
        self.flatten=nn.Flatten()
        
    def forward(self, x):
        """Runs the forward pass of the HS-STDCN backbone.

        Args:
            x (torch.Tensor): Input EEG tensor with shape `[B, input_channels,
                input_electrodes, input_times]`.

        Returns:
            torch.Tensor: Flattened feature vector with shape
            `[B, F2 * pooled_T]`, where `pooled_T` is the temporal length after
            average pooling.
        """
        # Reads the batch size, channel count, electrode count, and temporal length from the input tensor.
        B, C, E, T = x.size()    

        # Extracts hybrid-scale temporal features.
        x=self.hybrid_temporal(x)
        # Extracts cross-electrode spatial features.
        x=self.depthwise_spatial(x)
        # Applies batch normalization, ELU, average pooling, and Dropout.
        x=self.process(x)

        # Rearranges shape [B, C, E, T] to [B, C * E, T] for input to one-dimensional dilated convolution.
        x = rearrange(x, 'B C E T -> B (C E) T')

        # Passes the tensor through two dilated convolution residual blocks in sequence.
        x=self.dilated_block1(x)
        x=self.dilated_block2(x)
        return self.flatten(x)


class Dense(nn.Linear):  # Fully connected layer inherited from nn.Linear with optional L2 max-norm.
    """Fully connected layer with an optional L2 max-norm constraint.

    This layer inherits from `nn.Linear`. If the input tensor has more than two
    dimensions, it is first flattened to `[N, in_features]` before being passed
    to the linear layer. During the forward pass, if `max_norm` is not None, an
    L2 max-norm constraint is applied to the weight row corresponding to each
    output neuron.

    Attributes:
        max_norm (Optional[float]): L2 max-norm threshold. None disables the
            constraint.
        weight (torch.nn.Parameter): Linear layer weight with shape
            `[out_features, in_features]`.
        bias (Optional[torch.nn.Parameter]): Linear layer bias with shape
            `[out_features]`; absent when `bias=False`.
    """

    def __init__(self,
                 in_channels: int,  
                 out_channels: int,  
                 max_norm: Optional[float] = 0.25, 
                 bias: bool = True) -> None:
        """Initializes the fully connected layer with a max-norm constraint.

        Args:
            in_channels (int): Input feature dimension, corresponding to
                `in_features` in `nn.Linear`.
            out_channels (int): Output feature dimension, corresponding to
                `out_features` in `nn.Linear`.
            max_norm (Optional[float]): L2 max-norm threshold. If None, no
                max-norm constraint is applied. Defaults to 0.25.
            bias (bool): Whether to use a bias term. Defaults to True.
        """
        super().__init__(in_features=in_channels, 
                         out_features=out_channels,  
                         bias=bias) 
        # Records the max-norm threshold; passing None disables the max-norm constraint.
        self.max_norm = float(max_norm) if max_norm is not None else None


    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        """Runs the forward pass of the fully connected layer.

        Args:
            x (torch.Tensor): Input feature tensor. If it has more than two
                dimensions, it is first flattened to `[N, in_features]`.

        Returns:
            torch.Tensor: Output tensor after the linear transformation, with
            shape `[N, out_channels]`.
        """
        # Automatically flattens a multidimensional input tensor to [N, in_features].
        if x.dim() > 2:  # If the input is multidimensional, first flatten all dimensions except the batch dimension.
            x = x.view(x.size(0), -1)

        # If the max-norm constraint is enabled, constrains weight row norms before the forward computation.
        if self.max_norm is not None:
            max_norm_rows_(self.weight, self.max_norm)

        # Calls the nn.Linear forward pass to perform the linear transformation.
        return super().forward(x)  # Returns the linear transformation result.


#%%
class HS_STDCN(nn.Module):
    """HS-STDCN classification model.

    This model consists of an `HS_STDCNBackbone` feature extraction backbone and
    a `Dense` classifier. The input is an EEG tensor with shape
    `[N, input_channels, input_electrodes, input_times]`, and the output is the
    logits corresponding to each class.

    Attributes:
        backbone (HS_STDCNBackbone): Backbone network used to extract
            spatial-temporal EEG features.
        classifier (Dense): Fully connected classification layer with a max-norm
            constraint.
    """

    def __init__(self,
                 input_channels: int=1, 
                 input_electrodes: int=64,
                 input_times: int=641,    
                 fc_in_channels: int=640,  
                 num_classes: int=5,  
                 temporal_ks_list: list[int]=[7, 19, 31, 63],
                 avg_pool_ks: int=16,
                 F1: int=8, 
                 F2: int=16,
                 dilation_ks: int=3,
                 dilation_list: list[int]=[1, 2],
                 dropout_spatial: float=0.2,
                 dropout_dilated: float=0.3,):
        """Initializes the HS-STDCN classification model.

        Args:
            input_channels (int): Number of channels in the input EEG tensor.
                Defaults to 1.
            input_electrodes (int): Number of EEG electrodes. Defaults to 64.
            input_times (int): Number of EEG temporal samples. Defaults to 641.
            fc_in_channels (int): Input feature dimension of the classifier
                fully connected layer. Defaults to 640.
            num_classes (int): Number of classification classes. Defaults to 5.
            temporal_ks_list (list[int]): List of hybrid-scale temporal
                convolution kernel lengths.
            avg_pool_ks (int): Average pooling kernel length along the temporal
                dimension.
            F1 (int): Number of output channels for each hybrid-scale temporal
                convolution branch.
            F2 (int): Number of output channels for the dilated convolution
                blocks.
            dilation_ks (int): Dilated convolution kernel length.
            dilation_list (list[int]): List of dilation rates used by the two
                dilated convolution layers.
            dropout_spatial (float): Dropout probability after the spatial
                convolution processing stage.
            dropout_dilated (float): Dropout probability inside the dilated
                convolution blocks.
        """

        super().__init__()  # Calls the initializer of the parent nn.Module class.

        # Builds the HS-STDCN backbone network to extract spatial-temporal features from the EEG input.
        self.backbone=HS_STDCNBackbone(                 
            input_channels= input_channels, 
            input_electrodes=input_electrodes,
            input_times=input_times,    
            temporal_ks_list=temporal_ks_list,
            avg_pool_ks=avg_pool_ks,
            F1=F1, 
            F2=F2,
            dilation_ks=dilation_ks,
            dilation_list=dilation_list,
            dropout_spatial=dropout_spatial,
            dropout_dilated=dropout_dilated,)  # Passes in the Dropout probabilities.

        # Builds the fully connected classifier to map backbone output features to class logits.
        self.classifier=Dense(in_channels=fc_in_channels, 
                              out_channels=num_classes,  
                              max_norm=0.25)  

    def forward(self, x):  # Forward pass.
        """Runs the forward pass of HS-STDCN.

        Passes the input EEG tensor through `self.backbone` and
        `self.classifier` in sequence to obtain the final classification output.
        The output is logits without Softmax and can usually be used directly
        with `nn.CrossEntropyLoss`.

        Args:
            x (torch.Tensor): Input tensor, usually with shape
                `[N, input_channels, input_electrodes, input_times]`. A common
                configuration is `[N, 1, C, T]`, where `N` is the batch size,
                `1` is the pseudo-image channel dimension, `C` is the number of
                electrodes, and `T` is the temporal length.

        Returns:
            torch.Tensor: Classification output tensor with shape
            `[N, num_classes]`, representing logits before Softmax.
        """
        x=self.backbone(x)  
        x=self.classifier(x)  
        return x 


    #%%
# Tests the model.
if __name__ == '__main__':
    # Sets the test batch size.
    batch_size = 4
    # Sets the number of EEG electrodes.
    num_electrodes = 64
    # Sets the number of EEG temporal samples.
    seq_len = 641
    # Sets the number of classification classes.
    num_classes = 5

    # Constructs a random input tensor with shape [B, 1, C, T].
    x = torch.randn(batch_size, 1, num_electrodes, seq_len)

    # Instantiates the HS-STDCN model.

    model = HS_STDCN(input_channels=1, 
                     input_electrodes=num_electrodes,
                     input_times=seq_len,    
                     num_classes=num_classes, 
                     fc_in_channels=640,
                     F1=8, 
                     F2=16)

    # Runs the forward pass.
    output = model(x)
    print("模型输出形状：", output.shape)  # Expected to be [batch_size, num_classes].



























