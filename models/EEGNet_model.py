# -*- coding: utf-8 -*-  # Specify source file encoding as UTF-8.
"""EEGNet model definition for EEG-based classification.

This module defines an EEGNet implementation in PyTorch, including utility
functions for max-norm constraints, depthwise and separable convolution layers,
the EEGNet feature-extraction backbone, and the final classification model.

Modified from:
    https://github.com/vlawhern/arl-eegmodels

Reference:
    [1] V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon,
    C. P. Hung, and B. J. Lance, "EEGNet: a compact convolutional neural
    network for EEG-based brain-computer interfaces," Journal of Neural
    Engineering, vol. 15, no. 5, p. 056013, 2018/07/27 2018,
    doi: 10.1088/1741-2552/aace8c.

Last modification: 2026-06-21
Author: Fujie
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional


# in_channel: Number of input channels.
# out_channel: Number of output channels.
# k: kernel_size -> Convolution kernel size.
# s: stride -> Stride length.
# p: padding -> Padding size.
# b: bias -> Whether to include a bias term.
# tensor shape: [batch_size, feature channels, electrode channels, time] -> Expected tensor dimensions.

#%% 1. basic module
# Defines an in-place function that constrains the L2 norm of each row in a tensor.
def max_norm_rows_(w: torch.Tensor, max_norm: float, eps: float = 1e-8):
    """Constrains the L2 norm of each tensor row in place.

    This function treats the first dimension of `w` as rows and flattens all
    remaining dimensions into per-row feature vectors. If a row norm is greater
    than `max_norm`, that row is scaled down proportionally; otherwise, it is
    left unchanged. The trailing underscore in the function name indicates that
    the input tensor is modified in place.

    Args:
        w: Tensor to constrain. The first dimension is treated as the row
            dimension, and all remaining dimensions are flattened into each
            row's feature dimension.
        max_norm: Maximum allowed L2 norm for each row.
        eps: Lower bound for row norms to avoid division by zero.

    Returns:
        None. The tensor `w` is modified in place.
    """
    # Performs the in-place modification without tracking gradients, preventing this operation from entering the autograd graph.
    with torch.no_grad():
        # Keeps the first dimension as rows and flattens all remaining dimensions into columns.
        flat = w.view(w.size(0), -1)
        # Computes the L2 norm of each row and clamps it from below by eps to avoid division by zero.
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        # Limits the target norm of each row so it does not exceed max_norm.
        desired = norms.clamp(max=max_norm)
        # Scales each row in place by desired / norms so its norm is no greater than max_norm.
        flat.mul_(desired / norms)
        
        
class DepthwiseConv2D(nn.Conv2d):  # Defines a 2D depthwise convolution layer inherited from nn.Conv2d.
    """Two-dimensional depthwise convolution layer.

    This layer fixes `groups=in_channels`, so each input channel is convolved
    independently. `out_channels` must be an integer multiple of `in_channels`,
    and that ratio is the depth multiplier. The convolution stride is fixed to
    1. Padding can be `'valid'`, `'same'`, or an explicit integer or tuple.

    If `max_norm` is not None, an L2 max-norm constraint is applied to each
    output filter before the forward convolution so that each filter's weight
    norm does not exceed the given threshold.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. Must be an integer multiple of
            `in_channels`.
        kernel_size: Convolution kernel size as `(kH, kW)`.
        max_norm: Optional L2 max-norm threshold. If None, the max-norm
            constraint is disabled.
        padding: Padding mode. Supported values are `'valid'`, `'same'`, an
            integer, or a tuple.
        bias: Whether to include a bias term.

    Attributes:
        max_norm: L2 max-norm threshold, or None when the constraint is disabled.
        weight: Convolution weights inherited from the parent class. For
            depthwise convolution, the shape is typically `[C_out, 1, kH, kW]`.
        bias: Optional bias parameter inherited from the parent class when
            `bias=True`; otherwise, None.
    """

    def __init__(self,
                 in_channels: int,  
                 out_channels: int,  
                 kernel_size: Tuple[int, int],  
                 max_norm: Optional[float] = None, 
                 padding: Union[str, int, Tuple[int, int]] = 'valid',  
                 bias: bool = False):  
        # Checks whether out_channels is divisible by in_channels.
        if out_channels % in_channels != 0: 
            raise ValueError(
                f"out_channels ({out_channels}) 必须是 in_channels ({in_channels}) 的整数倍。"
            )

        # Normalizes the padding argument into a form accepted by nn.Conv2d.
        if isinstance(padding, str): 
            p = padding.lower()  
            if p == 'valid': 
                padding_arg = 0  
            elif p == 'same':  
                padding_arg = 'same' 
            else:
                raise ValueError("padding 仅支持 'valid' 或 'same'，或传入具体整数/二元组")
        else:
            padding_arg = padding 

        self.max_norm = float(max_norm) if max_norm is not None else None  # Stores the max-norm threshold, or None when disabled.

        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels,  
                         kernel_size=kernel_size, 
                         stride=1, 
                         padding=padding_arg,  
                         dilation=1, 
                         groups=in_channels,  
                         bias=bias)  

    @property  # Declares the method as a read-only property.
    def depth_multiplier(self) -> int:  # Defines the depth multiplier property.
        """Returns the depth multiplier.

        The depth multiplier indicates how many output channels are generated
        for each input channel.

        Returns:
            Integer ratio between the number of output channels and input channels.
        """
        return self.out_channels // self.in_channels  

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        """Runs the forward pass of the 2D depthwise convolution.

        If `self.max_norm` is not None, this method first applies the in-place
        L2 max-norm constraint to the convolution weights, and then delegates the
        actual convolution computation to the parent class implementation.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after depthwise convolution.
        """
        if self.max_norm is not None:  
            max_norm_rows_(self.weight, self.max_norm)  
        return super().forward(x)  
    

class SeparableConv2D(nn.Module):  
    """Depthwise separable 2D convolution module.

    This module first applies depthwise convolution, where each channel is
    convolved independently and the channel count is preserved. It then applies
    a 1x1 pointwise convolution to project the channel dimension to the desired
    output channel count. Both steps use stride 1, and the depthwise convolution
    uses `'same'` padding.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels produced by the pointwise convolution.
        kernel_size: Depthwise convolution kernel size as `(kH, kW)`.

    Attributes:
        conv: Sequential container that wraps the depthwise and pointwise
            convolution layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size) -> None:  # Constructor.
        """Initializes the depthwise separable convolution module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Depthwise convolution kernel size as `(kH, kW)`.
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
                      kernel_size=1,  # Uses a 1x1 convolution.
                      stride=1,  # Uses stride 1.
                      padding='same',  # Uses same padding to preserve spatial dimensions.
                      bias=False),  # Disables the bias term.
        )

    def forward(self, x):  # Forward propagation.
        """Runs the forward computation.

        Args:
            x: Input tensor with shape `[N, C_in, H, W]`.

        Returns:
            Output tensor with shape `[N, C_out, H, W]`; same padding preserves
            the height and width dimensions.
        """
        return self.conv(x)  


class EEGNetBackbone(nn.Module): 
    """EEGNet backbone network.

    The backbone contains two convolution-normalization-activation-pooling
    stages. The first stage includes temporal convolution and cross-electrode
    depthwise convolution. The second stage uses separable convolution. The final
    flattening operation produces features for the classification head.

    Args:
        input_channels: Number of input pseudo-image channels, usually 1 for EEG.
        input_electrodes: Number of EEG electrodes, corresponding to spatial
            dimension `C`.
        input_times: Temporal length `T` of each EEG segment.
        fs: Sampling rate, used to set temporal convolution kernel lengths.
        F1: Number of base filters in the first temporal convolution stage.
        D: Depth multiplier for the depthwise convolution.
        F2: Number of output channels from the separable convolution, often
            chosen as `D * F1`.
        p_drop: Dropout probability.

    Attributes:
        conv_ks_1_1: First-stage temporal convolution kernel size `(1, fs//2)`.
        pool_ks_1_1: First-stage pooling kernel size `(1, 4)`.
        conv_ks_2_1: Second-stage depthwise temporal kernel size `(1, fs//8)`.
        pool_ks_2_1: Second-stage pooling kernel size `(1, 8)`.
        block1: First sequential stage: Conv -> BN -> Depthwise -> BN -> ELU ->
            Pool -> Dropout.
        block2: Second sequential stage: Separable -> BN -> ELU -> Pool ->
            Dropout -> Flatten.
    """

    def __init__(self, 
                 input_channels = 1, 
                 input_electrodes = 64,  
                 input_times = 512,  
                 fs=256, 
                 F1=8,  
                 D=4,  
                 F2=32, 
                 p_drop=0.25):  
        super().__init__()  
        self.input_channels=input_channels  
        self.input_electrodes=input_electrodes  
        self.input_times=input_times  

        self.F1 = F1  
        self.D = D    
        self.F2 = F2   
        self.p_drop = p_drop   
        
        self.conv_ks_1_1=(1, int(fs//2))  
        self.pool_ks_1_1=(1, 4)  
        self.conv_ks_2_1=(1, max(1, fs//8))  
        self.pool_ks_2_1=(1, 8)  

        
        # Block 1.
        self.block1 = nn.Sequential( 
            nn.Conv2d(in_channels=self.input_channels, 
                      out_channels=self.F1,  
                      kernel_size=self.conv_ks_1_1,  
                      padding='same', 
                      bias=False,  
                      groups=1,),  
            nn.BatchNorm2d(self.F1),  
            DepthwiseConv2D(in_channels=self.F1,  
                            out_channels=(self.D * self.F1), 
                            kernel_size=(self.input_electrodes, 1), 
                            max_norm=1.0),  
            
            nn.BatchNorm2d(self.D*self.F1), 
            nn.ELU(inplace=True), 
            nn.AvgPool2d(kernel_size=self.pool_ks_1_1),  
            nn.Dropout(self.p_drop)  
            )
        
        self.block2 = nn.Sequential(  
            SeparableConv2D(in_channels=(self.D * self.F1),  
                            out_channels=self.F2,  
                            kernel_size=self.conv_ks_2_1),  
            nn.BatchNorm2d(self.F2),  
            nn.ELU(inplace=True), 
            nn.AvgPool2d(kernel_size= self.pool_ks_2_1),  
            nn.Dropout(self.p_drop),  
            nn.Flatten()  
            )
            
    def forward(self, x):  # Forward pass for EEGNet.
        """Runs the forward pass.

        The input tensor is passed through Block 1 and Block 2 in sequence to
        produce flattened features.

        Args:
            x: Input tensor, typically with shape `[N, 1, C, T]`, where `N` is
                the batch size, `1` is the pseudo-image channel dimension, `C`
                is the number of electrodes, and `T` is the temporal length.

        Returns:
            Flattened feature tensor with approximate shape `[N, F2*(T/32)]`.
        """
        x= self.block1(x)  
        x= self.block2(x) 
        return x  


class Dense(nn.Linear):  # Fully connected layer inherited from nn.Linear with an optional L2 max-norm constraint.
    """Linear layer with an optional L2 max-norm weight constraint.

    If the input tensor has more than two dimensions, it is automatically
    flattened to `[N, in_features]` before being passed to the linear layer.
    Before the forward computation, this layer applies an L2 max-norm constraint
    to the weights row-wise, where each row corresponds to one output unit.

    Args:
        in_channels: Input feature dimension, corresponding to `in_features`.
        out_channels: Output feature dimension, corresponding to `out_features`.
        max_norm: Optional L2 max-norm threshold. If None, weight clipping is
            disabled.
        bias: Whether to include a bias term.

    Attributes:
        max_norm: L2 max-norm threshold, or None when disabled.
        weight: Linear-layer weight parameter with shape
            `[out_features, in_features]`.
        bias: Optional bias parameter with shape `[out_features]` when enabled.
    """
    def __init__(self,
                 in_channels: int,  
                 out_channels: int, 
                 max_norm: Optional[float] = 0.25, 
                 bias: bool = True) -> None:  
        super().__init__(in_features=in_channels,  
                         out_features=out_channels,  
                         bias=bias)  
        self.max_norm = float(max_norm) if max_norm is not None else None  
        
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Runs the dense-layer forward pass.

        Args:
            x: Input tensor. If it has more than two dimensions, all dimensions
                except the batch dimension are flattened before the linear layer.

        Returns:
            Output tensor after the optional max-norm constraint and linear
            transformation.
        """
        # Automatically flattens to [N, in_features].
        if x.dim() > 2:  # If the input is multidimensional, flattens all dimensions except the batch dimension.
            x = x.view(x.size(0), -1)

        if self.max_norm is not None:
            max_norm_rows_(self.weight, self.max_norm)

        # Uses the parent implementation to complete the linear transformation.
        return super().forward(x)  # Returns the linear transformation result.


#%% 2. full mode

class EEGNet(nn.Module): 
    """Top-level EEGNet model with a backbone and linear classification head.

    This class wraps an `EEGNetBackbone` for convolutional feature extraction
    and a `Dense` classifier for class prediction. The typical computation flow
    is: input tensor -> backbone feature extraction with convolution, pooling,
    and flattening -> classifier output class scores.

    Args:
        input_channels: Number of input channels, usually 1 for the pseudo-image
            channel dimension.
        input_electrodes: Number of EEG electrodes, corresponding to spatial
            dimension `C`.
        input_times: Temporal length `T` of each EEG segment.
        fc_in_channels: Input feature dimension of the classifier `Dense` layer;
            this must match the flattened output size of the backbone.
        num_classes: Number of target classes; this is the classifier output
            dimension.
        fs: Sampling rate passed to `EEGNetBackbone` to set temporal kernel
            lengths.
        F1: Number of base filters in the first convolution block passed to
            `EEGNetBackbone`.
        D: Depth multiplier passed to `EEGNetBackbone`.
        F2: Number of output channels in the second separable convolution block
            passed to `EEGNetBackbone`.
        p_drop: Dropout probability passed to `EEGNetBackbone`.

    Attributes:
        backbone: Feature-extraction backbone for convolution, pooling, and
            flattening.
        classifier: Linear classification head that maps `fc_in_channels` to
            `num_classes`.
    """

    def __init__(self, 
                 input_channels = 1,  
                 input_electrodes = 64, 
                 input_times = 641,  
                 fc_in_channels=320,  
                 num_classes=5,  
                 fs=256, 
                 F1=8, 
                 D=2,  
                 F2=16,  
                 p_drop=0.5,):  
        super().__init__()  
        
        self.backbone=EEGNetBackbone(input_channels = input_channels, 
                                    input_electrodes = input_electrodes,  
                                    input_times = input_times,  
                                    fs=fs, 
                                    F1=F1, 
                                    D=D,  
                                    F2=F2, 
                                    p_drop=p_drop)  
        self.classifier=Dense(in_channels=fc_in_channels,  
                              out_channels=num_classes, 
                              max_norm=0.25)  

    def forward(self, x):  
        """Runs the forward pass.

        The input tensor is passed through `self.backbone` and then
        `self.classifier` to produce the final classification output.

        Args:
            x: Input tensor, typically with shape
                `[N, input_channels, input_electrodes, input_times]`. A common
                setting is `[N, 1, C, T]`, where `N` is the batch size, `1` is
                the pseudo-image channel dimension, `C` is the number of
                electrodes, and `T` is the temporal length.

        Returns:
            Classification output tensor with shape `[N, num_classes]`,
            containing logits before softmax.
        """
        x=self.backbone(x)  
        x=self.classifier(x)  
        return x 


#%%
if __name__ == "__main__":

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = EEGNet(input_channels=1,
                   input_electrodes=64,
                   input_times=641,
                   fc_in_channels=320,
                   num_classes=5,
                   fs=256, F1=8, D=2, F2=16, p_drop=0.5).to(device)
    
    x = torch.randn(8, 1, 64, 641, device=device)
    logits = model(x)
    print('logits shape:', logits.shape)  # [8, 5]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    
