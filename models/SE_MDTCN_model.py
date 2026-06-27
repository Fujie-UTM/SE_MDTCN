# -*- coding: utf-8 -*-
"""SE_MDTCN model implementation for EEG classification.


Last modification: 2026-06-21
@author: Fujie
"""


import torch
import torch.nn as nn
from einops import rearrange
from torch.nn.utils.parametrizations import weight_norm
from typing import Tuple, Union, Optional

#%% Modules

def param_orthogonal_weight_norm(module: nn.Module,
                                 name: str = 'weight',
                                 dim: int = 0) -> nn.Module:
    """Applies orthogonal initialization and weight normalization to a module parameter.

    The specified parameter is initialized with ``nn.init.orthogonal_`` first, and
    then wrapped with ``torch.nn.utils.parametrizations.weight_norm``.

    Args:
        module: Neural network module whose parameter will be initialized and
            normalized.
        name: Name of the parameter to initialize and normalize. Defaults to
            ``'weight'``.
        dim: Dimension over which weight normalization is computed. Defaults to
            ``0``.

    Returns:
        The input module after orthogonal initialization and weight
        normalization have been applied.
    """
    # 1) Orthogonally initialize the underlying tensor.
    # Get the specified parameter from the module. If it exists and is a Tensor,
    # apply orthogonal initialization to it.
    w = getattr(module, name, None)
    if isinstance(w, torch.Tensor):
        nn.init.orthogonal_(w)  # Orthogonally initialize the parameter.
    elif hasattr(w, 'data') and isinstance(w.data, torch.Tensor):
        nn.init.orthogonal_(w.data)  # Orthogonally initialize wrapped tensor data.

    # 2) Apply parametrizations.weight_norm.
    # Normalize the module parameter with weight_norm and return the module.
    return weight_norm(module, name=name, dim=dim)


class ElectrodeNormalization(nn.Module):
    """Normalizes channels at each time step using LayerNorm and einops.

    This module normalizes the input tensor across the channel dimension for
    each time step independently. It is equivalent to applying LayerNorm to all
    channels at every time point.

    Args:
        C: Number of channels in the input tensor, which is also the feature
            dimension at each time step.
        eps: Small value added for numerical stability during normalization.
            Defaults to ``1e-5``.
        elementwise_affine: Whether to apply learnable elementwise affine
            parameters after normalization. Defaults to ``True``.
        bias: Whether to include a bias term in the affine transformation.
            Defaults to ``True``.
    """
    
    def __init__(self, C: int, eps: float = 1e-5, elementwise_affine=True, bias=True):
        super().__init__()
        # Create a LayerNorm layer with normalized_shape set to C so that
        # normalization is applied over the last dimension, i.e., channels.
        self.ln = nn.LayerNorm(normalized_shape=C, eps=eps, elementwise_affine=elementwise_affine, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs channel-wise normalization at each time step.

        Args:
            x: Input tensor with shape ``(B, C, T)``, where ``B`` is the batch
                size, ``C`` is the number of channels, and ``T`` is the number
                of time steps.

        Returns:
            Output tensor with shape ``(B, C, T)``. The shape is the same as
            the input, but the channel dimension has been normalized at each
            time step.
        """
        # Convert the input tensor from (B, C, T) to (B, T, C), swapping the
        # positions of the time and channel dimensions.
        x_t = rearrange(x, 'b c t -> b t c')
        
        # Normalize the C channels at each time step t.
        x_t = self.ln(x_t)
        
        # Convert the normalized tensor from (B, T, C) back to (B, C, T).
        return rearrange(x_t, 'b t c -> b c t')


class DWCausalConv1d(nn.Conv1d):
    """Depthwise causal 1D convolution layer.

    This layer implements causal convolution, where each output at a given time
    step depends only on the current and previous time steps. It also follows
    the depthwise convolution pattern, where each input channel has an
    independent convolution kernel and no bias term is used.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor for the convolution. Defaults to ``1``.
        **kwargs: Additional keyword arguments accepted by the constructor.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        # Calculate the left padding size.
        pad = (kernel_size - 1) * dilation
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=pad,
            dilation=dilation,
            groups=in_channels,  # Use depthwise convolution, one kernel per input channel.
            bias=False  # Do not use a bias term.
        )
        self._left_pad = pad

    def forward(self, x):
        """Applies depthwise causal convolution to the input tensor.

        Args:
            x: Input tensor with shape ``(B, C_in, L)``, where ``B`` is the
                batch size, ``C_in`` is the number of input channels, and ``L``
                is the sequence length.

        Returns:
            Output tensor with shape ``(B, C_out, L)``, where ``C_out`` is the
            number of output channels and ``L`` is the sequence length after
            causal padding adjustment.
        """
        # Call the parent class forward method to perform convolution.
        y = super().forward(x)
        # Remove the padded tail so the output length matches the input length.
        if self._left_pad > 0:
            y = y[:, :, :-self._left_pad]
        return y


class ResidualBlock(nn.Module):
    """Residual block with depthwise causal convolutions and normalization.

    The block contains two depthwise causal convolution layers, ELU activation,
    electrode normalization, dropout, and a residual connection. The input is
    transformed through convolution, normalization, activation, and dropout,
    then added to the residual path to form a skip connection.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        dilation: Dilation factor for the convolution.
        dropout: Dropout probability.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        # First depthwise causal convolution layer.
        self.conv1 = param_orthogonal_weight_norm(
            DWCausalConv1d(in_channels, out_channels, kernel_size, dilation=dilation),
            name='weight', dim=0
        )
        
        # Second depthwise causal convolution layer.
        self.conv2 = param_orthogonal_weight_norm(
            DWCausalConv1d(out_channels, out_channels, kernel_size, dilation=dilation),
            name='weight', dim=0
        )
        
        # First normalization layer.
        self.norm1 = ElectrodeNormalization(C=in_channels, elementwise_affine=True, bias=False)
        
        # Second normalization layer.
        self.norm2 = ElectrodeNormalization(C=in_channels, elementwise_affine=True, bias=False)
        
        # ELU activation function.
        self.act = nn.ELU()
        
        # Dropout layer.
        self.drop = nn.Dropout1d(dropout)
        
        # Residual path. Use a 1x1 convolution to match channels when needed.
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """Runs the residual block forward pass.

        Args:
            x: Input tensor with shape ``(B, C_in, L)``, where ``B`` is the
                batch size, ``C_in`` is the number of input channels, and ``L``
                is the sequence length.

        Returns:
            Output tensor with shape ``(B, C_out, L)``, where ``C_out`` is the
            number of output channels and ``L`` is the sequence length after
            convolution, normalization, activation, and residual addition.
        """
        # Pass the input through the residual path.
        res = self.down(x)

        # Apply the first convolution, normalization, activation, and dropout.
        x = self.drop(self.act(self.norm1(self.conv1(x))))
        # Apply the second convolution, normalization, activation, and dropout.
        x = self.drop(self.act(self.norm2(self.conv2(x))))
        
        # Return the activated sum of the transformed path and residual path.
        return self.act(x + res)


class DWTCNLayer(nn.Module):
    """Depthwise separable temporal causal convolution layer.

    This layer is composed of multiple ``ResidualBlock`` instances. Each block
    contains depthwise causal convolution, normalization, activation, and a
    residual connection. Dilated convolutions are used to extract features at
    multiple temporal scales.

    Args:
        input_channels: Number of input tensor channels.
        n_layers: Number of network layers, i.e., the number of
            ``ResidualBlock`` instances.
        kernel_size: Size of the convolution kernel.
        dropout: Dropout probability.
        dilations: Optional sequence of dilation factors for each layer. If
            ``None``, the sequence ``[1, 2, 4, 8, ...]`` is generated
            automatically. If provided, its length must be at least
            ``n_layers``.
    """
    
    def __init__(self, input_channels, n_layers, kernel_size, dropout, dilations=None):
        super().__init__()
        # 1) Select the dilation sequence.
        if dilations is None:
            dilations = [2**i for i in range(n_layers)]
        else:
            assert len(dilations) >= n_layers, "dilations 长度需要 >= n_layers"
        
        # 2) Build layers according to the dilation sequence.
        layers = []
        for i in range(n_layers):
            layers.append(ResidualBlock(
                in_channels=input_channels,
                out_channels=input_channels,
                kernel_size=kernel_size,
                dilation=dilations[i],
                dropout=dropout
            ))
        
        # Combine all residual blocks into a sequential network.
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Runs the stacked residual blocks.

        Args:
            x: Input tensor with shape ``(B, C_in, L)``, where ``B`` is the
                batch size, ``C_in`` is the number of input channels, and ``L``
                is the sequence length.

        Returns:
            Output tensor with shape ``(B, C_in, L)``. The shape matches the
            input after processing through multiple residual blocks.
        """
        return self.network(x)


class DWTCNBlock(nn.Module):
    """Creates multiple branches for each structure configuration.

    This module creates multiple branches, where each branch is a network built
    from a ``DWTCNLayer``. Each branch can use independent parameters, such as
    the number of layers and dilation factors. Branch outputs are concatenated
    along the channel dimension.

    Args:
        input_channels: Number of input channels.
        branch_params: List of parameter dictionaries for the branches. Each
            dictionary should contain the following fields:

            * ``'n_branches'``: Optional number of branches for the structure.
              Defaults to ``1``.
            * ``'n_layers'``: Number of layers in each ``DWTCNLayer``, i.e.,
              the number of ``ResidualBlock`` instances.
            * ``'kernel_size'``: Size of the convolution kernel.
            * ``'dropout'``: Dropout probability.
            * ``'dilations'``: Optional sequence of dilation factors for each
              layer. The length must be at least ``n_layers``. If ``None``, the
              dilation sequence is generated automatically.
    """
    
    def __init__(self, input_channels, branch_params):
        super().__init__()
        self.branches = nn.ModuleList()
        for p in branch_params:
            n_b = p.get('n_branches', 1)  # Get the number of branches, defaulting to 1.
            for _ in range(n_b):
                branch = DWTCNLayer(
                    input_channels=input_channels,
                    n_layers=p['n_layers'],
                    kernel_size=p['kernel_size'],
                    dropout=p['dropout'],
                    dilations=p.get('dilations', None)  # Added: allow passing a dilation sequence.
                )
                self.branches.append(branch)

    def forward(self, x):
        """Runs all branches and concatenates their outputs.

        Args:
            x: Input tensor with shape ``(B, C_in, L)``, where ``B`` is the
                batch size, ``C_in`` is the number of input channels, and ``L``
                is the sequence length.

        Returns:
            Output tensor with shape ``(B, C_out, L)``, where ``C_out`` is the
            total number of output channels from all branches and ``L`` is the
            sequence length.
        """
        # Run the forward pass for each branch. Each output tensor has shape [B, C, T].
        outs = [b(x) for b in self.branches]
        
        # Concatenate all branch outputs along the channel dimension to obtain
        # an output with shape [B, C * total_branches, T].
        return torch.cat(outs, dim=1)


class SEChannelAttention(nn.Module):
    """Squeeze-and-Excitation channel attention module.

    This module implements Squeeze-and-Excitation (SE) channel attention. It
    uses global statistics to adaptively adjust the importance of each channel.
    Reference paper: https://arxiv.org/abs/1709.01507

    The procedure is as follows:

        1. Apply global variance pooling over the time dimension ``T`` to obtain
           a tensor with shape ``(B, C)``.
        2. Use two fully connected layers with an ELU activation to reduce and
           restore the channel dimension, then use a Sigmoid activation to
           obtain channel attention with shape ``(B, C)``.
        3. Reshape the channel attention so it can be broadcast back to the
           original input shape.

    Args:
        channels: Number of channels in the input tensor.
        reduction: Reduction factor controlling the intermediate dimension,
            where ``mid = max(channels // reduction, 1)``. Defaults to ``2``.
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
        """Computes channel attention from the input tensor.

        Args:
            x: Input tensor with shape ``(B, C, T)``, where ``B`` is the batch
                size, ``C`` is the number of channels, and ``T`` is the number
                of time steps.

        Returns:
            Channel attention tensor with shape ``(B, C, 1)``, broadcastable to
            an input tensor with shape ``(B, C, T)``.
        """
        # x: [B, C, T]
        B, C, T = x.shape
        
        # 1) Global variance pooling -> (B, C).
        y = torch.var(x, dim=-1, unbiased=False)
        
        # 2) Apply the fully connected excitation network -> (B, C).
        y = self.fc(y)
        
        # 3) Reshape channel attention for broadcasting back to the input.
        y = y.view(B, C, 1)  # Reshape y to (B, C, 1) for broadcasting.
        
        # Channel attention.
        attn = y
        return attn


class SE_SpatialFilter(nn.Module):
    """Squeeze-and-Excitation spatial filter.

    This module combines Squeeze-and-Excitation (SE) channel attention with a
    spatial filtering operation. It computes the spatial variability of input
    features and uses channel attention to generate an adaptive spatial filter
    factor. The factor is then used to adjust the response of the input features
    along the spatial dimension, represented here by the time dimension.

    The procedure is as follows:

        1. Compute the variance of the input features to measure spatial
           variability for each channel.
        2. Compute the spatial amplitude ratio from the variability.
        3. Use SE channel attention to generate channel weights.
        4. Compute the spatial filter value and multiply it with the input
           features to obtain the spatially filtered output.

    Args:
        channels: Number of channels in the input tensor.
        reduction: Reduction factor for channel attention. Defaults to ``2``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
    """
    
    def __init__(self, channels: int, reduction: int = 2,
                 spatial_filter_factor: float = 2.0,
                 spatial_filter_factor_learnable: bool = False):
        super().__init__()
        
        # Initialize the SE channel attention module.
        self.se_attn = SEChannelAttention(channels=channels, reduction=reduction)
        
        self.spatial_amp_ratio = None
        
        # Set whether the spatial filter factor is learnable.
        if spatial_filter_factor_learnable:
            self.spatial_filter_factor = nn.Parameter(torch.tensor(spatial_filter_factor, dtype=torch.float32))
        else:
            self.spatial_filter_factor = spatial_filter_factor
        
        self.spatial_filter_value = None

    def cal_amp_ratio(self, x: torch.Tensor):
        """Computes the spatial amplitude ratio for adjusting spatial responses.

        Args:
            x: Input tensor with shape ``(B, C, T)``, where ``B`` is the batch
                size, ``C`` is the number of channels, and ``T`` is the number
                of time steps.
        """
        with torch.no_grad():            
            # Reshape the input tensor and compute variance along the flattened
            # batch-time dimension.
            x_temp = rearrange(x, 'b c t -> c (b t)')
            x_var = torch.var(x_temp, dim=-1, unbiased=False)
            
            # Compute the maximum and minimum variance values.
            x_var_max = torch.max(x_var)
            x_var_min = torch.min(x_var)
            
            # Compute the ratio between the maximum and minimum values.
            spatial_amp_ratio = torch.sqrt(x_var_max / (x_var_min + 1e-9))
            
            # Smooth the spatial amplitude ratio.
            if self.spatial_amp_ratio is None:
                self.spatial_amp_ratio = spatial_amp_ratio
            else:
                self.spatial_amp_ratio = (self.spatial_amp_ratio + spatial_amp_ratio) / 2
                
    def get_spatial_filter_value(self):
        """Gets the current spatial filter value.

        Returns:
            The current spatial filter value detached from the computation
            graph.
        """
        return self.spatial_filter_value.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the spatial filtering forward pass.

        Args:
            x: Input tensor with shape ``(B, C, T)``, where ``B`` is the batch
                size, ``C`` is the number of channels, and ``T`` is the number
                of time steps.

        Returns:
            Output tensor with shape ``(B, C, T)``, adjusted by the spatial
            filter.
        """
        # Compute the spatial amplitude ratio.
        self.cal_amp_ratio(x)
    
        # Compute the spatial filter value and multiply it with the input tensor.
        self.spatial_filter_value = 1 + self.spatial_filter_factor * self.spatial_amp_ratio * self.se_attn(x)
        y = x * self.spatial_filter_value       

        return y


class FeatureProjector(nn.Module):
    """Projects input features into a higher-dimensional space.

    This module applies a linear feature transformation to the input tensor. It
    first normalizes the input with LayerNorm, then expands the feature
    dimension using a fully connected layer. GELU activation and Dropout are
    applied afterward to introduce nonlinearity and reduce overfitting.

    Args:
        input_channels: Number of input channels, or feature dimension.
        fc_expand_dim: Expansion factor used to determine the hidden dimension.
            The hidden dimension is ``input_channels * fc_expand_dim``.
        dropout: Dropout probability used to reduce overfitting. Defaults to
            ``0.2``.
    """
    
    def __init__(self, input_channels, fc_expand_dim, dropout=0.2):
        super().__init__()
        hidden_dim = input_channels * fc_expand_dim
        self.projector = nn.Sequential(
            nn.LayerNorm(normalized_shape=input_channels, 
                         elementwise_affine=True, bias=True),
            nn.Linear(input_channels, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Identity()  # Preserve the shape and pass through the result.
        )

    def forward(self, x):
        """Projects the input tensor.

        Args:
            x: Input tensor with shape ``(B, C)``, where ``B`` is the batch size
                and ``C`` is the number of input channels.

        Returns:
            Output tensor with shape ``(B, hidden_dim)``, where
            ``hidden_dim = input_channels * fc_expand_dim``.
        """
        return self.projector(x)


#%% Complete model   

class SE_MDTCN(nn.Module):
    """Inception-style multi-branch TCN backbone with spatial attention.

    This network combines a multi-branch temporal convolutional network (TCN)
    with a spatial filtering mechanism based on Squeeze-and-Excitation (SE)
    attention. The input tensor is first flattened across channel and electrode
    dimensions, then normalized. Temporal features are extracted through
    multiple TCN branches. For the spatial dimension, an SE-based module
    generates adaptive attention weights before the final classification stage.

    Input:
        x: Input tensor with shape ``(B, chs, els, T)``, where ``B`` is the
            batch size, ``chs`` is the number of channels per electrode,
            ``els`` is the number of electrodes, and ``T`` is the number of
            time steps.

    Output:
        x: Classification output tensor with shape determined by the classifier
            head. Before classification, the feature dimension is
            ``C_all * n_branches * fc_expand_dim``, where
            ``C_all = chs * els``, ``n_branches`` is the number of temporal
            branches, and ``fc_expand_dim`` is the feature expansion factor.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for the temporal
            branches. Each dictionary contains settings such as the number of
            layers and convolution kernel size for a branch configuration.
        spatial_filter_mode: Whether to use spatial attention. Defaults to
            ``'SE'``. If set to ``None``, spatial attention is not used.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        
        # Store input parameters.
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
        
        # Compute the total flattened channel count:
        # channels per electrode multiplied by the number of electrodes.
        self.input_channels_all = self.input_channels * self.input_electrodes

        # Preprocessing: normalize across channels at each time step.
        self.electrode_normalization = ElectrodeNormalization(
            C=self.input_channels_all, 
            elementwise_affine=True,
            bias=True
        )
        
        # ===== Channel attention (SE) and factor reparameterization =====
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

        # Multi-branch TCN backbone.
        self.temporal_filter = DWTCNBlock(
            input_channels=self.input_channels_all,
            branch_params=self.temporal_branch_params
        )

        # Global pooling layer: pools over the time dimension and outputs
        # shape (B, C_all * n_branches).
        self.avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        
        # Classification head: fully connected layers with
        # C_all * number_of_branches as the input dimension.
        self.fc_input_channels = self.input_channels_all * len(self.temporal_filter.branches)
        self.fc_output_channels = self.fc_input_channels * fc_expand_dim
        self.fea_proj = FeatureProjector(input_channels=self.fc_input_channels,
                                         fc_expand_dim=fc_expand_dim,
                                         dropout=fc_dropout)
        self.classifier = nn.Linear(self.fc_output_channels, num_classes, bias=True)
        
        # Checkpoints that can be used for debugging.
        self.ckpt_1 = nn.Identity()
        self.ckpt_2 = nn.Identity()
        self.ckpt_3 = nn.Identity()
        self.ckpt_4 = nn.Identity()
    
    def get_spatial_filter_value(self):
        """Gets the spatial filter value.

        Returns the current spatial filter value when a spatial filter is used.
        Otherwise, prints a message and returns ``None`` implicitly.

        Returns:
            The current spatial filter value if available; otherwise, ``None``.
        """
        if self.spatial_filter != nn.Identity():
            return self.spatial_filter.get_spatial_filter_value()
        else:
            print('non spatial filter')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the forward pass of the model.

        Args:
            x: Input tensor with shape ``(B, chs, els, T)``, where ``B`` is the
                batch size, ``chs`` is the number of channels per electrode,
                ``els`` is the number of electrodes, and ``T`` is the number of
                time steps.

        Returns:
            Classification output tensor with shape ``(B, num_classes)``.
        """
        # Input x: (B, chs, els, T) -> (B, C_all, T).
        if len(x.shape) == 4:
            x = rearrange(x, 'b chs els t -> b (chs els) t')

        # Normalize across channels at each time step.
        x = self.electrode_normalization(x)
        
        # Additionally remove the temporal mean so that each channel has zero
        # mean within each sample.
        x = x - x.mean(dim=-1, keepdim=True)
        
        # Checkpoint 1.
        x = self.ckpt_1(x)
        
        # Spatial filtering.
        x = self.spatial_filter(x)
        
        # Checkpoint 2.
        x = self.ckpt_2(x)

        # Extract TCN features.
        x = self.temporal_filter(x)                      
        
        # Checkpoint 3.
        x = self.ckpt_3(x)

        # Global temporal pooling.
        x = self.avg_pool(x).squeeze(-1)  # Output shape: (B, C_all * n_branches).
        
        # Feature projection.
        x = self.fea_proj(x)
        
        # Checkpoint 4.
        x = self.ckpt_4(x)
        
        # Classification head.
        x = self.classifier(x)

        return x


#%% Ablation experiments

class SE_MDTCN_woElectrodeNormalization(nn.Module):
    """SE-MDTCN ablation model without electrode normalization.

    This wrapper builds an ``SE_MDTCN`` model and replaces its electrode
    normalization module with ``nn.Identity`` so that electrode normalization is
    disabled while the remaining architecture is preserved.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)     


class SE_MDTCN_woSpatialFilter(nn.Module):
    """SE-MDTCN ablation model without spatial filtering.

    This wrapper builds an ``SE_MDTCN`` model and replaces its spatial filter
    module with ``nn.Identity`` so that the spatial filtering component is
    disabled.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)          
 
 
class Power(torch.nn.Module):
    """Elementwise power operation module.

    This module squares each element of the input tensor.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Squares the input tensor elementwise.

        Args:
            x: Input tensor.

        Returns:
            Tensor obtained by applying elementwise square to ``x``.
        """
        return x.pow(2)
    
    
class SE_MDTCN_woTemporalFilter(nn.Module):
    """SE-MDTCN ablation model without the temporal filter.

    This wrapper builds an ``SE_MDTCN`` model and replaces its temporal filter
    with ``Power``. The feature projector and classifier are rebuilt to match
    the resulting channel dimension.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)         
 
    
class SE_MDTCN_wZScoreNormalization(nn.Module):
    """SE-MDTCN variant configured for z-score normalization ablation.

    This wrapper builds an ``SE_MDTCN`` model and replaces its electrode
    normalization module with ``nn.Identity``. The remaining architecture is
    preserved.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)     
    

def max_norm_rows_(w: torch.Tensor, max_norm: float, eps: float = 1e-8):
    """Applies in-place row-wise L2 max-norm constraint.

    The input tensor is flattened from the second dimension onward so that each
    row corresponds to one output filter. Each row is rescaled in place when its
    L2 norm exceeds ``max_norm``.

    Args:
        w: Weight tensor to constrain in place.
        max_norm: Maximum allowed L2 norm for each flattened row.
        eps: Minimum value used to avoid division by zero. Defaults to
            ``1e-8``.
    """
    with torch.no_grad():
        flat = w.view(w.size(0), -1)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(eps)
        desired = norms.clamp(max=max_norm)
        flat.mul_(desired / norms)
        
        
class DepthwiseConv2D(nn.Conv2d):  # Depthwise convolution layer inherited from nn.Conv2d.
    """Depthwise 2D convolution layer.

    This layer fixes ``groups=in_channels`` to perform independent convolution
    for each input channel. The stride is fixed to ``1``. Padding can be
    ``'valid'`` (equivalent to ``0``), ``'same'``, or an explicit integer or
    two-element tuple. Optionally, an L2 max-norm constraint can be applied to
    each filter over its spatial dimensions ``kH * kW`` before the forward
    convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels. Must be an integer multiple of
            ``in_channels`` and corresponds to the depth multiplier.
        kernel_size: Convolution kernel size ``(kH, kW)``.
        max_norm: Optional L2 max-norm threshold. If not ``None``, max-norm is
            applied to each filter before the forward pass.
        padding: Padding mode. Supports ``'valid'``, ``'same'``, or explicit
            padding as an integer or two-element tuple.
        bias: Whether to use a bias term.

    Attributes:
        max_norm: L2 max-norm threshold. ``None`` means the constraint is
            disabled.
        weight: Convolution kernel weights inherited from ``nn.Conv2d`` with
            shape ``[C_out, 1, kH, kW]`` in the depthwise convolution case.
        bias: Optional bias parameter inherited from ``nn.Conv2d`` when
            ``bias=True``.
    """

    def __init__(self,
                 in_channels: int,  # Number of input channels.
                 out_channels: int,  # Number of output channels; must be a multiple of in_channels.
                 kernel_size: Tuple[int, int],  # Convolution kernel size.
                 max_norm: Optional[float] = None,  # Optional L2 max-norm threshold.
                 padding: Union[str, int, Tuple[int, int]] = 'valid',  # Padding mode.
                 bias: bool = False):  # Whether to use a bias term.
        # out_channels must be an integer multiple of in_channels.
        if out_channels % in_channels != 0:  # Validate the depth multiplier divisibility relation.
            raise ValueError(
                f"out_channels ({out_channels}) 必须是 in_channels ({in_channels}) 的整数倍。"
            )

        # Normalize the padding argument.
        if isinstance(padding, str):  # If padding is a string, normalize case before checking it.
            p = padding.lower()  # Normalize string case.
            if p == 'valid':  # 'valid' is equivalent to zero padding.
                padding_arg = 0  # Set padding to 0.
            elif p == 'same':  # 'same' is natively supported by recent PyTorch versions.
                padding_arg = 'same'  # Pass it directly to the parent class.
            else:
                raise ValueError("padding 仅支持 'valid' 或 'same'，或传入具体整数/二元组")
        else:
            padding_arg = padding  # Use integer or tuple padding directly.

        self.max_norm = float(max_norm) if max_norm is not None else None  # Store the max-norm threshold or None.

        super().__init__(in_channels=in_channels,  # Call the parent constructor with input channels.
                         out_channels=out_channels,  # Set output channels.
                         kernel_size=kernel_size,  # Set convolution kernel size.
                         stride=1,  # Use fixed stride 1 for depthwise convolution.
                         padding=padding_arg,  # Use normalized padding.
                         dilation=1,  # Use default dilation 1.
                         groups=in_channels,  # Key setting: groups=in_channels enables depthwise convolution.
                         bias=bias)  # Set whether to use a bias term.

    @property
    def depth_multiplier(self) -> int:  # Read-only property returning the depth multiplier.
        """Returns the depth multiplier.

        Returns:
            The ratio between output channels and input channels.
        """
        return self.out_channels // self.in_channels  # Number of output channels divided by input channels.

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        """Applies depthwise 2D convolution.

        Args:
            x: Input tensor passed to the depthwise convolution.

        Returns:
            Output tensor produced by ``nn.Conv2d.forward``.
        """
        if self.max_norm is not None:
            max_norm_rows_(self.weight, self.max_norm)
        return super().forward(x)  # Use the parent implementation to perform convolution.


class DWSpatialFilter2D(nn.Module):  # Depthwise convolution layer inherited from nn.Conv2d.
    """Depthwise 2D spatial filter.

    This module wraps ``DepthwiseConv2D`` to apply a depthwise spatial filter to
    an input tensor. The input is temporarily expanded with a channel dimension,
    filtered by a 2D depthwise convolution, and then squeezed back.

    Args:
        in_channels: Number of input channels for the internal 2D convolution.
        out_channels: Number of output channels for the internal 2D
            convolution. Must be an integer multiple of ``in_channels``.
        kernel_size: Convolution kernel size for the internal 2D convolution.
    """

    def __init__(self,
                 in_channels: int,  # Number of input channels.
                 out_channels: int,  # Number of output channels; must be a multiple of in_channels.
                 kernel_size: Tuple[int, int]):  # Convolution kernel size.
        super().__init__()
        self.dwconv2D=DepthwiseConv2D(
                 in_channels = in_channels,
                 out_channels = out_channels,
                 kernel_size = kernel_size,
                 max_norm=1.0,
                 padding='valid',
                 bias=False)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        """Applies the depthwise 2D spatial filter.

        Args:
            x: Input tensor to be filtered.

        Returns:
            Filtered tensor after adding a temporary dimension, applying
            depthwise 2D convolution, and squeezing the spatial dimension.
        """
        x = x.unsqueeze(dim=1)
        x = self.dwconv2D(x)
        x = x.squeeze(dim=2)

        return x
    

class SE_MDTCN_wDWConvSpatialFilter2D(nn.Module):
    """SE-MDTCN variant with a depthwise 2D convolution spatial filter.

    This wrapper builds an ``SE_MDTCN`` model and replaces its spatial filter
    with ``DWSpatialFilter2D``.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)     


class TemporalFilter2D(nn.Module):  # Depthwise convolution layer inherited from nn.Conv2d.
    """Temporal 2D convolution filter.

    This module applies a standard 2D convolution along the temporal dimension.
    The input is temporarily expanded with a channel dimension, processed by
    ``nn.Conv2d``, and then squeezed back.

    Args:
        in_channels: Number of input channels for the internal 2D convolution.
        out_channels: Number of output channels for the internal 2D
            convolution.
        kernel_size: Convolution kernel size for the internal 2D convolution.
    """

    def __init__(self,
                 in_channels: int,  # Number of input channels.
                 out_channels: int,  # Number of output channels.
                 kernel_size: Tuple[int, int]):  # Convolution kernel size.
        super().__init__() 
        self.conv2d=nn.Conv2d(in_channels=in_channels,  # Temporal convolution along the time dimension.
                              out_channels=out_channels,  # Number of output channels F1.
                              kernel_size=kernel_size,  # Kernel length.
                              padding='same',  # Keep the time dimension unchanged.
                              bias=False,  # Do not use a bias term.
                              groups=1,)  # Standard convolution, not grouped.
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # Forward pass.
        """Applies the temporal 2D convolution filter.

        Args:
            x: Input tensor to be filtered.

        Returns:
            Filtered tensor after adding a temporary dimension, applying 2D
            convolution, and squeezing the added dimension.
        """
        x = x.unsqueeze(dim=1)
        x = self.conv2d(x)
        x = x.squeeze(dim=1)

        return x


class SE_MDTCN_wTemporalFilter2D(nn.Module):
    """SE-MDTCN variant with an additional temporal 2D filter.

    This wrapper builds an ``SE_MDTCN`` model, stores the original spatial
    filter, and replaces the spatial filter with a sequential module containing
    a temporal 2D filter followed by the original spatial filter.

    Args:
        input_channels: Number of channels per electrode.
        input_electrodes: Number of electrodes.
        input_times: Number of time steps.
        num_classes: Number of target classes.
        temporal_branch_params: List of parameter dictionaries for temporal
            branches.
        spatial_filter_mode: Spatial filter mode. Defaults to ``'SE'``.
        spatial_filter_factor: Initial value of the spatial filter factor.
            Defaults to ``2.0``.
        spatial_filter_factor_learnable: Whether the spatial filter factor is
            learnable. Defaults to ``False``.
        fc_expand_dim: Feature expansion factor. Defaults to ``4``.
        fc_dropout: Dropout probability in the feature projector. Defaults to
            ``0.0``.
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
        """Runs the model forward pass.

        Args:
            x: Input tensor passed to the wrapped ``SE_MDTCN`` model.

        Returns:
            Output tensor produced by the wrapped model.
        """
        return self.model(x)     


#%% Test code
   

if __name__ == "__main__":
    batch_size=8
    num_channel= 1
    num_electrode=64
    num_time=641
    EEG_dummy = torch.randn(batch_size, num_channel, num_electrode, num_time)     

    # Define different parameters for three branch configurations.
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

    
    model =SE_MDTCN_woSpatialFilter(input_channels = num_channel, 
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
    print(f"SE_MDTCN_woSpatialFilter:{model_shape}")   
    
    model =SE_MDTCN_woTemporalFilter(input_channels = num_channel, 
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
    print(f"SE_MDTCN_woTemporalFilter:{model_shape}")   
    
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

    
    
    