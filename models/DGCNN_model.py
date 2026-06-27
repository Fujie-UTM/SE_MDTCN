# -*- coding: utf-8 -*-
"""DGCNN model implementation for EEG classification.
Modefied from
https://github.com/XJTU-EEG/LibEER
https://github.com/KeiraLalala/DGCNN_EEG_EmotionRecognition

[1]	T. Song, W. Zheng, P. Song, and Z. Cui, "EEG Emotion Recognition Using
 Dynamical Graph Convolutional Neural Networks," IEEE Transactions on Affective
 Computing, vol. 11, no. 3, pp. 532-541, 2020, doi: 10.1109/TAFFC.2018.2817622.

[2]	H. Liu et al., "LibEER: A Comprehensive Benchmark and Algorithm Library for
 EEG-Based Emotion Recognition," IEEE Transactions on Affective Computing, 
 vol. 16, no. 4, pp. 3596-3613, 2025, doi: 10.1109/TAFFC.2025.3605833.

Last modification: 2026-06-21
@author: Fujie
"""

import torch
import torch.nn as nn
import torch.utils.data

import math
from einops import rearrange
from typing import Optional, Sequence

#%% 1. basic module

class DifferentialEntropy(nn.Module):
    """Computes differential entropy features along the temporal dimension.

    The expected input tensor shape is `(B, C, E, T)`, and the default
    output shape is `(B, C, E)`.

    Args:
        eps: Small constant used to avoid `log(0)` when the variance is zero.
        unbiased: Whether to use the unbiased variance estimator.
        keepdim: Whether to keep the temporal dimension.
    """

    def __init__(
        self,
        eps: float = 1e-6,
        unbiased: bool = False,
        keepdim: bool = False,
    ):
        super().__init__()
        self.eps = eps
        self.unbiased = unbiased
        self.keepdim = keepdim
        self.log_2pi_e = math.log(2 * math.pi * math.e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes differential entropy.

        Args:
            x: Input tensor with shape `(B, C, E, T)`.

        Returns:
            Differential entropy tensor with shape `(B, C, E)` or
            `(B, C, E, 1)`.
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected input shape (B, C, E, T), but got {tuple(x.shape)}"
            )

        var = torch.var(
            x,
            dim=-1,
            unbiased=self.unbiased,
            keepdim=self.keepdim,
        )

        de = 0.5 * (torch.log(var + self.eps) + self.log_2pi_e)

        return de


# Computes the scaled normalized graph Laplacian for an adjacency matrix.
def laplacian(w, eps=1e-6):
    """Computes the scaled normalized graph Laplacian for an adjacency matrix.

    Compared with the original form:
        L = I - D^{-1/2} W D^{-1/2}

    This implementation explicitly uses:
        L = D - W
        L_norm = D^{-1/2} L D^{-1/2}
        L_tilde = 2 * L_norm / lambda_max - I

    For the normalized Laplacian, `lambda_max` is usually set to 2.0.

    Args:
        w (torch.Tensor): Adjacency matrix with shape `[N, N]`.
        eps (float): Small constant used to avoid division by zero.

    Returns:
        torch.Tensor: Scaled normalized Laplacian matrix with shape `[N, N]`.
    """
    # Number of nodes
    n = w.size(0)
    device = w.device
    dtype = w.dtype

    # 1. Compute the degree vector d_i = sum_j w_ij
    d = torch.sum(w, dim=1)

    # 2. Avoid division by zero for isolated nodes
    d_safe = d.clamp_min(eps)

    # 3. Construct the degree matrix D
    D = torch.diag(d_safe)

    # 4. Compute the unnormalized Laplacian L = D - W
    L = D - w

    # 5. Compute D^{-1/2}
    d_inv_sqrt = torch.pow(d_safe, -0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)

    # 6. Compute the normalized Laplacian L_norm = D^{-1/2} L D^{-1/2}
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    # 7. Scale to [-1, 1] for Chebyshev polynomials
    # For a normalized Laplacian, the maximum eigenvalue is theoretically no greater than 2
    lambda_max = 2.0

    I = torch.eye(n, device=device, dtype=dtype)

    # L_tilde = 2L_norm / lambda_max - I
    L_tilde = (2.0 * L_norm) / lambda_max - I

    return L_tilde


# Defines a Chebyshev polynomial-based graph convolution layer.
class GraphConv(nn.Module):
    """Chebyshev polynomial-based graph convolution layer.

    This layer first constructs polynomial components of the input graph
    signal at different Chebyshev orders, and then performs K-order graph
    filtering with learnable weights.

    Attributes:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output feature channels.
        k (int): Chebyshev polynomial order.
        weight (nn.Parameter): Graph convolution filter weights.
    """

    # Initializes the graph convolution layer.
    def __init__(self, k, in_channels, out_channels):
        """Initializes the graph convolution layer.

        Args:
            k (int): Chebyshev polynomial order.
            in_channels (int): Number of input feature channels.
            out_channels (int): Number of output feature channels.
        """
        # Calls the parent nn.Module initializer.
        super(GraphConv, self).__init__()
        # Stores the number of input channels.
        self.in_channels = in_channels
        # Stores the number of output channels.
        self.out_channels = out_channels
        # Stores the Chebyshev polynomial order.
        self.k = k
        # Defines the learnable graph convolution weights with input dimension k * in_channels.
        self.weight = nn.Parameter(torch.Tensor(k * in_channels, out_channels))
        # Initializes graph convolution weights with Xavier uniform initialization.
        nn.init.xavier_uniform_(self.weight)
        # Optional truncated normal initialization is commented out and kept from the original code.
        # self.truncated_normal_(self.weight)

    # Computes Chebyshev polynomial components from input features and the graph Laplacian.
    def chebyshev_polynomial(self, x, lap):
        """Explicitly generates Chebyshev polynomial terms and applies them to `x`.

        Args:
            x: Input graph signal with shape `[B, N, Fin]`, where `B` is the
                batch size, `N` is the number of EEG electrodes or graph nodes,
                and `Fin` is the input feature dimension of each node.
            lap: Scaled Laplacian matrix with shape `[N, N]`.

        Returns:
            Chebyshev graph signal terms with shape `[B, K, N, Fin]`, where
            `out[:, 0] = T0(lap) x = I x = x`,
            `out[:, 1] = T1(lap) x = lap x`, and
            `out[:, 2] = T2(lap) x = (2 lap^2 - I) x`.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x shape [B, N, Fin], but got {tuple(x.shape)}")
    
        if lap.dim() != 2 or lap.size(0) != lap.size(1):
            raise ValueError(f"Expected lap shape [N, N], but got {tuple(lap.shape)}")
    
        B, N, Fin = x.shape
    
        if lap.size(0) != N:
            raise ValueError(
                f"lap node size {lap.size(0)} does not match x node size {N}"
            )
    
        # ============================================================
        # 1. Explicitly generate Chebyshev matrix terms before applying them:
        #    T0(L), T1(L), ..., T_{K-1}(L)
        # ============================================================
    
        cheb_operators = []
    
        # T0(L) = I
        T0 = torch.eye(N, device=x.device, dtype=x.dtype)
        cheb_operators.append(T0)
    
        if self.k > 1:
            # T1(L) = L
            T1 = lap
            cheb_operators.append(T1)
    
        for _ in range(2, self.k):
            # Tk(L) = 2 L T_{k-1}(L) - T_{k-2}(L)
            Tk = 2 * torch.matmul(lap, cheb_operators[-1]) - cheb_operators[-2]
            cheb_operators.append(Tk)
    
        # ============================================================
        # 2. Apply each matrix term to the input x:
        #    T0(L)x, T1(L)x, ..., T_{K-1}(L)x
        # ============================================================
    
        cheb_signals = []
    
        for T in cheb_operators:
            # T: [N, N]
            # x: [B, N, Fin]
            # torch.matmul(T, x) -> [B, N, Fin]
            Tx = torch.matmul(T, x)
            cheb_signals.append(Tx)
    
        # [K tensors with shape [B, N, Fin]] -> [B, K, N, Fin]
        out = torch.stack(cheb_signals, dim=1)
    
        return out

    # Defines the forward pass of the graph convolution layer.
    def forward(self, x, lap):
        """Runs the graph convolution forward pass.

        Args:
            x (torch.Tensor): Input features with shape
                `(batch_size, ele_channel, in_channel)`.
            lap (torch.Tensor): Graph Laplacian matrix.

        Returns:
            torch.Tensor: Graph convolution output.
        """
        # Gets Chebyshev polynomial components; cp has shape (batch, k, ele_channel, in_channel).
        cp = self.chebyshev_polynomial(x, lap)
        # Permutes cp to (batch, ele_channel, in_channel, k).
        cp = cp.permute(0, 2, 3, 1)
        # Reshapes cp to (batch, ele_channel, in_channel * k).
        cp = cp.flatten(start_dim=2)
        # Performs the K-order filtering operation.
        out = torch.matmul(cp, self.weight)
        # Returns the graph convolution output.
        return out

# Defines a new sparse L2 regularization module.
class NewSparseL2Regularization(nn.Module):
    """Computes an L2 regularization term for all model parameters.

    Attributes:
        l2_lambda (float): L2 regularization coefficient.
    """

    # Initializes the new sparse L2 regularization module.
    def __init__(self, l2_lambda):
        """Initializes the new sparse L2 regularization module.

        Args:
            l2_lambda (float): L2 regularization coefficient.
        """
        # Calls the parent nn.Module initializer.
        super(NewSparseL2Regularization, self).__init__()
        # Stores the L2 regularization coefficient.
        self.l2_lambda = l2_lambda
    # Computes the L2 regularization term for all parameters of the input model.
    def forward(self, x):
        """Computes the L2 regularization loss for model parameters.

        Args:
            x (nn.Module): Model or module to regularize.

        Returns:
            torch.Tensor: Weighted L2 regularization loss.
        """
        # Creates an initial zero regularization tensor on the model parameter device.
        l2_reg = torch.tensor(0.).to(next(x.parameters()).device)
        # Iterates over all parameters in the model.
        for param in x.parameters():
            # Accumulates the norm of each parameter tensor.
            l2_reg += torch.norm(param)
        # Multiplies by the L2 regularization coefficient and returns the result.
        return l2_reg * self.l2_lambda

# Defines a module that computes an L2 regularization term for a single input tensor.
class SparseL2Regularization(nn.Module):
    """Computes an L2 regularization term for an input tensor.

    Attributes:
        l2_lambda (float): L2 regularization coefficient.
    """

    # Initializes the sparse L2 regularization module.
    def __init__(self, l2_lambda):
        """Initializes the sparse L2 regularization module.

        Args:
            l2_lambda (float): L2 regularization coefficient.
        """
        # Calls the parent nn.Module initializer.
        super(SparseL2Regularization, self).__init__()
        # Stores the L2 regularization coefficient.
        self.l2_lambda = l2_lambda

    # Computes the L2 regularization term for the input tensor.
    def forward(self, x):
        """Computes the L2 regularization loss for the input tensor.

        Args:
            x (torch.Tensor): Input tensor to regularize.

        Returns:
            torch.Tensor: Weighted L2 regularization loss.
        """
        # Computes the L2 norm of the input tensor.
        l2_norm = torch.norm(x, p=2)
        # Returns the L2 norm multiplied by the regularization coefficient.
        return self.l2_lambda * l2_norm


# Defines the first ReLU activation module with a learnable bias.
class B1ReLU(nn.Module):
    """Channel-level ReLU activation module with a learnable bias.

    Attributes:
        bias (nn.Parameter): Learnable bias with shape `(1, 1, bias_shape)`.
        relu (nn.ReLU): ReLU activation function.
    """

    # Initializes the B1ReLU module.
    def __init__(self, bias_shape):
        """Initializes the B1ReLU module.

        Args:
            bias_shape (int): Size of the bias tensor along the channel dimension.
        """
        # Calls the parent nn.Module initializer.
        super(B1ReLU, self).__init__()
        # Defines the learnable bias with shape (1, 1, bias_shape).
        self.bias = nn.Parameter(torch.Tensor(1, 1, bias_shape))
        # Creates the ReLU activation function.
        self.relu = nn.ReLU()
        # Initializes the learnable bias to zero.
        nn.init.zeros_(self.bias)

    # Runs the biased ReLU activation.
    def forward(self, x):
        """Runs the B1ReLU forward pass.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Output after adding the learnable bias and applying ReLU.
        """
        # Adds the learnable bias to the input tensor, applies ReLU, and returns the result.
        return self.relu(self.bias + x)


# Defines the second ReLU activation module with a learnable bias.
class B2ReLU(nn.Module):
    """Electrode-level and channel-level ReLU activation module with a learnable bias.

    Attributes:
        bias (nn.Parameter): Learnable bias with shape `(1, bias_shape1, bias_shape2)`.
        relu (nn.ReLU): ReLU activation function.
    """

    # Initializes the B2ReLU module.
    def __init__(self, bias_shape1, bias_shape2):
        """Initializes the B2ReLU module.

        Args:
            bias_shape1 (int): Size of the bias tensor along the electrode dimension.
            bias_shape2 (int): Size of the bias tensor along the channel dimension.
        """
        # Calls the parent nn.Module initializer.
        super(B2ReLU, self).__init__()
        # Defines the learnable bias with shape (1, bias_shape1, bias_shape2).
        self.bias = nn.Parameter(torch.Tensor(1, bias_shape1, bias_shape2))
        # Creates the ReLU activation function.
        self.relu = nn.ReLU()
        # Initializes the learnable bias to zero.
        nn.init.zeros_(self.bias)

    # Runs the biased ReLU activation.
    def forward(self, x):
        """Runs the B2ReLU forward pass.

        Args:
            x (torch.Tensor): Input feature tensor.

        Returns:
            torch.Tensor: Output after adding the learnable bias and applying ReLU.
        """
        # Adds the learnable bias to the input tensor, applies ReLU, and returns the result.
        return self.relu(self.bias + x)
    

# Defines the DGCNN model class that inherits from PyTorch's nn.Module.
class DGCNNBackbone(nn.Module):
    """Dynamical graph convolutional neural network for EEG emotion recognition.

    This model uses a learnable adjacency matrix to construct the graph
    structure among EEG electrodes, extracts spatial relationship features with
    Chebyshev polynomial-based graph convolution, and performs classification
    with fully connected layers.

    Attributes:
        dropout_rate (float): Dropout probability.
        layers (list[int] | None): Output channels for each graph convolution layer.
        k (int): Chebyshev polynomial order.
        in_channels (int): Input feature dimension of each electrode.
        num_electrodes (int): Number of EEG electrodes.
        num_classes (int): Number of classes.
        relu_is (int): Type of biased ReLU module to use.
        graphConvs (nn.ModuleList): List of graph convolution layers.
        fc (nn.Linear): First fully connected classification layer.
        fc2 (nn.Linear): Second fully connected classification layer.
        adj (nn.Parameter): Learnable adjacency matrix parameter.
        adj_bias (nn.Parameter): Learnable bias for the adjacency matrix.
        relu (nn.ReLU): ReLU used to constrain the adjacency matrix to be non-negative.
        b_relus (nn.ModuleList): List of ReLU layers with learnable biases.
        dropout (nn.Dropout): Dropout layer.
    """

    # Initializes the DGCNN model and all of its submodules.
    def __init__(self, num_electrodes=62, in_channels=5, num_classes=3, k=2, relu_is=1, layers=[64], dropout_rate=0.5):
        """Initializes the DGCNN model.

        Args:
            num_electrodes (int): Number of EEG electrodes.
            in_channels (int): Input feature dimension of each electrode.
            num_classes (int): Number of classes to predict.
            k (int): Chebyshev polynomial order.
            relu_is (int): Type of biased ReLU to use; 1 indicates B1ReLU and
                2 indicates B2ReLU.
            layers (list[int] | None): Output channels of each graph convolution
                layer.
            dropout_rate (float): Dropout probability.
        """
        # num_electrodes(int): Number of electrodes.
        # in_channels(int): Feature dimension of each electrode.
        # num_classes(int): Number of classes to predict.
        # k_(int): Chebyshev polynomial order used in the graph convolution layer.
        # relu_is(int): Type of activation function to use.
        # out_channel(int): Dimension of graph features after the GCN.
        # Calls the parent nn.Module initializer.
        super().__init__()

        # Stores the dropout probability.
        self.dropout_rate = dropout_rate
        # Stores the channel configuration of graph convolution layers.
        self.layers = layers
        # Stores the Chebyshev polynomial order.
        self.k = k
        # Stores the input feature dimension of each electrode.
        self.in_channels = in_channels
        # Stores the number of EEG electrodes.
        self.num_electrodes = num_electrodes
        # Stores the number of classes.
        self.num_classes = num_classes
        # Stores the type index of the biased ReLU module.
        self.relu_is = relu_is

        # Creates a ModuleList for multiple graph convolution layers.
        self.graphConvs = nn.ModuleList()
        # Adds the first graph convolution layer, mapping input features to the first output channels.
        self.graphConvs.append(GraphConv(self.k, self.in_channels, self.layers[0]))
        # Iterates over the remaining layer configuration and builds subsequent graph convolution layers.
        for i in range(len(self.layers) - 1):
            # Adds the i+1-th graph convolution layer, mapping the previous channels to the next channels.
            self.graphConvs.append(GraphConv(self.k, self.layers[i], self.layers[i + 1]))


        # Defines the first fully connected layer, mapping flattened electrode graph features to 256 dimensions.
        self.fc = nn.Linear(self.num_electrodes * self.layers[-1], 256, bias=True)
        # Defines the second fully connected layer, mapping 256-dimensional features to class logits.
        self.fc2 = nn.Linear(256, self.num_classes, bias=True)
        # Defines the learnable adjacency matrix parameter with shape num_electrodes by num_electrodes.
        self.adj = nn.Parameter(torch.Tensor(self.num_electrodes, self.num_electrodes))
        # Defines the learnable adjacency matrix bias parameter.
        self.adj_bias = nn.Parameter(torch.Tensor(1))
        # Defines an in-place ReLU used to constrain the adjacency matrix to be non-negative.
        self.relu = nn.ReLU(inplace=True)
        # Creates a ModuleList for ReLU layers with learnable biases.
        self.b_relus = nn.ModuleList()
        # Uses B1ReLU if relu_is is 1.
        if self.relu_is == 1:
            # Creates one B1ReLU for each graph convolution output layer.
            for i in range(len(self.layers)):
                # Adds a biased ReLU for the channel dimension.
                self.b_relus.append(B1ReLU(self.layers[i]))
        # Uses B2ReLU if relu_is is 2.
        elif self.relu_is == 2:
            # Creates one B2ReLU for each graph convolution output layer.
            for i in range(len(self.layers)):
                # Adds a biased ReLU for the electrode and channel dimensions.
                self.b_relus.append(B2ReLU(self.adj.shape[0], self.layers[i]))
        # Creates the Dropout layer.
        self.dropout = nn.Dropout(p=self.dropout_rate)
        # Initializes learnable parameters in the model.
        self.init_weight()


    # Initializes weights and biases in the DGCNN model.
    def init_weight(self):
        """Initializes model parameters in place.

        Xavier initialization is used for the adjacency matrix and fully
        connected layer weights. A truncated normal distribution is used for the
        adjacency matrix bias, and fully connected layer biases are initialized
        to zero.
        """
        # Initializes the learnable adjacency matrix with Xavier uniform initialization.
        nn.init.xavier_uniform_(self.adj)
        # Initializes the adjacency matrix bias with a truncated normal distribution.
        nn.init.trunc_normal_(self.adj_bias, mean=0, std=0.1)
        # Initializes the first fully connected layer weights with Xavier normal initialization.
        nn.init.xavier_normal_(self.fc.weight)
        # Initializes the first fully connected layer bias to zero.
        nn.init.zeros_(self.fc.bias)
        # Initializes the second fully connected layer weights with Xavier normal initialization.
        nn.init.xavier_normal_(self.fc2.weight)
        # Initializes the second fully connected layer bias to zero.
        nn.init.zeros_(self.fc2.bias)

    # Defines the model forward pass.
    def forward(self, x):
        """Runs the DGCNN forward pass.

        Args:
            x (torch.Tensor): Input EEG feature tensor, usually with shape
                `(batch_size, num_electrodes, in_channels)`.

        Returns:
            torch.Tensor: Classification logits with shape `(batch_size, num_classes)`.
        """
        # Adds the learnable adjacency matrix and bias, then applies ReLU to obtain a non-negative adjacency matrix.
        adj = self.relu(self.adj + self.adj_bias)
        # Computes the normalized graph Laplacian from the adjacency matrix.
        lap = laplacian(adj)
        # Sequentially applies each graph convolution layer, Dropout layer, and biased ReLU layer.
        for i in range(len(self.layers)):
            # Applies the i-th graph convolution operation to the input features.
            x = self.graphConvs[i](x, lap)
            # Applies Dropout to the graph convolution output.
            x = self.dropout(x)
            # Applies biased ReLU activation to the features after Dropout.
            x = self.b_relus[i](x)

        # Flattens the graph convolution output into a 2D tensor while preserving the batch dimension.
        x = x.reshape(x.shape[0], -1)
        # Applies Dropout to the flattened features.
        x = self.dropout(x)
        # Obtains 256-dimensional features through the first fully connected layer.
        x = self.fc(x)
        # Applies Dropout to the output of the first fully connected layer.
        x = self.dropout(x)
        # Obtains final classification logits through the second fully connected layer.
        x = self.fc2(x)
        # Returns the model output.
        return x


#%% 2. full mode

class DGCNN(nn.Module):
    """Wrapper module for the DGCNN model.

    This module processes raw EEG time-series input. The input data first passes
    through Differential Entropy (DE) feature extraction to compress the temporal
    dimension into a feature representation. The tensor is then reshaped into the
    format required by DGCNNBackbone and fed into the dynamical graph convolution
    backbone for classification.
    """

    def __init__(
        self,
        input_channels: int = 1,
        input_electrodes: int = 64,
        input_times: int = 641,
        num_classes: int = 5,
        fs: int = 256,
        k: int = 2,
        relu_is: int = 1,
        layers: Optional[Sequence[int]] = None,
        dropout_rate: float = 0.5,
    ) -> None:
        """Initializes the DGCNN model.

        Args:
            input_channels (int): Number of channels in the input EEG data. For
                common input with shape `[B, 1, E, T]`, this value is usually 1.
            input_electrodes (int): Number of EEG electrodes. For example, this
                value is 64 for 64-channel EEG data.
            input_times (int): EEG time-series length, namely the number of
                temporal samples in each example.
            num_classes (int): Number of classes. For example, this value is 5
                in a five-class classification task.
            fs (int): Sampling rate in Hz. This parameter is currently not used
                directly in the computation.
            k (int): Order of the Chebyshev graph convolution. A larger `k`
                aggregates information from a wider graph neighborhood.
            relu_is (int): Type of ReLU module with a learnable bias. 1 indicates
                B1ReLU, and 2 indicates B2ReLU.
            layers (Optional[Sequence[int]]): Output channels of each graph
                convolution layer in DGCNNBackbone. For example, `[64]` indicates
                a single graph convolution layer with 64 output channels.
            dropout_rate (float): Dropout probability used to reduce overfitting.

        Returns:
            None.
        """
        super().__init__()

        # Uses a single graph convolution layer with 64 output channels if no layer structure is specified
        if layers is None:
            layers = [64]

        # Stores input configuration parameters for later inspection or debugging
        self.input_channels = input_channels
        self.input_electrodes = input_electrodes
        self.input_times = input_times
        self.num_classes = num_classes
        self.fs = fs
        self.k = k
        self.relu_is = relu_is
        self.layers = list(layers)
        self.dropout_rate = dropout_rate

        # Differential entropy feature extraction module
        # Input:  [B, input_channels, input_electrodes, input_times]
        # Output: [B, input_channels, input_electrodes]
        # Purpose: Computes differential entropy features for each channel and electrode along the temporal dimension
        self.de = DifferentialEntropy(
            eps=1e-8,
            unbiased=False,
            keepdim=False,
        )

        # DGCNN graph convolution backbone network
        # Input:  [B, input_electrodes, input_channels]
        # Output: [B, num_classes]
        self.backbone = DGCNNBackbone(
            num_electrodes=input_electrodes,
            in_channels=input_channels,
            num_classes=num_classes,
            k=k,
            relu_is=relu_is,
            layers=self.layers,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the DGCNN forward pass.

        Args:
            x (torch.Tensor): Input EEG tensor with expected shape
                `[B, input_channels, input_electrodes, input_times]`.

                For example:
                    `[32, 1, 64, 641]`

                where:
                    `32` is the batch size,
                    `1` is the number of input channels,
                    `64` is the number of EEG electrodes, and
                    `641` is the number of temporal samples.

        Returns:
            torch.Tensor: Classification logits with shape `[B, num_classes]`.
            The output is not passed through Softmax and can be used directly
            with CrossEntropyLoss.
        """

        # Checks whether the input is a 4D tensor
        # Expected input shape is [B, input_channels, input_electrodes, input_times]
        if x.dim() != 4:
            raise ValueError(
                f"Input tensor must be 4-dimensional. Expected shape is "
                f"[B, input_channels, input_electrodes, input_times], "
                f"but got {tuple(x.shape)}."
            )

        # Optional shape consistency check
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Input channel count mismatch. The model expects input_channels="
                f"{self.input_channels}, but got {x.shape[1]}."
            )

        if x.shape[2] != self.input_electrodes:
            raise ValueError(
                f"Input electrode count mismatch. The model expects input_electrodes="
                f"{self.input_electrodes}, but got {x.shape[2]}."
            )

        # Computes differential entropy features along the temporal dimension
        # Input: [B, input_channels, input_electrodes, input_times]
        # Output: [B, input_channels, input_electrodes]
        x = self.de(x)

        # Rearranges tensor dimensions to match the input format required by DGCNNBackbone
        # Original shape: [B, input_channels, input_electrodes]
        # Rearranged shape: [B, input_electrodes, input_channels]
        x = rearrange(x, "b chs els -> b els chs")

        # Feeds the tensor into the DGCNN graph convolution backbone for classification
        # Input: [B, input_electrodes, input_channels]
        # Output: [B, num_classes]
        x = self.backbone(x)

        return x

#%% test code  
    
if __name__ == "__main__":

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = DGCNN(
        input_channels = 1,
        input_electrodes = 64,
        input_times= 641,
        num_classes = 5,
        fs = 256,
        k = 2,
        relu_is = 1,
        layers = [64],
        dropout_rate = 0.5).to(device)

    
    x = torch.randn(8, 1, 64, 641, device=device)
    logits = model(x)
    print('logits shape:', logits.shape)  # [8, 5]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
