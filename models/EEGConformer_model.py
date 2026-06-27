# -*- coding: utf-8 -*-
"""Defines the EEGConformer model for EEG signal classification.

Modified from:
https://github.com/eeyhsong/EEG-Conformer

Reference:
    [1] Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional
    Transformer for EEG Decoding and Visualization," IEEE Transactions on Neural
    Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, 2023,
    doi: 10.1109/TNSRE.2022.3230250.

This module defines the EEGConformer model, including patch embedding,
Transformer encoder, classification head, and related components for EEG signal
classification tasks.

Last modification: 2026-06-21
Author:
    Fujie.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
import math
'''
PatchEmbedding: Extracts low-dimensional embeddings from the input EEG signal.
TransformerEncoderBlock: Implements a single Transformer encoder block.
TransformerEncoder: Stacks multiple encoder blocks to extract global features.
ClassificationHead: Maps extracted features to classification logits.
Conformer: Defines the overall model for EEG data classification tasks.
'''
#%% 1. Basic module.

class PatchEmbedding(nn.Module):
    """Converts input EEG signals into patch embeddings.

    This module extracts temporal and spatial features from EEG signals and
    converts them into the sequence format expected by the Transformer encoder.

    Attributes:
        temporal_conv_ks (int): Temporal convolution kernel size.
        spatial_conv_ks (int): Spatial convolution kernel size.
        temp_pool_ks (int): Temporal average-pooling kernel size.
        temp_pool_s (int): Temporal average-pooling stride.
        shallownet (nn.Sequential): Shallow CNN used to extract local and
            global EEG features.
        projection (nn.Sequential): Projection layer that maps channels to the
            target embedding dimension and reshapes the output tensor.
    """

    def __init__(self,
                 temp_conv_ks=25, 
                 spatial_conv_ks=22,
                 temp_pool_ks=75,
                 dropout=0.5,
                 k=40,
                 emb_dim=40):
        """Initializes the patch embedding module.

        Args:
            temp_conv_ks (int, optional): Temporal convolution kernel size.
                Defaults to 25.
            spatial_conv_ks (int, optional): Spatial convolution kernel size.
                Defaults to 22.
            temp_pool_ks (int, optional): Temporal average-pooling kernel size.
                Defaults to 75.
            dropout (float, optional): Dropout probability applied after
                average pooling. Defaults to 0.5.
            k (int, optional): Number of convolutional feature channels after
                temporal and spatial filtering. Defaults to 40.
            emb_dim (int, optional): Target embedding dimension. Defaults to
                40.
        """

        super().__init__()
        self.temporal_conv_ks = temp_conv_ks  # Temporal convolution kernel size.
        self.spatial_conv_ks = spatial_conv_ks  # Spatial convolution kernel size.
        self.temp_pool_ks = temp_pool_ks  # Temporal pooling kernel size.
        self.temp_pool_s = self.temp_pool_ks // 5  # Temporal pooling stride.
        
        # Defines the shallow network used for feature extraction.
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=1,  # Number of input channels.
                      out_channels=k,  # Number of output channels.
                      kernel_size=(1, self.temporal_conv_ks),  # Temporal convolution kernel size.
                      stride=1),  # Uses a stride of 1.
            nn.Conv2d(in_channels=k,  # Number of input channels.
                      out_channels=k,  # Number of output channels.
                      kernel_size=(self.spatial_conv_ks , 1),  # Spatial convolution kernel size.
                      stride=1,  # Uses a stride of 1.
                      ),  
            nn.BatchNorm2d(k),  # Batch normalization.
            nn.ELU(),  # ELU activation function.
            nn.AvgPool2d(kernel_size=(1, self.temp_pool_ks),  # Average pooling.
                         stride=(1, self.temp_pool_s)),  # Pooling stride.
            nn.Dropout(dropout),  # Dropout to reduce overfitting.
        )
        
        # Defines the projection layer used to adjust the output feature dimension.
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=k, out_channels=emb_dim, kernel_size=1),  # Maps channels to the target embedding size with a 1x1 convolution.
            Rearrange('b e h w -> b (h w) e')  # Rearranges the output shape to [batch_size, sequence_length, embedding_dim].
        )

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x (torch.Tensor): Input EEG tensor with shape
                [batch_size, 1, eeg_channels, time_steps].

        Returns:
            torch.Tensor: Embedded output tensor with shape
            [batch_size, sequence_length, embedding_dim].
        """
        x = self.shallownet(x)  # Extracts features with the shallow network.
        x = self.projection(x)  # Projects features into the embedding space.
        return x


class FeedForwardBlock(nn.Module):
    """Feed-forward network block used inside a Transformer encoder block."""

    def __init__(self, emb_dim, expansion, dropout):
        """Initializes the feed-forward network block.

        Args:
            emb_dim (int): Embedding dimension.
            expansion (int): Expansion factor that controls the hidden width of
                the feed-forward network.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),  # Expands the feature dimension.
            nn.GELU(),  # GELU activation function.
            nn.Dropout(dropout),  # Dropout to reduce overfitting.
            nn.Linear(expansion * emb_dim, emb_dim),  # Restores the original feature dimension.
        )

    def forward(self, x: Tensor) -> Tensor:
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after the feed-forward network.
        """
        return self.ffn(x)


class ResidualAdd(nn.Module):
    """Applies a module and adds its output to the original input.

    Attributes:
        fn (nn.Module): Module applied before adding the residual connection.
    """

    def __init__(self, fn):
        """Initializes the residual wrapper.

        Args:
            fn (nn.Module): Operation to execute, such as multi-head attention
                or a feed-forward network.
        """

        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor.
            **kwargs: Additional keyword arguments passed to ``fn``.

        Returns:
            Tensor: Output tensor after applying the residual connection.
        """
        res = x  # Stores the input tensor as the residual branch.
        x = self.fn(x, **kwargs)  # Applies the wrapped operation.
        return x + res  # Adds the residual connection and returns the result.


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer.

    Attributes:
        emb_size (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        keys (nn.Linear): Linear projection for keys.
        queries (nn.Linear): Linear projection for queries.
        values (nn.Linear): Linear projection for values.
        att_drop (nn.Dropout): Dropout applied to attention weights.
        projection (nn.Linear): Output projection layer.
    """

    def __init__(self, emb_size, num_heads, dropout):
        """Initializes the multi-head self-attention layer.

        Args:
            emb_size (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout probability applied to attention weights.
        """
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor with shape
                [batch_size, sequence_length, embedding_dim].
            mask (Tensor, optional): Boolean attention mask. Defaults to None.

        Returns:
            Tensor: Output tensor with shape
            [batch_size, sequence_length, embedding_dim].
        """
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.masked_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class TransformerEncoderBlock(nn.Module):
    """Single Transformer encoder block.

    The block contains a multi-head self-attention layer and a feed-forward
    network, each wrapped with layer normalization, dropout, and a residual
    connection.

    Attributes:
        attn (ResidualAdd): Residual multi-head self-attention sublayer.
        ffn (ResidualAdd): Residual feed-forward sublayer.
    """

    def __init__(self, emb_dim, num_heads=10, ffn_expansion=4, dropout_mha=0.5, dropout_ffn=0.5, dropout_encoder=0.5):
        """Initializes the Transformer encoder block.

        Args:
            emb_dim (int): Embedding dimension.
            num_heads (int, optional): Number of attention heads. Defaults to
                10.
            ffn_expansion (int, optional): Feed-forward network expansion
                factor. Defaults to 4.
            dropout_mha (float, optional): Dropout probability in multi-head
                attention. Defaults to 0.5.
            dropout_ffn (float, optional): Dropout probability in the
                feed-forward network. Defaults to 0.5.
            dropout_encoder (float, optional): Dropout probability in the
                encoder block. Defaults to 0.5.
        """

        super().__init__()
        self.attn = ResidualAdd(nn.Sequential( 
            nn.LayerNorm(emb_dim),
            MultiHeadAttention(emb_size=emb_dim,
                               num_heads=num_heads,
                               dropout=dropout_mha),
            nn.Dropout(dropout_encoder)
        ))
        
        self.ffn = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_dim),
            FeedForwardBlock(emb_dim=emb_dim, 
                             expansion=ffn_expansion,
                             dropout=dropout_ffn),
            nn.Dropout(dropout_encoder)
        ))

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after the Transformer encoder block.
        """
        x = self.attn(x)  # Applies multi-head self-attention.
        x = self.ffn(x)  # Applies the feed-forward network.
        return x


class TransformerEncoder(nn.Module):
    """Transformer encoder composed of stacked encoder blocks.

    Attributes:
        layers (nn.Sequential): Sequential stack of Transformer encoder blocks.
    """

    def __init__(self, 
                 depth, 
                 emb_dim, 
                 num_heads=8,
                 ffn_expansion=4,                    
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5):
        """Initializes the Transformer encoder.

        Args:
            depth (int): Number of Transformer encoder blocks.
            emb_dim (int): Embedding dimension.
            num_heads (int, optional): Number of attention heads. Defaults to
                8.
            ffn_expansion (int, optional): Feed-forward network expansion
                factor. Defaults to 4.
            dropout_mha (float, optional): Dropout probability in multi-head
                self-attention. Defaults to 0.5.
            dropout_ffn (float, optional): Dropout probability in the
                feed-forward network. Defaults to 0.5.
            dropout_encoder (float, optional): Dropout probability in the
                encoder blocks. Defaults to 0.5.
        """

        super().__init__()
        # Stacks multiple Transformer encoder blocks sequentially with nn.Sequential.
        self.layers = nn.Sequential(
            *[TransformerEncoderBlock(emb_dim=emb_dim,
                                      num_heads=num_heads,
                                      ffn_expansion=ffn_expansion,
                                      dropout_mha=dropout_mha,
                                      dropout_ffn=dropout_ffn,
                                      dropout_encoder=dropout_encoder,
                                      ) for _ in range(depth)]
        )

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after the stacked Transformer encoder blocks.
        """
        return self.layers(x)  # Processes the input through the stacked Transformer encoder blocks.


class EEGConformerBackbone(nn.Sequential):
    """Backbone network of EEGConformer.

    The backbone applies patch embedding, Transformer encoding, and flattening
    to produce features for the classification head.

    Attributes:
        conformer (nn.Sequential): Sequential backbone containing patch
            embedding, Transformer encoder, and flattening layers.
    """

    def __init__(self, 
                 temp_conv_ks=25, 
                 spatial_conv_ks=22,
                 temp_pool_ks=75,
                 k=40,
                 emb_dim=40,
                 encoder_depth=6, 
                 num_heads=10,
                 ffn_expansion=4,  
                 dropout_prepatch=0.5,                  
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5):
        """Initializes the EEGConformer backbone.

        Args:
            temp_conv_ks (int, optional): Temporal convolution kernel size.
                Defaults to 25.
            spatial_conv_ks (int, optional): Spatial convolution kernel size.
                Defaults to 22.
            temp_pool_ks (int, optional): Temporal average-pooling kernel size.
                Defaults to 75.
            k (int, optional): Number of feature-map channels after
                convolution. Defaults to 40.
            emb_dim (int, optional): Embedding dimension. Defaults to 40.
            encoder_depth (int, optional): Number of Transformer encoder
                blocks. Defaults to 6.
            num_heads (int, optional): Number of multi-head self-attention
                heads. Defaults to 10.
            ffn_expansion (int, optional): Feed-forward network expansion
                factor. Defaults to 4.
            dropout_prepatch (float, optional): Dropout probability before the
                patch representation is produced. Defaults to 0.5.
            dropout_mha (float, optional): Dropout probability in multi-head
                self-attention. Defaults to 0.5.
            dropout_ffn (float, optional): Dropout probability in the
                feed-forward network. Defaults to 0.5.
            dropout_encoder (float, optional): Dropout probability in the
                encoder blocks. Defaults to 0.5.
        """

        super().__init__()
        self.conformer = nn.Sequential(
            PatchEmbedding(temp_conv_ks=temp_conv_ks, 
                           spatial_conv_ks=spatial_conv_ks,
                           temp_pool_ks=temp_pool_ks,
                           dropout=dropout_prepatch,
                           k=k,
                           emb_dim=emb_dim),  # Uses PatchEmbedding for feature extraction.
            TransformerEncoder(depth=encoder_depth,
                               emb_dim=emb_dim,
                               num_heads=num_heads,
                               ffn_expansion=ffn_expansion,
                               dropout_mha=dropout_mha,
                               dropout_ffn=dropout_ffn,
                               dropout_encoder=dropout_encoder,),  # Uses the Transformer encoder for sequence modeling.
            nn.Flatten(),  # Flattens the input to prepare it for classification.
        )

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Flattened backbone features.
        """
        return self.conformer(x)  # Processes the input through the backbone network.


class ClassificationHead(nn.Sequential):
    """Classification head that maps features to class logits.

    Attributes:
        fc (nn.Sequential): Fully connected classification network.
    """

    def __init__(self, 
                 in_channels: int,  # Input dimension of the linear layer.
                 out_channels: int,):  # Output dimension of the linear layer.
        """Initializes the classification head.

        Args:
            in_channels (int): Input dimension of the first linear layer.
            out_channels (int): Output dimension of the final linear layer,
                equal to the number of classes.
        """

        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),  # Maps input features to 256 dimensions.
            nn.ELU(),  # Activation function.
            nn.Dropout(0.5),  # Dropout to reduce overfitting.
            nn.Linear(256, 32),  # Maps 256 dimensions to 32 dimensions.
            nn.ELU(),  # Activation function.
            nn.Dropout(0.3),  # Dropout to reduce overfitting.
            nn.Linear(32, out_channels)  # Final classification layer that outputs class logits.
        )

    def forward(self, x):
        """Runs the forward pass.

        Args:
            x (Tensor): Input feature tensor.

        Returns:
            Tensor: Classification logits.
        """
        return self.fc(x)  # Returns the classification logits.

#%%
class EEGConformer(nn.Sequential):
    """EEGConformer model for EEG signal classification.

    Attributes:
        backbone (EEGConformerBackbone): Feature extraction backbone.
        classifier (ClassificationHead): Classification head that maps features
            to class logits.
    """

    def __init__(self, 
                 input_channels = 1,  # Number of input tensor channels.
                 input_electrodes = 64,  # Number of EEG electrodes.
                 input_times = 641,  # Number of time points.
                 temp_conv_ks=25, 
                 temp_pool_ks=75,
                 k=40,
                 emb_dim=40,
                 encoder_depth=6, 
                 num_heads=10,
                 ffn_expansion=4,     
                 dropout_prepatch=0.5,
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5,
                 fc_in_channels=1480,
                 num_classes=5,
                 ):
        """Initializes the EEGConformer model.

        Args:
            input_channels (int, optional): Number of input tensor channels.
                Defaults to 1.
            input_electrodes (int, optional): Number of EEG electrodes.
                Defaults to 64.
            input_times (int, optional): Input temporal length. Defaults to
                641.
            temp_conv_ks (int, optional): Temporal convolution kernel size.
                Defaults to 25.
            temp_pool_ks (int, optional): Temporal average-pooling kernel size.
                Defaults to 75.
            k (int, optional): Number of feature-map channels. Defaults to 40.
            emb_dim (int, optional): Embedding dimension. Defaults to 40.
            encoder_depth (int, optional): Transformer encoder depth, measured
                by the number of blocks. Defaults to 6.
            num_heads (int, optional): Number of multi-head attention heads.
                Defaults to 10.
            ffn_expansion (int, optional): Feed-forward network expansion
                factor. Defaults to 4.
            dropout_prepatch (float, optional): Dropout probability before the
                patch representation is produced. Defaults to 0.5.
            dropout_mha (float, optional): Dropout probability in multi-head
                self-attention. Defaults to 0.5.
            dropout_ffn (float, optional): Dropout probability in the
                feed-forward network. Defaults to 0.5.
            dropout_encoder (float, optional): Dropout probability in the
                encoder blocks. Defaults to 0.5.
            fc_in_channels (int, optional): Input dimension of the classifier.
                Defaults to 1480.
            num_classes (int, optional): Number of output classes. Defaults to
                5.
        """
        super().__init__()

        self.backbone = EEGConformerBackbone(
            temp_conv_ks=temp_conv_ks,
            spatial_conv_ks=input_electrodes,
            temp_pool_ks=temp_pool_ks,
            dropout_prepatch=dropout_prepatch,
            k=k,
            emb_dim=emb_dim,
            encoder_depth=encoder_depth,
            num_heads=num_heads,
            ffn_expansion=ffn_expansion,
            dropout_mha=dropout_mha,
            dropout_ffn=dropout_ffn,
            dropout_encoder=dropout_encoder,
        )
        self.classifier = ClassificationHead(in_channels=fc_in_channels, out_channels=num_classes)  # Classification head.

    def forward(self, x):
        """Runs the forward pass.

        The input tensor is passed through ``self.backbone`` and
        ``self.classifier`` to produce the final classification output.

        Args:
            x (torch.Tensor): Input tensor, usually with shape
                [N, input_channels, input_electrodes, input_times]. A common
                setting is [N, 1, C, T], where N is the batch size, 1 is the
                pseudo-image channel dimension, C is the number of electrodes,
                and T is the temporal length.

        Returns:
            torch.Tensor: Classification output tensor with shape
            [N, num_classes]. The returned values are logits without Softmax.
        """
        x = self.backbone(x)  # Extracts and flattens features with the backbone network.
        x = self.classifier(x)  # Sends features to the linear classifier to obtain logits.
        return x  # Returns the classification logits.


#%%%
# Model test code.
if __name__ == "__main__":
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initializes the model.
    model = EEGConformer(
                        input_channels = 1,  # Number of input tensor channels.
                        input_electrodes = 64,  # Number of EEG electrodes.
                        input_times = 641,  # Number of time points.
                        temp_conv_ks=25, 
                        temp_pool_ks=75,
                        k=40,
                        emb_dim=40,
                        encoder_depth=6, 
                        num_heads=10,
                        ffn_expansion=4,     
                        dropout_prepatch=0.5,
                        dropout_mha=0.5,
                        dropout_ffn=0.5,              
                        dropout_encoder=0.5,
                        fc_in_channels=1480,
                        num_classes=5,)
        
    
    # Creates dummy input data with 16 samples, one channel, 64 EEG electrodes, and 641 time steps.
    dummy_input = torch.randn(16, 1,64, 641)  # [batch_size, 1, C, T]
    
    # Runs one forward pass.
    output = model(dummy_input)
    
    # Outputs the result shape: [batch_size, n_classes].
    print("Output shape:", output.shape)  # Expected output shape: [16, 5]

    # Loss function and optimizer.
    criterion = nn.CrossEntropyLoss()  # Uses cross-entropy loss for the classification task.
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))  # Uses the Adam optimizer with a learning rate of 2e-4.
    
    
