# -*- coding: utf-8 -*-
"""
EEGConformer 模型实现（符合 Google Python Style Guide）。
[1]	Y. Song, Q. Zheng, B. Liu, and X. Gao, "EEG Conformer: Convolutional 
Transformer for EEG Decoding and Visualization," IEEE Transactions on Neural 
Systems and Rehabilitation Engineering, vol. 31, pp. 710-719, 2023,
doi: 10.1109/TNSRE.2022.3230250.


本模块定义了 EEGConformer 模型，包括 Patch 嵌入、Transformer 编码器、分类头等组件。
用于 EEG 信号分类任务。

创建日期: 2025 年 1 月 17 日
作者: Fujie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
import math
'''
PatchEmbedding: 提取输入的低维嵌入。
TransformerEncoderBlock: 实现单个 Transformer 编码器块。
TransformerEncoder: 堆叠多个编码器块以提取全局特征。
ClassificationHead: 将特征映射到分类结果。
Conformer: 整体模型，用于 EEG 数据分类任务。
'''
#%% 1. basic module

class ZScoreNormalization(nn.Module):
    """
    初始化 ZScoreNormalization 模块。

    ZScoreNormalization 是一种用于将输入张量按照 Z-score 标准化的方法，
    它将每个通道和电极的信号沿 B*T 维度进行标准化。
    """
    def __init__(self):

        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对输入张量 [B, C, E, T] 中每个通道(C)和电极(E)，沿着 B*T 维度进行 Z-score 标准化。

        Args:
            x (torch.Tensor): 输入张量，形状为 [B, C, E, T]，其中 B 是 batch_size，C 是通道数，E 是电极数，T 是时间长度。

        Returns:
            torch.Tensor: 标准化后的张量，形状与输入相同 [B, C, E, T]。
        """
        batch_size, num_channels, num_electrodes, time_steps = x.shape
        
        # 将数据调整为 [B, C, E, T] -> [C, E, B * T]，将 B 和 T 合并，便于计算
        x_reshaped = x.view(num_channels, num_electrodes, batch_size * time_steps)  # [C, E, B*T]
        
        # 计算 B*T 维度上的均值和标准差
        mean = x_reshaped.mean(dim=2, keepdim=True)  # [C, E, 1]
        std = x_reshaped.std(dim=2, keepdim=True)    # [C, E, 1]
        
        # 防止除零错误，加入一个小常数 epsilon
        epsilon = 1e-10
        x_normalized = (x_reshaped - mean) / (std + epsilon)
        
        # 恢复回原始形状 [B, C, E, T]
        x_normalized = x_normalized.view(batch_size, num_channels, num_electrodes, time_steps)

        return x_normalized



class PatchEmbedding(nn.Module):
    """输入 EEG 信号的 Patch 处理和特征嵌入。

    该模块用于提取 EEG 信号的时间和空间特征，并将其转换为 Transformer 可接受的格式。

    Attributes:
        shallownet (nn.Sequential): 用于提取 EEG 数据局部和全局特征的浅层 CNN。
        projection (nn.Sequential): 通过 1x1 卷积映射通道维度到目标嵌入大小，并调整张量形状。

    Args:
        k (int, optional): 通过频域、空域滤波后的特征维度大小，默认为 40。
        emb_dim (int, optional): 嵌入维度大小，默认为 =k=40。
    """

    def __init__(self,
                 temp_conv_ks=25, 
                 spatial_conv_ks=22,
                 temp_pool_ks=75,
                 dropout=0.5,
                 k=40,
                 emb_dim=40):

        super().__init__()
        self.temporal_conv_ks = temp_conv_ks  # 时间卷积核大小
        self.spatial_conv_ks = spatial_conv_ks  # 空间卷积核大小
        self.temp_pool_ks = temp_pool_ks  # 时间池化核大小
        self.temp_pool_s = self.temp_pool_ks // 5  # 时间池化步幅
        
        # 定义浅层网络，用于提取特征
        self.shallownet = nn.Sequential(
            nn.Conv2d(in_channels=1,  # 输入通道数
                      out_channels=k,  # 输出通道数
                      kernel_size=(1, self.temporal_conv_ks),  # 时间卷积核大小
                      stride=1),  # 步幅为 1
            nn.Conv2d(in_channels=k,  # 输入通道数为 k
                      out_channels=k,  # 输出通道数为 k
                      kernel_size=(self.spatial_conv_ks , 1),  # 空间卷积核大小
                      stride=1,  # 步幅为 1
                      groups=k),  # 使用深度可分离卷积
            nn.BatchNorm2d(k),  # 批归一化
            nn.ELU(),  # 激活函数：ELU
            nn.AvgPool2d(kernel_size=(1, self.temp_pool_ks),  # 平均池化
                         stride=(1, self.temp_pool_s)),  # 池化步幅
            nn.Dropout(dropout),  # Dropout 防止过拟合
        )
        
        # 定义投影层，用于调整输出特征维度
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels=k, out_channels=emb_dim, kernel_size=1),  # 通过 1x1 卷积映射到目标嵌入大小
            Rearrange('b e h w -> b (h w) e')  # 重新排列输出形状：[batch_size, sequence_length, embedding_dim]
        )

    def forward(self, x):
        """前向传播函数。

        Args:
            x (torch.Tensor): 输入 EEG 张量，形状为 [batch_size, 1, eeg_channels, time_steps]。

        Returns:
            torch.Tensor: 处理后的嵌入，形状为 [batch_size, sequence_length, embedding_dim]。
        """
        x = self.shallownet(x)  # 通过浅层网络提取特征
        x = self.projection(x)  # 通过投影层将特征映射到嵌入空间
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_dim, expansion, dropout):
        """
        初始化前馈网络模块。

        Args:
            emb_dim (int): 嵌入维度。
            expansion (int): 扩展因子，控制前馈网络的宽度。
            dropout (float): Dropout 率。
        """
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, expansion * emb_dim),  # 扩展线性层
            nn.GELU(),  # GELU 激活函数
            nn.Dropout(dropout),  # Dropout 防止过拟合
            nn.Linear(expansion * emb_dim, emb_dim),  # 恢复原始维度
        )

    def forward(self, x: Tensor) -> Tensor:
        """前向传播函数。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 输出张量。
        """
        return self.ffn(x)


class ResidualAdd(nn.Module):
    """
    初始化 ResidualAdd 模块。

    Args:
        fn (nn.Module): 需要执行的操作，例如多头自注意力或前馈网络。
    """
    def __init__(self, fn):

        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        """前向传播函数。

        Args:
            x (Tensor): 输入张量。
            **kwargs: 额外参数，传递给 fn。

        Returns:
            Tensor: 经过残差连接的输出张量。
        """
        res = x  # 保留输入作为残差
        x = self.fn(x, **kwargs)  # 执行操作
        return x + res  # 返回加法后的结果（残差连接）


class MultiheadAttention(nn.MultiheadAttention):
    """
    初始化多头自注意力模块。

    Args:
        embed_dim (int): 嵌入维度。
        num_heads (int): 注意力头数量。
        dropout (float): Dropout 率。
    """
    def __init__(self, embed_dim, num_heads, dropout):

        super().__init__(embed_dim=embed_dim,
                         num_heads=num_heads,
                         dropout=dropout,
                         batch_first=True)
        
    def forward(self, x):
        """前向传播函数。

        只返回加权后的输出张量，不返回注意力权重。

        Args:
            x (Tensor): 输入张量，形状为 [batch_size, seq_len, embed_dim]。

        Returns:
            Tensor: 经过注意力机制加权后的输出张量。
        """
        output, _ = super().forward(query=x, key=x, value=x, need_weights=False)
        return output  # 只返回输出张量，不返回注意力权重



class TransformerEncoderBlock(nn.Module):
    """单个 Transformer 编码器块。

    包含一个多头自注意力层和一个前馈网络。
    """
    """
    初始化 Transformer 编码器块。

    Args:
        emb_dim (int): 嵌入维度。
        num_heads (int): 注意力头数量。
        ffn_expansion (int): 前馈网络扩展因子。
        dropout_mha (float): 多头注意力中的 Dropout 率。
        dropout_ffn (float): 前馈网络中的 Dropout 率。
        dropout_encoder (float): 编码器中的 Dropout 率。
    """

    def __init__(self, emb_dim, num_heads=8, ffn_expansion=4, dropout_mha=0.5, dropout_ffn=0.5, dropout_encoder=0.5):

        super().__init__()
        self.attn = ResidualAdd(nn.Sequential( 
            nn.LayerNorm(emb_dim),
            MultiheadAttention(embed_dim=emb_dim,
                               num_heads=num_heads,
                               dropout=dropout_encoder),
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
        """前向传播函数。

        Args:
            x (Tensor): 输入张量。

        Returns:
            Tensor: 经过 Transformer 编码器块处理后的输出张量。
        """
        x = self.attn(x)  # 通过多头自注意力
        x = self.ffn(x)  # 通过前馈网络
        return x



class TransformerEncoder(nn.Module):
    """Transformer 编码器，由多个 Transformer 编码器块堆叠组成。"""
    """
    初始化 TransformerEncoder。

    参数:
        depth (int): Transformer 编码器块的数量。
        emb_dim (int): 嵌入维度大小。
        num_heads (int, optional): 注意力头的数量，默认为 8。
        dropout_mha (float, optional): 多头自注意力的 Dropout 率，默认为 0.5。
        dropout_ffn (float, optional): 前馈网络的 Dropout 率，默认为 0.5。
        dropout_encoder (float, optional): 编码器中的 Dropout 率，默认为 0.5。
    """

    def __init__(self, 
                 depth, 
                 emb_dim, 
                 num_heads=8,
                 ffn_expansion=4,                    
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5):

        super().__init__()
        # 使用 nn.Sequential 来按顺序堆叠多个 Transformer 编码器块
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
        """前向传播函数。"""
        return self.layers(x)  # 通过堆叠的 Transformer 编码器块处理输入


class EEGConformerBackbone(nn.Sequential):
    """
    初始化 EEGConformerBackbone。

    参数:
        temp_conv_ks (int): 时间卷积核大小。
        spatial_conv_ks (int): 空间卷积核大小。
        temp_pool_ks (int): 时间池化核大小。
        k (int): 卷积后特征图的通道数。
        emb_dim (int): 嵌入维度。
        encoder_depth (int): Transformer 编码器深度（块的数量）。
        num_heads (int): 多头自注意力的头数。
        ffn_expansion (int): 前馈网络扩展因子。
        dropout_prepatch (float): Patch 处理前的 Dropout 率。
        dropout_mha (float): 多头自注意力的 Dropout 率。
        dropout_ffn (float): 前馈网络的 Dropout 率。
        dropout_encoder (float): 编码器中的 Dropout 率。
    """
    def __init__(self, 
                 temp_conv_ks=25, 
                 spatial_conv_ks=22,
                 temp_pool_ks=75,
                 k=40,
                 emb_dim=40,
                 encoder_depth=6, 
                 num_heads=8,
                 ffn_expansion=4,  
                 dropout_prepatch=0.5,                  
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5):

        super().__init__()
        self.conformer = nn.Sequential(
            PatchEmbedding(temp_conv_ks=temp_conv_ks, 
                           spatial_conv_ks=spatial_conv_ks,
                           temp_pool_ks=temp_pool_ks,
                           dropout=dropout_prepatch,
                           k=k,
                           emb_dim=k),  # 使用 PatchEmbedding 进行特征提取
            TransformerEncoder(depth=encoder_depth,
                               emb_dim=emb_dim,
                               dropout_mha=dropout_mha,
                               dropout_ffn=dropout_ffn,
                               dropout_encoder=dropout_encoder,),  # 使用 Transformer 编码器进行序列建模
            nn.Flatten(),  # 展平输入以准备分类
        )

    def forward(self, x):
        """前向传播函数。"""
        return self.conformer(x)  # 通过主干网络处理输入


class ClassificationHead(nn.Sequential):
    """
    初始化分类头模块。
    
    参数:
        in_channels (int): 线性层输入维度。
        out_channels (int): 线性层输出维度，即分类数。
    """
    def __init__(self, 
                 in_channels: int,  # 线性层输入维度  
                 out_channels: int,):  # 线性层输出维度

        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 256),  # 输入特征维度映射到 256
            nn.ELU(),  # 激活函数
            nn.Dropout(0.5),  # Dropout 防止过拟合
            nn.Linear(256, 32),  # 将 256 映射到 32
            nn.ELU(),  # 激活函数
            nn.Dropout(0.3),  # Dropout 防止过拟合
            nn.Linear(32, out_channels)  # 最终分类层，输出类别数
        )

    def forward(self, x):
        """前向传播函数。"""
        return self.fc(x)  # 返回分类结果


class EEGConformer(nn.Sequential):
    """
    初始化 EEGConformer 模型。
    
    参数:
        input_channels (int): 输入张量的通道数，默认为 1。
        input_electrodes (int): EEG 电极数量，默认为 64。
        input_times (int): 输入时间长度，默认为 512。
        temp_conv_ks (int): 时间卷积核大小，默认为 25。
        temp_pool_ks (int): 时间池化核大小，默认为 75。
        k (int): 特征图的通道数，默认为 40。
        emb_dim (int): 嵌入维度，默认为 40。
        encoder_depth (int): Transformer 编码器的深度（块的数量），默认为 6。
        num_heads (int): 多头注意力的头数，默认为 8。
        ffn_expansion (int): 前馈网络扩展因子，默认为 4。
        dropout_prepatch (float): Patch 处理前的 Dropout 率，默认为 0.5。
        dropout_mha (float): 多头自注意力的 Dropout 率，默认为 0.5。
        dropout_ffn (float): 前馈网络的 Dropout 率，默认为 0.5。
        dropout_encoder (float): 编码器中的 Dropout 率，默认为 0.5。
        fc_in_channels (int): 分类器的输入通道数，默认为 1024。
        num_classes (int): 类别数量，默认为 5。
        is_zscore (bool): 是否使用 Z-score 标准化，默认为 True。
    """
    def __init__(self, 
                 input_channels = 1,  # 输入张量的通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 temp_conv_ks=25, 
                 temp_pool_ks=75,
                 k=40,
                 emb_dim=40,
                 encoder_depth=6, 
                 num_heads=8,
                 ffn_expansion=4,     
                 dropout_prepatch=0.5,
                 dropout_mha=0.5,
                 dropout_ffn=0.5,              
                 dropout_encoder=0.5,
                 fc_in_channels=1024,
                 num_classes=5,
                 is_zscore=True,
                 ):
        super().__init__()
        self.zscore = ZScoreNormalization() if is_zscore else nn.Identity()  # 选择是否使用 Z-score 标准化
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
        self.classifier = ClassificationHead(in_channels=fc_in_channels, out_channels=num_classes)  # 分类头

    def forward(self, x):
        """前向传播函数。

        将输入张量依次通过 `self.backbone` 与 `self.classifier`，得到最终分类输出。

        参数:
            x (torch.Tensor): 输入张量，形状通常为 [N, input_channels, input_electrodes, input_times]；
                常见设置为 [N, 1, C, T]，其中 N 为批大小，1 为“伪图像”的通道维，C 为电极数，T 为时间长度。

        返回:
            torch.Tensor: 分类输出张量，形状为 [N, num_classes]（未做 Softmax 的 logits）。
        """
        x = self.zscore(x)  # 执行 Z-score 标准化（如果启用了）
        x = self.backbone(x)  # 通过主干网络提取并展平特征
        x = self.classifier(x)  # 送入线性分类器得到 logits
        return x  # 返回分类结果

        



#%%%
# 模型测试代码
if __name__ == "__main__":
    # 初始化模型
    model = EEGConformer(
                        input_channels = 1,  # 输入张量的通道数
                        input_electrodes = 64,  # EEG 电极数量
                        input_times = 512,  # 时间长度
                        temp_conv_ks=25, 
                        temp_pool_ks=75,
                        k=40,
                        emb_dim=40,
                        encoder_depth=6, 
                        num_heads=8,
                        ffn_expansion=4,     
                        dropout_prepatch=0.5,
                        dropout_mha=0.5,
                        dropout_ffn=0.5,              
                        dropout_encoder=0.5,
                        fc_in_channels=1120,
                        num_classes=5,
                        is_zscore=True,)
        
    
    
    # 创建一个 dummy 输入数据：16个样本，1个通道，22个EEG通道，1000个时间步
    dummy_input = torch.randn(16, 1,64, 512)  # [batch_size, 1, C, T]
    
    # 进行一次前向传播
    output = model(dummy_input)
    
    # 输出结果的形状：[batch_size, n_classes]
    print("Output shape:", output.shape)  # 预期输出形状：[16, 4]

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))  # 使用 Adam 优化器，学习率为 1e-3

