# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:51:33 2025

[1]	F. Li et al., "Decoding imagined speech from EEG signals using hybrid-scale 
spatial-temporal dilated convolution network," Journal of Neural Engineering, 
vol. 18, no. 4, p. 0460c4, 2021/08/11 2021, doi: 10.1088/1741-2552/ac13c0.


@author: Fujie
"""

import numpy as np                                   # 导入 numpy 库，用于生成随机信号矩阵
import torch                                        # 导入 PyTorch 顶层库
import torch.nn as nn                               # 导入神经网络模块
import torch.nn.functional as F                     # 导入功能函数模块
from einops import rearrange, reduce
from einops.layers.torch import Rearrange, Reduce
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional
import torch.optim as optim                         # 导入优化器模块
#%%

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


# 1. 混合尺度时域卷积模块
class HybridTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_list=[7, 19, 31, 63]):
        """
        in_channels: 输入通道数（这里为1，因为我们将EEG样本reshape成 [B,1,C,T]）
        out_channels: 输出通道数，即F1
        kernel_sizes: 不同尺度的卷积核长度
        """
        super().__init__()
        # 对每个核大小建立一个卷积分支，采用padding保证时域维度不变
        self.conv_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1, k), padding='same'),
                nn.BatchNorm2d(out_channels)
            )
            for k in kernel_size_list
        ])

    def forward(self, x):
        # x: [B, 1, C, T]
        # 对每个分支做卷积后求和
        out = torch.sum(torch.stack([conv(x) for conv in self.conv_list]), dim=0)
        # out = sum(conv(x) for conv in self.conv_list)  # 形状：[B, F1, C, T]
        return out
    
# 2. 空间卷积模块（深度可分离卷积）
class DepthwiseSpatialConv(nn.Module):
    def __init__(self, in_channels, out_channels, input_electrodes):
        """
        in_channels: 上一层输出的通道数（F1）
        num_electrodes: 电极数量 C，用作卷积核高度
        """
        super(DepthwiseSpatialConv, self).__init__()
        # 使用groups=in_channels实现每个通道独立卷积，卷积核大小为 (C, 1)
        self.conv = nn.Conv2d(in_channels, 
                              out_channels,
                              kernel_size=(input_electrodes, 1),
                              groups=in_channels,
                              padding='valid',
                              bias=False)
    def forward(self, x):
        # 输入 x: [B, F1, C, T]
        out = self.conv(x)  # 输出形状：[B, F1, 1, T]

        return out


# 3. Dilated卷积块（用于1D序列）
class DilatedBlock(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size=3, 
                 dilation_list=[1, 2],
                 p_dropout=0.3):
        super().__init__()

        self.dilated_conv1 = nn.Conv1d(in_channels=in_channels, 
                                       out_channels=out_channels, 
                                       kernel_size = kernel_size,
                                       dilation = dilation_list[0],
                                       padding = 'same')
        
        self.dilated_conv2 = nn.Conv1d(in_channels=out_channels,
                                       out_channels=out_channels, 
                                       kernel_size = kernel_size, 
                                       dilation = dilation_list[1],
                                       padding ='same')
        
        self.down = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.bn1=nn.BatchNorm1d(num_features=out_channels)
        self.bn2=nn.BatchNorm1d(num_features=out_channels)
        
        self.act=nn.ELU()
        self.drop = nn.Dropout(p=p_dropout)


    def forward(self, x):
        res = self.down(x)      
        x=self.drop(self.act(self.bn1(self.dilated_conv1(x))))
        x=self.drop(self.act(self.bn2(self.dilated_conv2(x))))
        x= x+res
        return x
    
    
class HS_STDCNBackbone(nn.Module):
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
        super().__init__()
        
        # 混合尺度时域卷积：输入 shape [B, 1, C, T] -> 输出 [B, F1, C, T]
        self.hybrid_temporal = HybridTemporalConv(in_channels=input_channels,
                                                  out_channels=F1,
                                                  kernel_size_list=temporal_ks_list)
        
        # 空间卷积：对每个电极进行卷积，将电极维度降为1，输出 [B, F1, 1, T]
        self.depthwise_spatial = DepthwiseSpatialConv(in_channels=F1, 
                                                      out_channels=F1,
                                                      input_electrodes=input_electrodes)
        self.process= nn.Sequential(
            nn.BatchNorm2d(num_features=F1),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_ks)),
            nn.Dropout(dropout_spatial),
            )
        self.dilated_block1=DilatedBlock(in_channels=F1, 
                                         out_channels=F2, 
                                         kernel_size=dilation_ks, 
                                         dilation_list=dilation_list,
                                         p_dropout=dropout_dilated)
        
        self.dilated_block2=DilatedBlock(in_channels=F2, 
                                         out_channels=F2, 
                                         kernel_size=dilation_ks, 
                                         dilation_list=dilation_list,
                                         p_dropout=dropout_dilated)
        self.flatten=nn.Flatten()
    def forward(self, x):
        B, C, E, T = x.size()    
        
        x=self.hybrid_temporal(x)
        x=self.depthwise_spatial(x)
        x=self.process(x)
        
        x = rearrange(x, 'B C E T -> B (C E) T')

        x=self.dilated_block1(x)
        x=self.dilated_block2(x)
        return self.flatten(x)
        
    
class Dense(nn.Linear):  # 继承自 nn.Linear 的全连接层（可选 L2 max-norm）
    """
    线性层（继承 nn.Linear），带可选的权重 L2 最大范数约束。

    若输入张量维度大于 2，会自动展平成 [N, in_features] 再送入线性层；
    在前向前对权重按行（每个输出单元的权重向量）施加 L2 max-norm。

    Args:
        in_channels (int): 输入特征维度（in_features）。
        out_channels (int): 输出特征维度（out_features）。
        max_norm (Optional[float]): 若不为 None，则对权重做 L2 最大范数裁剪（沿 dim=1）。
        bias (bool): 是否使用偏置项。

    Attributes:
        max_norm (Optional[float]): L2 最大范数阈值（None 表示关闭）。
        weight (torch.nn.Parameter): 线性层权重，形状 [out_features, in_features]。
        bias (Optional[torch.nn.Parameter]): 偏置，形状 [out_features]（若启用）。
    """
    def __init__(self,
                 in_channels: int,  # 线性层输入维度
                 out_channels: int,  # 线性层输出维度
                 max_norm: Optional[float] = 0.25,  # L2 max-norm 阈值
                 bias: bool = True) -> None:  # 是否使用偏置
        super().__init__(in_features=in_channels,  # 父类构造：指定输入特征数
                         out_features=out_channels,  # 指定输出特征数
                         bias=bias)  # 是否使用偏置
        self.max_norm = float(max_norm) if max_norm is not None else None  # 记录阈值

    def _apply_max_norm(self) -> None:  # 内部工具：对权重做 L2 max-norm
        if self.max_norm is None:  # 未启用则直接返回
            return
        with torch.no_grad():  # 不记录梯度
            # weight: [out_features, in_features]，对每一行（一个输出神经元）做裁剪
            self.weight.data.copy_(  # 原位更新权重
                torch.renorm(self.weight.data, p=2, dim=1, maxnorm=self.max_norm)  # 沿输入维做 L2 裁剪
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # 自动展平到 [N, in_features]
        if x.dim() > 2:  # 若输入为多维，先展平除 batch 外的维度
            x = x.view(x.size(0), -1)

        # 可选的 L2 max-norm（先裁剪再前向，使输出与裁剪后权重一致）
        self._apply_max_norm()  # 若启用则裁剪权重

        # 用父类实现完成线性变换
        return super().forward(x)  # 线性变换返回


#%%
class HS_STDCN(nn.Module):
    def __init__(self,
                 input_channels: int, 
                 input_electrodes: int,
                 input_times: int,    
                 fc_in_channels=256,  # 分类器全连接层输入特征维度
                 num_classes=5,  # 类别数
                 temporal_ks_list=[7, 19, 31, 63],
                 avg_pool_ks=16,
                 F1=8, 
                 F2=16,
                 dilation_ks=3,
                 dilation_list=[1, 2],
                 dropout_spatial=0.2,
                 dropout_dilated=0.3,
                 is_zscore=True):
        
        super().__init__()  # Initialize base nn.Module  # 调用父类 nn.Module 的初始化
        self.zscore= ZScoreNormalization() if is_zscore else nn.Identity()
        
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
            dropout_dilated=dropout_dilated,)  # 传入 Dropout 概率
        
        self.classifier=Dense(in_channels=fc_in_channels,  # 线性分类头输入维度（需与 backbone 输出展平维度一致）
                              out_channels=num_classes,  # 输出类别数
                              max_norm=0.25)  # 可选的 L2 max-norm 约束阈值     
        
    def forward(self, x):  # Forward pass for EEGNet  # 定义前向传播：输入 -> 顺序模型
        """前向传播。

        将输入张量依次通过 `self.backbone` 与 `self.classifier`，得到最终分类输出。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 [N, input_channels, input_electrodes, input_times]；
                常见设置为 [N, 1, C, T]，其中 N 为批大小，1 为“伪图像”的通道维，C 为电极数，T 为时间长度。

        Returns:
            torch.Tensor: 分类输出张量，形状为 [N, num_classes]（未做 Softmax 的 logits）。
        """
        x=self.zscore(x)
        x=self.backbone(x)  # 通过主干网络提取并展平特征
        x=self.classifier(x)  # 送入线性分类器得到 logits
        return x  # 返回分类结果        
        
    #%%
# 测试模型
if __name__ == '__main__':
    # 假设每个EEG样本有64个电极，256个采样点，8个分类
    batch_size = 4
    num_electrodes = 64
    seq_len = 256
    num_classes = 8

    # 构造一个随机输入张量，形状为 [B, 1, C, T]
    x = torch.randn(batch_size, 1, num_electrodes, seq_len)

    # 实例化模型
    model = HS_STDCN(input_channels=1, 
                     input_electrodes=num_electrodes,
                     input_times=seq_len,    
                     num_classes=num_classes, 
                     fc_in_channels=256,
                     F1=8, 
                     F2=16)

    # 前向传播
    output = model(x)
    print("模型输出形状：", output.shape)  # 应该为 [batch_size, num_classes]
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
