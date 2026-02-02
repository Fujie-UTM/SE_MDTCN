# -*- coding: utf-8 -*-
"""
EEGTCNet 模型实现（符合 Google Python Style Guide）。
[1]	T. M. Ingolfsson, M. Hersche, X. Wang, N. Kobayashi, L. Cavigelli, and 
L. Benini, "EEG-TCNet: An Accurate Temporal Convolutional Network for Embedded 
Motor-Imagery Brain–Machine Interfaces," in 2020 IEEE International Conference 
on Systems, Man, and Cybernetics (SMC), 11-14 Oct. 2020 2020, pp. 2958-2965, 
doi: 10.1109/SMC42975.2020.9283028. 

本模块定义了 EEGTCNet 模型，包括二维卷积、深度可分离卷积、扩张卷积、TCN（时间卷积网络）、激活函数、池化层、
Dropout 以及全连接层等组件。

创建日期: 2025 年 10 月 23 日
作者: Fujie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional


#%% 1. 基础模块（Basic Blocks）
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



class DepthwiseConv2D(nn.Conv2d):  # 继承自 nn.Conv2d 的深度卷积层
    """
    深度卷积（Depthwise Conv2D）。

    本层固定 groups=in_channels，从而实现“每个输入通道独立卷积”。
    步幅固定为 1；padding 可为 'valid'(即 0)、'same'，或显式整数/二元组。
    可选对每个滤波器在空间维度 (kH*kW) 上施加 L2 最大范数约束（max-norm）。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数；需为 `in_channels` 的整数倍（depth multiplier）。
        kernel_size (Tuple[int, int]): 卷积核尺寸 (kH, kW)。
        max_norm (Optional[float]): 若不为 None，则在前向前对每个滤波器做 L2 max-norm 约束。
        padding (Union[str, int, Tuple[int, int]]): 填充方式，支持 'valid'、'same'、或显式 padding。
        bias (bool): 是否使用偏置项。

    Attributes:
        max_norm (Optional[float]): L2 最大范数阈值（None 表示不启用）。
        weight (torch.nn.Parameter): 从父类继承的卷积核权重，形状为 [C_out, 1, kH, kW]（深度卷积情形）。
        bias (Optional[torch.nn.Parameter]): 偏置，若 `bias=True` 则存在。
    """

    def __init__(self,
                 in_channels: int,  # 输入通道数
                 out_channels: int,  # 输出通道数（需是 in_channels 的整数倍）
                 kernel_size: Tuple[int, int],  # 卷积核尺寸
                 max_norm: Optional[float] = None,  # L2 max-norm 阈值（可选）
                 padding: Union[str, int, Tuple[int, int]] = 'valid',  # 填充方式
                 bias: bool = False):  # 是否使用偏置
        # out_channels 必须是 in_channels 的整数倍
        if out_channels % in_channels != 0:  # 校验 depth multiplier 整除关系
            raise ValueError(
                f"out_channels ({out_channels}) 必须是 in_channels ({in_channels}) 的整数倍。"
            )

        # 规范化 padding
        if isinstance(padding, str):  # 若传入字符串，则统一小写后判断
            p = padding.lower()  # 统一大小写
            if p == 'valid':  # 'valid' 等价于 0 填充
                padding_arg = 0  # 置为 0
            elif p == 'same':  # 'same' 由较新版本 PyTorch 原生支持
                padding_arg = 'same'  # 直接传给父类
            else:
                raise ValueError("padding 仅支持 'valid' 或 'same'，或传入具体整数/二元组")
        else:
            padding_arg = padding  # 若是 int/tuple 则直接使用

        self.max_norm = float(max_norm) if max_norm is not None else None  # 记录 max-norm 阈值（或 None）

        super().__init__(in_channels=in_channels,  # 调用父类构造，设定输入通道
                         out_channels=out_channels,  # 设定输出通道
                         kernel_size=kernel_size,  # 卷积核尺寸
                         stride=1,  # 深度卷积固定步幅为 1
                         padding=padding_arg,  # 使用规范化后的 padding
                         dilation=1,  # 膨胀系数默认 1
                         groups=in_channels,  # 关键：groups=in_channels 实现深度卷积
                         bias=bias)  # 是否使用偏置

    @property
    def depth_multiplier(self) -> int:  # 只读属性：返回 depth multiplier
        return self.out_channels // self.in_channels  # 输出通道数 / 输入通道数

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # 前向传播
        # 对每个 filter（C_out）在 (kH*kW) 维度上做 L2 max-norm 约束
        if self.max_norm is not None:  # 若启用了 max-norm
            with torch.no_grad():  # 不记录梯度
                w = self.weight.data                         # [C_out, 1, kH, kW]
                w_flat = w.view(self.out_channels, -1)       # [C_out, kH*kW] 展平至空间维
                w_flat = torch.renorm(w_flat, p=2, dim=1, maxnorm=self.max_norm)  # L2 范数裁剪
                w.copy_(w_flat.view_as(w))  # 写回原形状权重
        return super().forward(x)  # 使用父类实现完成卷积计算
    


class SeparableConv2D(nn.Module):  # 定义深度可分离卷积模块（depthwise + pointwise）
    """深度可分离 2D 卷积（Depthwise + Pointwise）。

    先进行 depthwise 卷积（每通道独立卷积，保持通道数不变），再用 1×1 的 pointwise 卷积
    进行通道投影，得到目标输出通道数。两步均使用 stride=1；depthwise 使用 'same' 填充。

    Args:
        in_channels (int): 输入通道数。
        out_channels (int): 输出通道数（pointwise 的输出通道数）。
        kernel_size (tuple): depthwise 卷积的卷积核尺寸 (kH, kW)。

    Attributes:
        conv (nn.Sequential): 顺序容器，依次封装 depthwise 与 pointwise 两层卷积。
    """

    def __init__(self, in_channels, out_channels, kernel_size) -> None:  # 构造函数
        """初始化深度可分离卷积模块。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (tuple): depthwise 卷积核尺寸 (kH, kW)。
        """
        super().__init__()  # 调用父类初始化
        self.conv = nn.Sequential(  # 按顺序组合 depthwise 与 pointwise
            nn.Conv2d(in_channels=in_channels,  # depthwise 卷积
                      out_channels=in_channels,  # 通道数保持不变
                      kernel_size=kernel_size,  # 空间滤波核大小
                      stride=1,  # 步幅 1
                      groups=in_channels,  # 每个通道独立卷积
                      padding='same',  # same 填充，保持空间尺寸
                      bias=False),  # 不使用偏置
            nn.Conv2d(in_channels=in_channels,  # pointwise 卷积（1x1）
                      out_channels=out_channels,  # 投影至新的通道数
                      kernel_size=1,  # 1x1 卷积
                      stride=1,  # 步幅 1
                      padding='same',  # same 填充，保持空间尺寸
                      bias=False),  # 不使用偏置
        )

    def forward(self, x):  # 前向传播
        """执行前向计算。

        Args:
            x (torch.Tensor): 输入张量，形状为 [N, C_in, H, W]。

        Returns:
            torch.Tensor: 输出张量，形状为 [N, C_out, H, W]（same 填充保持 H/W 不变）。
        """
        return self.conv(x)  # 依次执行 depthwise 与 pointwise 并返回结果




#TemporalConvNet输入格式为[batch_size, feature channels, 1d time series data]

#TCNet 的作用是将 TCN模块作用于EEG张量，对于每一个通道进行TCN变换
#TCNet 输入格式为: [batch_size, feature channels, electrode channels, time]
class CausalConv1d(nn.Conv1d):
    """
    因果 1D 卷积：通过左侧 padding 实现因果性，并在前向中剪掉右端多余的 padding。
    等价于：Conv1d(padding=(k-1)*dilation) + Chomp1d((k-1)*dilation)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=False):
        # 只允许 stride=1（TCN 常见设置）；如需支持其他 stride，可在 forward 中相应处理裁剪长度。
        if stride != 1:
            raise ValueError("CausalConv1d 目前仅支持 stride=1。")
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
        y = super().forward(x)           # 先做普通卷积（含左侧 padding）
        if self._left_pad > 0:
            y = y[:, :, :-self._left_pad]  # 再裁掉右侧同等长度，保持长度&因果性
        return y.contiguous()



class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.5):
        super().__init__()
        # 这里的 padding 仍按 (kernel_size-1)*dilation 传入，但只用于初始化检查；真正生效在 CausalConv1d 里自动计算
        assert padding == (kernel_size - 1) * dilation, "因果卷积要求 padding = (k-1)*dilation"

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
        nn.init.kaiming_uniform_(self.net[0].weight)  # 第一层 CausalConv1d 的 weight
        nn.init.kaiming_uniform_(self.net[4].weight)  # 第二层 CausalConv1d 的 weight
        nn.init.ones_(self.net[1].weight); nn.init.zeros_(self.net[1].bias)
        nn.init.ones_(self.net[5].weight); nn.init.zeros_(self.net[5].bias)
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.elu(out + res)



class TCNLayer(nn.Module):
    """TCN（Temporal Convolutional Network）时间卷积网络。"""

    def __init__(self, in_channels, out_channels_list, kernel_size=2, dropout=0.3):
        """初始化 TemporalConvNet 层。

        Args:
            in_channels (int): 输入通道数。
            out_channels_list (list of int): 每层输出通道数列表。
            kernel_size (int, optional): 1D 卷积核大小，默认为 2。
            dropout (float, optional): Dropout 率，默认为 0.3。
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
        """前向传播函数。"""
        return self.network(x)



class TCNet(nn.Module):
    def __init__(self, in_channels, out_channels_list, kernel_size, dropout=0.3):
        super().__init__()
        self.tcn_block = TCNLayer(
            in_channels=in_channels,
            out_channels_list=out_channels_list,
            kernel_size=kernel_size,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # [N, C, E, T] -> 高效批处理
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

class EEGTCNetBackbone(nn.Module):
    """EEGTCNet 神经网络模型。

    Attributes:
        model (nn.Sequential): 由多个卷积层、扩张卷积层、TCN、池化层、Dropout 和全连接层组成。
    初始化 EEGTCNet 模型。

    Args:

        F1 (int): 第一组卷积核数量。
        D (int): 深度卷积扩展因子。
        F2 (int): 第二组卷积核数量。
        KE (int): 空间卷积核大小。
        L (int): TCN 层数。
        FT (int): TCN 特征通道数。
        KT (int): TCN 卷积核大小。
        pe_drop (float): 空间 Dropout 率。
        pt_drop (float): 时间 Dropout 率。
        
    """

    def __init__(self, 
                 input_channels = 1,  # 输入张量的通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 fs=256,  # 采样率（传入 backbone）
                 F1=8, 
                 D=2, 
                 F2=16, 
                 KE=128, 
                 L=2, 
                 FT=12, 
                 KT=4, 
                 pe_drop=0.2, 
                 pt_drop=0.3,
                 ):

        super().__init__()
        
        self.input_channels=input_channels  # 记录输入通道数
        self.input_electrodes=input_electrodes  # 记录电极数
        self.input_times=input_times  # 记录时间长度
        self.fs=fs  # 采样率（传入 backbone）
        self.F1=F1 
        self.D=D
        self.F2=F2
        self.KE=KE
        self.L=L
        self.FT=FT
        self.KT=KT
        self.pe_drop= pe_drop
        self.pt_drop=pt_drop
        
        self.layer1 = nn.Sequential(  # 第一段顺序模块
            nn.Conv2d(in_channels=self.input_channels,  # 时间卷积：沿时间维
                      out_channels=self.F1,  # 输出通道 F1
                      kernel_size=(1, KE),  # 核长 fs//2
                      padding='same',  # same 保持时间维长度不变
                      bias=False,  # 不使用偏置
                      groups=1,),  # 标准卷积（非分组）
            nn.BatchNorm2d(self.F1),  # BN：归一化}
            )
        
        self.layer2= nn.Sequential(
            DepthwiseConv2D(in_channels=self.F1,  # 深度卷积：跨电极（高度）方向
                            out_channels=(self.D * self.F1),  # 输出 D*F1
                            kernel_size=(self.input_electrodes, 1),  # 核覆盖全部电极
                            max_norm=1.0),  # 跨“通道维（电极维）”做 depthwise 卷积；可选权重重归一化
            
            nn.BatchNorm2d(self.D*self.F1),  # BN：深度卷积输出上归一化
            nn.ELU(inplace=True),  # 激活：ELU
            nn.AvgPool2d(kernel_size=(1, 8)),  # 时间维平均池化（/4）
            nn.Dropout(p=self.pe_drop)  # Dropout
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
        
        self.layer4=nn.Sequential(
            TCNet(in_channels=self.F2, 
                  out_channels_list=[self.FT]*self.L, 
                  kernel_size=self.KT, 
                  dropout=self.pt_drop),
            nn.Flatten()
            )

    def forward(self, x):
        """前向传播函数。"""
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        return x

    
#%% 2. EEGTCNet 模型

class EEGTCNet(nn.Module):
    """EEGTCNet 神经网络模型。

    Attributes:
        model (nn.Sequential): 由多个卷积层、扩张卷积层、TCN、池化层、Dropout 和全连接层组成。
    """

    def __init__(self, 
                 input_channels = 1,  # 输入张量的通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 fc_in_channels=256,  # 分类器全连接层输入特征维度
                 num_classes=5,
                 fs=256,  # 采样率（传入 backbone）
                 F1=8, 
                 D=2, 
                 F2=16, 
                 KE=128, 
                 L=2, 
                 FT=12, 
                 KT=4, 
                 pe_drop=0.2, 
                 pt_drop=0.3,
                 is_zscore=True):
                 
                
        super().__init__()
        self.zscore= ZScoreNormalization() if is_zscore else nn.Identity()
        self.backbone=EEGTCNetBackbone( input_channels=input_channels,  # 记录输入通道数
                                        input_electrodes=input_electrodes,  # 记录电极数
                                        input_times=input_times,  # 记录时间长度
                                        fs=fs,  # 采样率（传入 backbone）
                                        F1=F1, 
                                        D=D,
                                        F2=F2,
                                        KE=KE,
                                        L=L,
                                        FT=FT,
                                        KT=KT,
                                        pe_drop= pe_drop,
                                        pt_drop=pt_drop,)
            
        self.classifier=Dense(in_channels=fc_in_channels,  # 线性分类头输入维度（需与 backbone 输出展平维度一致）
                              out_channels=num_classes,  # 输出类别数
                              max_norm=0.25)  # 可选的 L2 max-norm 约束阈值

    def forward(self, x):
        x=self.zscore(x)
        x=self.backbone(x)  # 通过主干网络提取并展平特征
        x=self.classifier(x)  # 送入线性分类器得到 logits
        return x  # 返回分类结果


#%%%
if __name__ == "__main__":

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = EEGTCNet(input_channels= 1, 
                     input_electrodes=64,    
                     input_times=512,
                     fc_in_channels=96,
                     F1=8, 
                     D=2, 
                     F2=16, 
                     KE=128, 
                     L=2,
                     FT=12,
                     KT=4,
                     pe_drop=0.2,
                     pt_drop=0.3).to(device)
    
    x = torch.randn(8, 1, 64, 512, device=device)
    logits = model(x)
    print('logits shape:', logits.shape)  # [8, 5]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    













