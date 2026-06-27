# -*- coding: utf-8 -*-  # Specify source file encoding as UTF-8
"""
Ref:
[1]	V. J. Lawhern, A. J. Solon, N. R. Waytowich, S. M. Gordon, C. P. Hung, and 
B. J. Lance, "EEGNet: a compact convolutional neural network for EEG-based 
brain–computer interfaces," Journal of Neural Engineering, vol. 15, no. 5,
 p. 056013, 2018/07/27 2018, doi: 10.1088/1741-2552/aace8c.

20251023 编辑
@author: Fujie 
"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.nn.parameter import UninitializedParameter 
from typing import Tuple, Union, Optional

# in_channel:  Number of input channels
# out_channel: Number of output channels
# k: kernel_size  -> Convolution kernel size
# s: stride  -> Stride length
# p: padding  -> Padding size
# b: bias  -> Whether to include bias term
# tensor shape: [batch_size, feature channels, electrode channels, time]  -> Expected tensor dimensions

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



class EEGNetBackbone(nn.Module): 
    """EEGNet 主干网络（Backbone）。

    由两段卷积-归一化-激活-池化的结构组成：第一段包含时间卷积与跨通道的深度卷积，
    第二段为可分离卷积。末尾进行展平，为分类头提供特征。

    Args:
        input_channels (int): 输入的“伪图像”通道数（EEG 通常为 1）。
        input_electrodes (int): EEG 电极（通道）数量 C。
        input_times (int): 单段信号的时间长度 T。
        fs (int): 采样率（用于设定时间卷积核长度等）。
        F1 (int): 第一段时间卷积的基础滤波器数量。
        D (int): 深度卷积的倍增系数（depth multiplier）。
        F2 (int): 可分离卷积的输出通道数（通常取 D*F1）。
        p_drop (float): Dropout 比例。

    Attributes:
        conv_ks_1_1 (Tuple[int, int]): 第一段时间卷积核大小 (1, fs//2)。
        pool_ks_1_1 (Tuple[int, int]): 第一段池化核大小 (1, 4)。
        conv_ks_2_1 (Tuple[int, int]): 第二段 depthwise 时间核大小 (1, fs//8)。
        pool_ks_2_1 (Tuple[int, int]): 第二段池化核大小 (1, 8)。
        block1 (nn.Sequential): 第一段模块（Conv → BN → Depthwise → BN → ELU → Pool → Dropout）。
        block2 (nn.Sequential): 第二段模块（Separable → BN → ELU → Pool → Dropout → Flatten）。
    """

    def __init__(self, 
                 input_channels = 1,  # 输入伪图像通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 fs=256,  # 采样率
                 F1=8,  # 第一段基础滤波器数
                 D=4,  # 深度卷积倍增系数
                 F2=32,  # 可分离卷积输出通道数
                 p_drop=0.25):  # Constructor  # 构造函数，设置默认超参数
        super().__init__()  # Initialize base nn.Module  # 调用父类 nn.Module 的初始化
        self.input_channels=input_channels  # 记录输入通道数
        self.input_electrodes=input_electrodes  # 记录电极数
        self.input_times=input_times  # 记录时间长度

        self.F1 = F1  # 保存 F1 滤波器数量
        self.D = D    # 保存深度倍增系数 D
        self.F2 = F2   # 保存 F2 滤波器数量
        self.p_drop = p_drop    # 保存 Dropout 概率
        
        self.conv_ks_1_1=(1, int(fs//2))  # 第一段时间卷积核：采样率一半
        self.pool_ks_1_1=(1, 4)  # 第一段池化核：固定 4
        self.conv_ks_2_1=(1, max(1, fs//8))  # 第二段 depthwise 时间核：约 0.5 秒
        self.pool_ks_2_1=(1, 8)  # 第二段池化核：固定 8

        
        #Block 1
        self.block1 = nn.Sequential(  # 第一段顺序模块
            nn.Conv2d(in_channels=self.input_channels,  # 时间卷积：沿时间维
                      out_channels=self.F1,  # 输出通道 F1
                      kernel_size=self.conv_ks_1_1,  # 核长 fs//2
                      padding='same',  # same 保持时间维长度不变
                      bias=False,  # 不使用偏置
                      groups=1,),  # 标准卷积（非分组）
            nn.BatchNorm2d(self.F1),  # BN：归一化
            DepthwiseConv2D(in_channels=self.F1,  # 深度卷积：跨电极（高度）方向
                            out_channels=(self.D * self.F1),  # 输出 D*F1
                            kernel_size=(self.input_electrodes, 1),  # 核覆盖全部电极
                            max_norm=1.0),  # 跨“通道维（电极维）”做 depthwise 卷积；可选权重重归一化
            
            nn.BatchNorm2d(self.D*self.F1),  # BN：深度卷积输出上归一化
            nn.ELU(inplace=True),  # 激活：ELU
            nn.AvgPool2d(kernel_size=self.pool_ks_1_1),  # 时间维平均池化（/4）
            nn.Dropout(self.p_drop)  # Dropout
            )
        
        self.block2 = nn.Sequential(  # 第二段顺序模块
            SeparableConv2D(in_channels=(self.D * self.F1),  # 可分离卷积（depthwise+pointwise）
                            out_channels=self.F2,  # 输出 F2
                            kernel_size=self.conv_ks_2_1),  # 先 depthwise 再 pointwise 的可分离卷积（沿时间维 1×(fs//8)）
            nn.BatchNorm2d(self.F2),  # 批归一化（通道数按此处写法保持不变）
            nn.ELU(inplace=True),  # 激活：ELU
            nn.AvgPool2d(kernel_size= self.pool_ks_2_1),  # 时间维平均池化（/8）
            nn.Dropout(self.p_drop),  # 再次正则化
            nn.Flatten()  # 展平为 [N, F2*(T/32)]
            )
            
    def forward(self, x):  # Forward pass for EEGNet  # 定义前向传播：输入 -> 顺序模型
        """执行前向传播。

        将输入张量依次通过 Block 1 与 Block 2，得到展平后的特征。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 [N, 1, C, T]，
                其中 N 为批大小，1 为“伪图像”的通道维，C≈n_eegchs 为电极数，T 为时间长度。

        Returns:
            torch.Tensor: 展平后的特征张量，形状约为 [N, F2*(T/32)]。
        """
        x= self.block1(x)  # 通过第一段
        x= self.block2(x)  # 通过第二段并展平
        return x  # 返回特征


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


    

#%% 2. full mode

class EEGNet(nn.Module): 
    """EEGNet 顶层模型：由主干特征提取网络与线性分类头组成。

    本类封装了一个 EEGNetBackbone（用于卷积特征提取）以及一个 Dense 分类器（线性层）。
    典型流程为：输入张量 → backbone 提取特征（卷积/池化/展平）→ classifier 输出各类别分数。

    Args:
        input_channels (int): 输入张量的通道数（通常为 1，表示“伪图像”的通道维）。
        input_electrodes (int): EEG 电极数量（空间维度 C）。
        input_times (int): 单段 EEG 信号的时间长度（时间维度 T）。
        fc_in_channels (int): 分类器 `Dense` 层的输入特征维度（需与 backbone 输出展平后的大小一致）。
        num_classes (int): 分类类别数（分类器输出维度）。
        fs (int): 采样率（传入 `EEGNetBackbone`，用于设定时间卷积核长度等）。
        F1 (int): 第一块卷积的基础滤波器数量（传入 `EEGNetBackbone`）。
        D (int): 深度卷积的倍增系数（depth multiplier，传入 `EEGNetBackbone`）。
        F2 (int): 第二块可分离卷积的输出通道数（传入 `EEGNetBackbone`）。
        p_drop (float): Dropout 概率（传入 `EEGNetBackbone`）。

    Attributes:
        backbone (EEGNetBackbone): 特征提取主干网络（卷积/池化/展平）。
        classifier (Dense): 线性分类头，输入为 `fc_in_channels`，输出为 `num_classes`。
    """

    def __init__(self, 
                 input_channels = 1,  # 输入张量的通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 fc_in_channels=256,  # 分类器全连接层输入特征维度
                 num_classes=5,  # 类别数
                 fs=256,  # 采样率（传入 backbone）
                 F1=8,  # 传入 backbone 的超参：第一块卷积的基础滤波器数
                 D=4,  # 传入 backbone 的超参：深度卷积倍增系数
                 F2=32,  # 传入 backbone 的超参：可分离卷积输出通道数
                 p_drop=0.25,
                 is_zscore=True):  # Constructor  # 构造函数，设置默认超参数
        super().__init__()  # Initialize base nn.Module  # 调用父类 nn.Module 的初始化
        self.zscore= ZScoreNormalization() if is_zscore else nn.Identity()
        self.backbone=EEGNetBackbone(input_channels = input_channels,  # 主干网络：负责卷积特征提取
                                    input_electrodes = input_electrodes,  # 传入电极数
                                    input_times = input_times,  # 传入时间长度
                                    fs=fs,  # 传入采样率
                                    F1=F1,  # 传入 F1
                                    D=D,  # 传入 D
                                    F2=F2,  # 传入 F2
                                    p_drop=p_drop)  # 传入 Dropout 概率
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
if __name__ == "__main__":

    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    model = EEGNet(input_channels=1,
                   input_electrodes=64,
                   input_times=512,
                   fc_in_channels=32 * (512 // 32),  # = 512
                   num_classes=5,
                   fs=256, F1=8, D=4, F2=32, p_drop=0.5).to(device)
    
    x = torch.randn(8, 1, 64, 512, device=device)
    logits = model(x)
    print('logits shape:', logits.shape)  # [8, 5]
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    
    