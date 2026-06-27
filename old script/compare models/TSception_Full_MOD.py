# -*- coding: utf-8 -*-
"""
Modefied from
https://github.com/yi-ding-cs/TSception

[1]	Y. Ding, N. Robinson, S. Zhang, Q. Zeng, and C. Guan, "TSception: Capturing
 Temporal Dynamics and Spatial Asymmetry From EEG for Emotion Recognition,
 " IEEE Transactions on Affective Computing, vol. 14, no. 3, pp. 2238-2250,
 2023, doi: 10.1109/TAFFC.2022.3169001.

Created on Wed Nov 26 02:16:53 2025

@author: Fujie
"""

electrode_dict = {
    1:  "Fp1", 2: "Fp2",  3: "F7",   4: "F3",    5: "Fz",   6: "F4", 
    7:  "F8",  8: "FC5",  9: "FC1",  10: "FC2",  11: "FC6", 12: "T7",  
    13: "C3",  14: "Cz",  15: "C4",  16: "T8",   17: "TP9", 18: "CP5", 
    19: "CP1", 20: "CP2", 21: "CP6", 22: "TP10", 23: "P7",  24: "P3", 
    25: "Pz",  26: "P4",  27: "P8",  28: "PO9",  29: "O1",  30: "Oz",  
    31: "O2",  32: "PO10",33: "AF7", 34: "AF3",  35: "AF4", 36: "AF8",
    37: "F5",  38: "F1",  39: "F2",  40: "F6",   41: "FT9", 42: "FT7",
    43: "FC3", 44: "FC4", 45: "FT8", 46: "FT10", 47: "C5",  48: "C1",  
    49: "C2",  50: "C6",  51: "TP7", 52: "CP3",  53: "CPz", 54: "CP4",
    55: "TP8", 56: "P5",  57: "P1",  58: "P2",   59: "P6",  60: "PO7",
    61: "PO3", 62: "POz", 63: "PO4", 64: "PO8"
    }

electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]
TS_electrode_list_L=['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1','FC3', 
                     'FC5', 'FT7', 'FT9', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3',
                     'CP5', 'TP7', 'TP9', 'P1', 'P3', 'P5', 'P7', 'PO3', 'O1', 
                     'PO7','PO9',]
TS_electrode_list_R=['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2','FC4', 
                     'FC6', 'FT8', 'FT10', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4',
                     'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'PO4', 'O2', 
                     'PO8','PO10',]
TS_electrode_list=TS_electrode_list_L+TS_electrode_list_R
result = list(set(electrode_list) - set(TS_electrode_list))


import torch
import torch.nn as nn


class TSception(nn.Module):
    """TSception 网络结构，实现时序-空间卷积编码，用于 EEG 信号的分类。

    Attributes:
        Tception1 (nn.Sequential): 第一个时间卷积分支，使用较长的时间窗口。
        Tception2 (nn.Sequential): 第二个时间卷积分支，使用中等的时间窗口。
        Tception3 (nn.Sequential): 第三个时间卷积分支，使用较短的时间窗口。
        Sception1 (nn.Sequential): 第一个空间卷积分支，基于全通道深度卷积。
        Sception2 (nn.Sequential): 第二个空间卷积分支，基于半通道深度卷积。
        fusion_layer (nn.Sequential): 融合后的空间卷积层，用于特征聚合。
        BN_t (nn.BatchNorm2d): 时间分支融合后的批量归一化层。
        BN_s (nn.BatchNorm2d): 空间分支融合后的批量归一化层。
        BN_fusion (nn.BatchNorm2d): 融合卷积后的批量归一化层。
        fc (nn.Sequential): 全连接分类层序列，包括隐藏层、激活、Dropout 和输出层。
    """

    def conv_block(self, in_chan, out_chan, kernel, step, pool):
        """构建一个基本卷积块，包含 Conv2d、LeakyReLU 和 AvgPool2d。

        Args:
            in_chan (int): 输入通道数。
            out_chan (int): 输出通道数。
            kernel (tuple[int, int]): 卷积核大小 (height, width)。
            step (int or tuple[int, int]): 卷积步幅 (stride)。
            pool (int): 池化窗口宽度，仅在时间维度上使用。

        Returns:
            nn.Sequential: 包含卷积、激活和池化的顺序容器。
        """
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_chan,            # 输入通道数
                out_channels=out_chan,          # 输出通道数
                kernel_size=kernel,             # 卷积核大小
                stride=step                     # 卷积步幅
            ),
            nn.LeakyReLU(),                     # 使用 LeakyReLU 激活函数
            nn.AvgPool2d(                       # 在时间维度上做平均池化
                kernel_size=(1, pool),          # 池化窗口 (1, pool)
                stride=(1, pool)                # 池化步幅 (1, pool)
            )
        )

    def __init__(self, num_classes, input_size, sampling_rate,
                 num_T, num_S, hidden, dropout_rate):
        """初始化 TSception 网络。

        Args:
            num_classes (int): 输出分类数。
            input_size (tuple[int, int, int]): 输入张量形状 (1, channels, time_points)。
            sampling_rate (int): EEG 信号采样率（Hz）。
            num_T (int): 每个时间卷积分支的输出通道数。
            num_S (int): 每个空间卷积分支的输出通道数。
            hidden (int): 全连接层隐藏单元数。
            dropout_rate (float): Dropout 比例。

        初始化后属性:
            Tception1, Tception2, Tception3, Sception1, Sception2, fusion_layer,
            BN_t, BN_s, BN_fusion, fc
        """
        super(TSception, self).__init__()     # 调用父类构造函数

        # 不同时间窗口（秒）列表，用于时间分支卷积核长度
        self.inception_window = [0.5, 0.25, 0.125]
        # 时间分支池化比例
        self.pool = 8

        # 时间分支卷积：三个不同长度的卷积核
        self.Tception1 = self.conv_block(
            1,                               # 输入 1 通道（无空间混合）
            num_T,                           # 输出 num_T 通道
            (1, int(self.inception_window[0] * sampling_rate)),  # kernel 对应 0.5s
            1,                               # 步幅为 1
            self.pool                        # 池化窗口宽度
        )
        self.Tception2 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[1] * sampling_rate)),  # kernel 对应 0.25s
            1,
            self.pool
        )
        self.Tception3 = self.conv_block(
            1,
            num_T,
            (1, int(self.inception_window[2] * sampling_rate)),  # kernel 对应 0.125s
            1,
            self.pool
        )

        # 空间分支卷积：基于通道深度可分离卷积
        self.Sception1 = self.conv_block(
            num_T,                           # 输入来自时间分支输出
            num_S,                           # 输出 num_S 通道
            (int(input_size[1]), 1),        # kernel 覆盖所有通道
            1,                               # 步幅为 1
            int(self.pool * 0.25)            # 池化窗口为 pool*0.25
        )
        self.Sception2 = self.conv_block(
            num_T,
            num_S,
            (int(input_size[1] * 0.5), 1),   # kernel 覆盖一半通道
            (int(input_size[1] * 0.5), 1),   # 在空间维度上也步幅覆盖半通道
            int(self.pool * 0.25)
        )

        # 融合分支卷积，用于进一步整合空间特征
        self.fusion_layer = self.conv_block(
            num_S,                           # 输入为两路空间分支拼接后的通道数
            num_S,                           # 保持相同通道数
            (3, 1),                          # 小范围融合卷积核
            1,                               # 步幅为 1
            4                                # 池化窗口为 4
        )

        # 批量归一化层
        self.BN_t = nn.BatchNorm2d(num_T)    # 时间分支融合后
        self.BN_s = nn.BatchNorm2d(num_S)    # 空间分支融合后
        self.BN_fusion = nn.BatchNorm2d(num_S)  # 融合卷积后

        # 全连接分类层序列
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),       # 输入 num_S -> hidden
            nn.ReLU(),                      # ReLU 激活
            nn.Dropout(dropout_rate),       # Dropout
            nn.Linear(hidden, num_classes)  # hidden -> num_classes
        )

    def forward(self, x):
        """定义 TSception 的前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, 1, channels, time_points)。

        Returns:
            torch.Tensor: 网络输出 logits，形状 (batch_size, num_classes)。
        """
        # 3 条时间分支并行计算
        y = self.Tception1(x)               # 第一分支输出
        out = y                             # 初始化拼接输出
        y = self.Tception2(x)               # 第二分支输出
        out = torch.cat((out, y), dim=-1)   # 在时间维度（最后一维）拼接
        y = self.Tception3(x)               # 第三分支输出
        out = torch.cat((out, y), dim=-1)   # 拼接到 out 中

        out = self.BN_t(out)                # 时间分支归一化

        # 两条空间分支并行计算
        z = self.Sception1(out)             # 空间分支1输出
        out_ = z                            # 初始化空间拼接输出
        z = self.Sception2(out)             # 空间分支2输出
        out_ = torch.cat((out_, z), dim=2)  # 在空间维度（height 维）拼接

        out = self.BN_s(out_)               # 空间分支归一化

        out = self.fusion_layer(out)        # 融合卷积
        out = self.BN_fusion(out)           # 融合归一化

        # 全局平均池化：在时间维度上求均值，然后去除多余维度
        out = torch.squeeze(torch.mean(out, dim=-1), dim=-1)

        out = self.fc(out)                  # 全连接分类层

        return out                          # 返回最终 logits


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


class TSception_Full(nn.Module):
    def __init__(self, 
                 all_electrodes_list,
                 ts_electrodes_list,
                 num_classes, 
                 fs=256,
                 num_T=15,
                 num_S=15, 
                 hidden=60, 
                 dropout_rate=0.5,
                 input_channels = 1,  # 输入张量的通道数
                 input_electrodes = 64,  # EEG 电极数量
                 input_times = 512,  # 时间长度
                 is_zscore=False,
                 ):
        
        super().__init__()     # 调用父类构造函数
        self.input_size=tuple([input_channels, input_electrodes, input_times])
        
        self.indices = [all_electrodes_list.index(item) for item in ts_electrodes_list if item in all_electrodes_list]
        
        self.zscore= ZScoreNormalization() if is_zscore else nn.Identity()
        
        self.tsception=TSception(num_classes = num_classes,
                                 input_size = self.input_size, 
                                 sampling_rate = fs,
                                 num_T = num_T, 
                                 num_S = num_S, 
                                 hidden = hidden, 
                                 dropout_rate = dropout_rate)


    def forward(self, x):
        
        x = x[:, :, self.indices, :]
        x=self.zscore(x)
        return self.tsception(x)

#%%
if __name__ == "__main__":

    electrode_dict = {
        1:  "Fp1", 2: "Fp2",  3: "F7",   4: "F3",    5: "Fz",   6: "F4", 
        7:  "F8",  8: "FC5",  9: "FC1",  10: "FC2",  11: "FC6", 12: "T7",  
        13: "C3",  14: "Cz",  15: "C4",  16: "T8",   17: "TP9", 18: "CP5", 
        19: "CP1", 20: "CP2", 21: "CP6", 22: "TP10", 23: "P7",  24: "P3", 
        25: "Pz",  26: "P4",  27: "P8",  28: "PO9",  29: "O1",  30: "Oz",  
        31: "O2",  32: "PO10",33: "AF7", 34: "AF3",  35: "AF4", 36: "AF8",
        37: "F5",  38: "F1",  39: "F2",  40: "F6",   41: "FT9", 42: "FT7",
        43: "FC3", 44: "FC4", 45: "FT8", 46: "FT10", 47: "C5",  48: "C1",  
        49: "C2",  50: "C6",  51: "TP7", 52: "CP3",  53: "CPz", 54: "CP4",
        55: "TP8", 56: "P5",  57: "P1",  58: "P2",   59: "P6",  60: "PO7",
        61: "PO3", 62: "POz", 63: "PO4", 64: "PO8"
        }

    electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]
    TS_electrode_list_L=['Fp1', 'AF3', 'AF7', 'F1', 'F3', 'F5', 'F7', 'FC1','FC3', 
                         'FC5', 'FT7', 'FT9', 'C1', 'C3', 'C5', 'T7', 'CP1', 'CP3',
                         'CP5', 'TP7', 'TP9', 'P1', 'P3', 'P5', 'P7', 'PO3', 'O1', 
                         'PO7','PO9',]
    TS_electrode_list_R=['Fp2', 'AF4', 'AF8', 'F2', 'F4', 'F6', 'F8', 'FC2','FC4', 
                         'FC6', 'FT8', 'FT10', 'C2', 'C4', 'C6', 'T8', 'CP2', 'CP4',
                         'CP6', 'TP8', 'TP10', 'P2', 'P4', 'P6', 'P8', 'PO4', 'O2', 
                         'PO8','PO10',]
    
    TS_electrode_list=TS_electrode_list_L+TS_electrode_list_R
    
    device = 'cpu'
    

    model = TSception_Full(                 
        all_electrodes_list = electrode_list,
        ts_electrodes_list = TS_electrode_list,
        input_channels = 1,
        input_electrodes = len(TS_electrode_list),
        input_times = 512,
        num_classes=5,
        fs=256,
        num_T = 15,
        num_S = 15, 
        hidden = 60, 
        dropout_rate = 0.5,
        ).to(device)

    
    
    x = torch.randn(8, 1, len(electrode_list), 512, device=device)
    print(model(x).shape)
    # logits = model(x)
    # print('logits shape:', logits.shape)  # [8, 5]
    
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)





















    
    
    
    
    
    
    
    
    
    
    
    
    
    
    