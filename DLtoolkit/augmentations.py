# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:51:01 2025

@author: Fujie
"""

import torch
import numpy as np
from scipy.interpolate import interp1d
#%% Mixup Method

def mixup_data(x: torch.Tensor,
               y: torch.Tensor,
               alpha: float = 0.2,
               device: torch.device = torch.device('cuda')):
    """执行 Mixup 数据增强，返回混合后的输入、标签对及混合系数。

    Mixup 是一种数据增强技术，将两个样本按比例线性组合，从而生成新的训练样本，
    有助于提高模型的泛化能力。

    公式：
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a = y
        y_b = y[index]

    Args:
        x (torch.Tensor): 输入特征张量，shape 为 (batch_size, ...)，可以是图像、语音或其他特征。
        y (torch.Tensor): 对应的标签张量，shape 为 (batch_size,) 或 (batch_size, num_classes)。
        alpha (float, optional): Beta 分布的 α 参数，控制混合强度。默认为 0.2。
                                 当 alpha <= 0 时，不进行混合（lam = 1.0）。
        device (torch.device, optional): 存放张量的设备（如 'cuda' 或 'cpu'）。默认为 GPU。

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
            - mixed_x: 混合后的输入特征张量。
            - y_a: 原始标签。
            - y_b: 随机打乱后的标签。
            - lam: 混合系数（float）。
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 从 Beta 分布中采样混合系数 lam
    else:
        lam = 1.0  # alpha <= 0 时，不做混合，lam 固定为 1
    batch_size = x.size(0)  # 获取 batch 大小
    index = torch.randperm(batch_size).to(device)  # 在 [0, batch_size) 范围内生成随机排列索引，并放到指定设备
    mixed_x = lam * x + (1 - lam) * x[index]  # 按比例混合原样本和打乱样本
    y_a, y_b = y, y[index]  # 保留原标签和打乱后的标签
    return mixed_x, y_a, y_b, lam  # 返回混合数据、两组标签及混合系数



def mixup_criterion(criterion, pred, y_a, y_b, lam: float):
    """计算 Mixup 数据增强下的加权损失。

    在 Mixup 训练中，一个样本的标签由两部分组成 (y_a, y_b)，
    它们分别对应原始样本和随机混合样本的标签。
    该函数按混合系数 lam 对两份标签的损失进行加权求和。

    Args:
        criterion (Callable): 损失函数（例如 nn.CrossEntropyLoss 实例）。
        pred (torch.Tensor): 模型的预测输出。
        y_a (torch.Tensor): 原始标签。
        y_b (torch.Tensor): 混合后对应的另一份标签。
        lam (float): 混合系数，取值范围 [0, 1]。

    Returns:
        torch.Tensor: 按 lam 加权后的总损失值。
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)  # 按 lam 比例加权两个标签的损失

