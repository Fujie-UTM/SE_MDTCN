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

#%%
def _single_pink_noise(n: int, scale: float) -> np.ndarray:
    """
    生成长度为 n、标准差为 scale 的 1/f（粉红）噪声。
    """
    # 频率分量
    f = np.fft.rfftfreq(n, d=1.0)
    # 避免直流分量除以零
    if len(f) > 1:
        f[0] = f[1]
    # 振幅按 1/f 下降，随机相位
    amplitudes = 1.0 / f
    phases = np.exp(2j * np.pi * np.random.rand(len(amplitudes)))
    spectrum = amplitudes * phases
    # 逆傅里叶变换
    noise = np.fft.irfft(spectrum, n=n)
    # 归一化到单位方差后再缩放
    noise = noise / np.std(noise)
    return noise * scale

def _generate_pink_noise(shape: tuple, scale: float) -> np.ndarray:
    """
    对给定 shape，最后一维生成独立的粉红噪声，其它维度并行。
    """
    out = np.zeros(shape, dtype=float)
    # 遍历除最后一维之外的所有索引
    for idx in np.ndindex(shape[:-1]):
        out[idx] = _single_pink_noise(shape[-1], scale)
    return out

def augment_with_noise(
    X_batch: np.ndarray,
    y_batch: np.ndarray,
    noise_type: str = 'white',
    scale_ratio: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """
    对每个样本按其自身方差添加噪声。

    参数
    ----
    X_batch : np.ndarray
        形状 (batch_size, ..., T) 的数据，最后一维为时间/信号通道。
    y_batch : np.ndarray
        对应标签，不变返回。
    noise_type : {'white', 'pink'}
        噪声类型：'white'（高斯白噪声）或 'pink'（1/f 粉红噪声）。
    scale_ratio : float
        噪声标准差为 sample_std * scale_ratio。

    返回
    ----
    X_noisy : np.ndarray
        添加噪声后的数据，shape 同 X_batch。
    y_batch : np.ndarray
        原标签，未修改。
    """
    X_noisy = X_batch.astype(float).copy()
    batch_size = X_batch.shape[0]

    for i in range(batch_size):
        sample = X_batch[i]
        sample_std = np.std(sample)
        scale = sample_std * scale_ratio

        if noise_type == 'white':
            noise = np.random.normal(loc=0.0, scale=scale, size=sample.shape)
        elif noise_type == 'pink':
            noise = _generate_pink_noise(sample.shape, scale)
        else:
            raise ValueError(f"Unsupported noise_type '{noise_type}'. Choose 'white' or 'pink'.")

        X_noisy[i] += noise

    return X_noisy, y_batch

def augment_time_series_sliding(
    X: np.ndarray,
    y: np.ndarray,
    N: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    对 3D 时序信号做：线性插值到长度 T+N + 滑动窗口切片。

    参数
    ----
    X : np.ndarray, shape (S, E, T)
        S 个样本，E 个通道，T 个时间点。
    y : np.ndarray, shape (S,)
        原标签。
    N : int
        插值后多出的点数，也是滑窗切片数，输出 S*N 条样本。

    返回
    ----
    X_aug : np.ndarray, shape (S*N, E, T)
    y_aug : np.ndarray, shape (S*N,)
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (S,E,T), got {X.shape}")
    S, E, T = X.shape

    y = np.asarray(y)
    if y.ndim != 1 or y.shape[0] != S:
        raise ValueError(f"y must be 1D length S={S}, got {y.shape}")

    # 1) 线性插值到 T+N
    t_orig = np.arange(T)
    t_new = np.linspace(0, T - 1, T + N)

    # 把 (S,E,T) 拉成 (S*E, T)，批量用 np.interp
    flat = X.reshape(-1, T)  # shape = (S*E, T)
    interp_flat = np.stack([
        np.interp(t_new, t_orig, row)
        for row in flat
    ], axis=0)            # (S*E, T+N)
    X_interp = interp_flat.reshape(S, E, T + N)

    # 2) 滑动窗口切片，直接按切片次数填充
    X_aug = np.empty((S * N, E, T), dtype=X.dtype)
    for i in range(N):
        # 对所有样本，第 i 段窗口 [i : i+T]
        X_aug[i * S : (i + 1) * S] = X_interp[:, :, i : i + T]

    # 3) 标签重复
    y_aug = np.repeat(y, N)

    return X_aug, y_aug


class EEGCropTransformer:
    """
    使用 fit/transform/fit_transform 接口：
      - 插值扩展：将每个通道的时域序列从 orig_length 插值到 new_length
      - 滑动窗口裁剪：窗口长度 window_size，步长 stride
      - 支持同时处理标签：标签按窗口重复

    示例：
        transformer = EEGCropTransformer(new_length=1050, window_size=1000, stride=10)
        X_aug, y_aug = transformer.fit_transform(X, y)
    """
    def __init__(self,
                 new_length: int = 1050,
                 window_size: int = 1000,
                 stride: int = 10,
                 kind: str = 'linear'):
        self.new_length = new_length
        self.window_size = window_size
        self.stride = stride
        self.kind = kind
        self.n_windows = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        X: np.ndarray of shape (B, C, T)
        y: np.ndarray of shape (B,) or None
        """
        B, C, T = X.shape
        # 插值后窗口数量
        self.n_windows = (self.new_length - self.window_size) // self.stride + 1
        return self

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """
        返回增强后的 (X_aug, y_aug)：
            X_aug: shape (B * n_windows, C, window_size)
            y_aug: shape (B * n_windows,) if y is provided, else None
        """
        B, C, T = X.shape
        # 1) 插值到 new_length
        orig_x = np.arange(T)
        new_x = np.linspace(0, T - 1, self.new_length)
        X_interp = np.zeros((B, C, self.new_length), dtype=X.dtype)
        for i in range(B):
            for ch in range(C):
                f = interp1d(orig_x, X[i, ch], kind=self.kind)
                X_interp[i, ch] = f(new_x)
        # 2) 滑动窗口裁剪
        X_crops = np.zeros((B, self.n_windows, C, self.window_size), dtype=X.dtype)
        for i in range(self.n_windows):
            start = i * self.stride
            end = start + self.window_size
            X_crops[:, i, :, :] = X_interp[:, :, start:end]
        # 3) 重塑成 (B*n_windows, C, window_size)
        X_aug = X_crops.reshape(B * self.n_windows, C, self.window_size)
        # 4) 标签重复
        y_aug = None
        if y is not None:
            # y: (B,) -> (B, n_windows) -> flatten to (B*n_windows,)
            y_repeat = np.repeat(y[:, None], self.n_windows, axis=1)
            y_aug = y_repeat.reshape(-1)
        return (X_aug, y_aug) if y is not None else X_aug

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        """
        直接对 X, y 进行 fit + transform，返回 (X_aug, y_aug)
        """
        return self.fit(X, y).transform(X, y)
