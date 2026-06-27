# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:54:09 2025

@author: Fujie
"""
# Standard library
import copy
import json
import logging
import math
import os
import random
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Union, Tuple, Dict, Any, List, Literal

# Third-party libraries
import h5py
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import  LambdaLR, SequentialLR, LinearLR, CosineAnnealingLR
from torch.optim import Optimizer
import torch.profiler
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

from pathlib import Path
# Scikit-learn

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    train_test_split,
    ParameterGrid
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
)

import warnings

from .augmentations import mixup_data, mixup_criterion
from .utils import plot_acc_loss, ChannelZScoreScaler, EEGDataset
#%%
def run_one_epoch( 
    dataloader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: torch.device,
    mode: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_norm: Optional[float] = None,
    log_steps: int = 100,
    show_progress: bool = False,
    mixup_alpha: Optional[float] = None,    # 新增 MixUp 强度参数
) -> tuple:
    """运行一个训练或验证 epoch，并返回损失、准确率及（验证模式下）预测与真实标签。

    支持 Mixup 数据增强。在训练模式下可进行反向传播与参数更新，
    在验证模式下仅计算指标并保存预测结果。

    Args:
        dataloader (DataLoader): PyTorch 数据加载器。
        model (torch.nn.Module): 待训练或评估的模型。
        loss_fn (torch.nn.Module): 损失函数。
        device (torch.device): 运行设备（CPU 或 GPU）。
        mode (str): 'train' 表示训练模式，'val' 或 'test' 表示评估模式。
        optimizer (Optional[torch.optim.Optimizer], optional): 优化器，仅在训练模式下必需。
        max_norm (Optional[float], optional): 梯度裁剪的最大范数，None 表示不裁剪。
        log_steps (int, optional): 训练时每隔多少步更新一次进度条信息。默认为 100。
        show_progress (bool, optional): 是否显示进度条。默认为 False。
        mixup_alpha (Optional[float], optional): Mixup 数据增强的 alpha 参数，None 或 <=0 表示不使用 Mixup。

    Returns:
        tuple:
            - 训练模式: (epoch_loss, epoch_acc)
            - 验证模式: (epoch_loss, epoch_acc, all_true, all_pred)
              其中 all_true 与 all_pred 为 numpy 数组。
    
    Raises:
        AssertionError: 在训练模式下未提供 optimizer 时抛出。
        ValueError: 标签格式或形状不符合预期时抛出。
    """
    assert len(dataloader) > 0, f"{mode}: empty dataloader"
    is_train = (mode == "train")  # 判断是否为训练模式
    if is_train:
        assert optimizer is not None, "In train mode must provide optimizer"  # 训练模式必须提供优化器
        model.train()  # 设置模型为训练模式
    else:
        model.eval()   # 设置模型为评估模式

    total_loss, total_correct, total_samples = 0.0, 0, 0  # 初始化累计损失、正确数、样本数
    all_true, all_pred = [], []  # 用于保存验证模式下的真实标签和预测标签

    ctx = torch.enable_grad() if is_train else torch.no_grad()  # 根据模式选择上下文管理器
    with ctx:
        iterator = (
            tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=mode, leave=False)
            if show_progress else
            enumerate(dataloader, 1)
        )  # 根据是否显示进度条选择迭代器

        for step, (X, y_orig) in iterator:  # 遍历批次数据
            X, y_orig = X.to(device), y_orig.to(device)  # 将数据转移到指定设备

            if is_train:
                optimizer.zero_grad()  # 清空梯度

            # —— MixUp 分支 —— 
            if is_train and mixup_alpha is not None and mixup_alpha > 0:
                # 得到混合输入和两份标签
                mixed_X, y_a, y_b, lam = mixup_data(X, y_orig, alpha=mixup_alpha, device=device)
                logits = model(mixed_X)  # 前向传播
                # 按 lam 加权两份标签的损失
                loss = mixup_criterion(loss_fn, logits, y_a, y_b, lam)

                # 计算“近似”准确度：这里我们以 y_a 为准
                preds = logits.argmax(dim=1)  # 获取预测类别
                soft_correct = (preds.eq(y_a).float() * lam + preds.eq(y_b).float() * (1 - lam)).sum().item()
                batch_correct = soft_correct
                # batch_correct = preds.eq(y_a).sum().item()  # 计算预测正确数
            else:
                # —— 原始逻辑 —— 
                logits = model(X)  # 前向传播 [B, C]

                # 标签预处理
                if (y_orig.dtype == torch.long and y_orig.dim() == 1) or \
                   (y_orig.dtype == torch.float and y_orig.dim() == 1):
                    true_labels = y_orig.long()  # 确保标签为 long 类型
                    loss = loss_fn(logits, true_labels)  # 计算损失
                else:
                    if y_orig.dtype != torch.float or y_orig.shape != logits.shape:
                        raise ValueError(
                            f"Unsupported target: expected 1D long labels of shape [B] "
                            f"or 2D float distributions of shape {tuple(logits.shape)}, "
                            f"but got dtype={y_orig.dtype}, shape={tuple(y_orig.shape)}"
                        )  # 检查标签格式是否符合要求
                    row_sums = y_orig.sum(dim=1)  # 计算每行的和
                    probs = y_orig if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6) \
                            else F.softmax(y_orig, dim=1)  # 若行和不是 1，则转为概率分布
                    loss = loss_fn(logits, probs)  # 计算损失
                    true_labels = probs.argmax(dim=1)  # 获取标签类别索引

                preds = logits.argmax(dim=1)  # 获取预测类别
                batch_correct = preds.eq(true_labels).sum().item()  # 计算预测正确数

            # —— 反向 & 更新，仅在训练 —— 
            if is_train:
                loss.backward()  # 反向传播
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # 梯度裁剪
                optimizer.step()  # 更新参数

            # —— 指标累加 —— 
            batch_size = X.size(0)  # 获取当前批次大小
            total_loss   += loss.item() * batch_size  # 累加总损失（按样本数加权）
            total_correct+= batch_correct            # 累加正确预测数
            total_samples+= batch_size               # 累加样本总数

            if not is_train:
                all_true.append(true_labels.cpu())  # 保存真实标签
                all_pred.append(preds.cpu())        # 保存预测标签

            if is_train and show_progress and (step % log_steps == 0 or step == len(dataloader)):
                cur_loss = total_loss / total_samples  # 当前平均损失
                cur_acc  = total_correct / total_samples  # 当前平均准确率
                iterator.set_postfix({"loss":f"{cur_loss:.4f}", "acc":f"{cur_acc:.4f}"})  # 更新进度条信息

    epoch_loss = total_loss / total_samples  # 计算整个 epoch 的平均损失
    epoch_acc  = total_correct / total_samples  # 计算整个 epoch 的平均准确率
    logging.info(f"{mode} epoch — loss: {epoch_loss:.4f}, acc: {epoch_acc:.1%}")  # 打印日志

    if is_train:
        return epoch_loss, epoch_acc  # 训练模式返回损失和准确率
    else:
        return epoch_loss, epoch_acc, \
               torch.cat(all_true).numpy(), torch.cat(all_pred).numpy()  # 验证模式返回损失、准确率、真实标签和预测标签



def get_epoch_warmup_cosine_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    num_epochs: int,
) -> SequentialLR:
    """构造 epoch 级别的 Linear Warm-up + Cosine Decay 学习率调度器。

    该函数使用 ``torch.optim.lr_scheduler.SequentialLR`` 将
    ``LinearLR``（线性预热）和 ``CosineAnnealingLR``（余弦衰减）串联，
    用于训练中前期逐步升高学习率，后期按余弦曲线衰减学习率。

    调度器每个 epoch 更新一次学习率，适合以 epoch 为粒度的训练。

    Args:
        optimizer (Optimizer): PyTorch 优化器实例。
        warmup_epochs (int): 预热阶段的 epoch 数，必须 >= 0。
        num_epochs (int): 总训练轮数，必须 >= 1。

    Returns:
        SequentialLR: 串联的学习率调度器。

    Raises:
        AssertionError: 当 warmup_epochs < 0 或 num_epochs < 1 时抛出。
    """
    assert warmup_epochs >= 0, "warmup_epochs must be >= 0"  # 确保预热阶段的 epoch 数有效
    assert num_epochs >= 1, "num_epochs must be >= 1"  # 确保总 epoch 数有效

    # 计算衰减阶段的 epoch 数
    decay_epochs = num_epochs - warmup_epochs  # 衰减阶段的 epoch 数

    # 如果衰减阶段的 epoch 数为 0 或负数，抛出异常
    assert decay_epochs > 0, "num_epochs must be greater than warmup_epochs"

    schedulers = []   # 用于存放各阶段调度器
    milestones = []   # 用于 SequentialLR 指定切换点

    # 1) 线性预热阶段：factor 从 (1/warmup_epochs) → 1
    if warmup_epochs > 0:
        start_factor = 1.0 / warmup_epochs  # 初始学习率缩放因子
        schedulers.append(
            LinearLR(
                optimizer,
                start_factor=start_factor,  # 起始缩放比例
                end_factor=1.0,              # 结束缩放比例（全量）
                total_iters=warmup_epochs    # 迭代次数（按 epoch 计）
            )
        )
        milestones.append(warmup_epochs)  # 第一次切换的 epoch 索引

    # 2) 余弦衰减阶段
    decay_epochs = max(decay_epochs, 1)  # 至少衰减 1 个 epoch
    schedulers.append(
        CosineAnnealingLR(
            optimizer,
            T_max=decay_epochs,  # 衰减周期
            eta_min=0.0          # 最小学习率
        )
    )

    # 3) 串联两个调度器
    return SequentialLR(
        optimizer,
        schedulers=schedulers,  # 顺序执行的调度器列表
        milestones=milestones   # 阶段切换点
    )



def EEG_train_validate(
    train_ds: Dataset,
    valid_ds: Dataset,

    model_cls: Callable[..., torch.nn.Module],
    model_kwargs: dict,

    optimizer_cls: Callable[..., torch.optim.Optimizer],
    optimizer_kwargs: dict,

    loss_fn: torch.nn.Module,

    batch_size: int,
    num_epochs: int,
    device: torch.device,
    save_dir: str,
    random_state: int,

    warmup_epochs: Optional[int] = None,  # None → 恒定 LR；否则 warmup_epochs>=0
    mixup_alpha: Optional[float] = None,    # 新增 MixUp 强度参数

    early_stopping_patience: Optional[int] = None,
    max_norm: Optional[float] = None,
    log_steps: int = 100,
    is_plot: bool = True,
    is_parameters_plot: bool =True,
    figure_note: Optional[str] = None,
    best_by: Literal['acc', 'loss'] = 'acc',
    
) -> pd.DataFrame:
    """基于 epoch 的训练/验证主循环，支持 warmup+cosine LR、MixUp、早停与可视化。

    该函数封装了完整的 EEG 训练流程：构建 DataLoader、实例化模型与优化器、
    可选的 epoch 级 warm-up + cosine 衰减学习率调度、MixUp 数据增强、训练与验证指标统计、
    早停（基于验证集）、日志与曲线绘制与导出。

    - 当 `warmup_ratio is None` 时：全程使用 `optimizer_kwargs` 中的基础学习率（恒定 LR）。
    - 当 `warmup_ratio in [0, 1)` 时：按比例进行 **Linear Warm-up**（前 `warmup_ratio * num_epochs` 个 epoch）
      与 **Cosine Decay**（其余 epoch），每个 epoch 末调用 `scheduler.step()`。

    Args:
        train_ds (Dataset): 训练集 `torch.utils.data.Dataset`。
        valid_ds (Dataset): 验证集 `torch.utils.data.Dataset`。
        model_cls (Callable[..., torch.nn.Module]): 模型类（可调用，返回 `nn.Module` 实例）。
        model_kwargs (dict): 传给 `model_cls` 的关键字参数。
        optimizer_cls (Callable[..., torch.optim.Optimizer]): 优化器类（如 `torch.optim.Adam`）。
        optimizer_kwargs (dict): 传给 `optimizer_cls` 的关键字参数。
        loss_fn (torch.nn.Module): 损失函数实例（如 `nn.CrossEntropyLoss`）。
        batch_size (int): 批大小。
        num_epochs (int): 训练的总 epoch 数。
        device (torch.device): 训练/验证使用的设备（CPU/GPU）。
        save_dir (str): 保存输出（最佳模型、日志、图像等）的目录。
        random_state (int): 随机种子，用于可复现。
        warmup_epochs (Optional[int]): 学习率预热的epoch数目。`None` 表示恒定学习率；
            否则应>=1，启用 epoch 级 warmup + cosine 衰减。
        mixup_alpha (Optional[float]): MixUp 的 Beta 分布 α，`None` 或 ≤0 表示不启用。
        early_stopping_patience (Optional[int]): 早停的容忍 epoch 数；为 `None` 时不早停。
        max_norm (Optional[float]): 梯度裁剪的最大范数；为 `None` 时不裁剪。
        log_steps (int): 训练进度条的日志步频（仅在 `show_progress=True` 时生效）。
        is_plot (bool): 是否在训练结束后绘制/保存 Loss 与 Accuracy 曲线。
        is_parameters_plot (bool): 是否在图注中附加模型与优化器参数信息。
        figure_note (Optional[str]): 额外的图注文字（可多行）。

    Returns:
        pd.DataFrame: 训练历史的 DataFrame，包含列：
            `epoch`, `train_loss`, `train_acc`, `valid_loss`, `valid_acc`。

    Raises:
        AssertionError: 当 `warmup_ratio` 不在 `[0, 1)` 范围内（且不为 `None`）时。
        ValueError: 由内部调用（如标签检查）在不支持的数据格式时抛出。

    Notes:
        - 最佳模型（按验证准确率优先、再比较验证损失）将保存为 `best_model.pth`。
        - 训练日志保存为 `train_log.csv`；若 `is_plot=True`，则在 `save_dir` 中保存曲线图。
    """
    # —— 参数校验 & 可复现性 —— 
    # 若 warmup_epochs 不为 None，则需满足 >=0；否则启用恒定学习率
    if warmup_epochs is not None:
        assert  warmup_epochs >=0 , "warmup_epochs must be >= 0"  # 预热比例校验
    if early_stopping_patience is not None:
        assert early_stopping_patience >= 1
    torch.manual_seed(random_state)  # 设置 PyTorch 随机种子
    torch.cuda.manual_seed_all(random_state)  # 设置所有 GPU 的随机种子
    np.random.seed(random_state)  # 设置 NumPy 随机种子
    torch.backends.cudnn.deterministic = True  # 确保 cuDNN 的确定性行为
    torch.backends.cudnn.benchmark = False  # 关闭基于输入形状的自动优化，增强复现性

    # —— 日志（请在脚本入口调用一次） —— 
    save_dir = Path(save_dir)  # 转为 Path 对象
    save_dir.mkdir(parents=True, exist_ok=True)  # 创建输出目录（包含父目录）

    # —— 模型 & 优化器 ——  
    model     = model_cls(**model_kwargs).to(device)  # 实例化模型并移动到设备
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)  # 实例化优化器

    # —— DataLoader ——  
    def _worker_init_fn(worker_id):
        np.random.seed(random_state + worker_id)  # 为各个 worker 设置独立随机种子
    dl_kwargs = dict(
        batch_size=batch_size,     # 批大小
        pin_memory=True,           # 固定内存，加速 GPU 传输
        num_workers=0,             # DataLoader 线程数；此处 0 以保证复现性
        worker_init_fn=_worker_init_fn,  # 初始化函数，设定 worker 的随机种子
    )
    train_dl = DataLoader(train_ds, shuffle=True, **dl_kwargs)   # 训练集 DataLoader（打乱）
    valid_dl = DataLoader(valid_ds, shuffle=False, **dl_kwargs)  # 验证集 DataLoader（不打乱）

    
    # —— Scheduler —— 
    if warmup_epochs is not None:
        scheduler = get_epoch_warmup_cosine_scheduler(optimizer=optimizer,
                                                      num_epochs=num_epochs, 
                                                      warmup_epochs=warmup_epochs)  # 预热+余弦调度
    else:
        scheduler = None # 恒定学习率，无调度  # 不使用调度器
        
        
    # —— 训练循环 ——  
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],   # 训练指标
        "valid_loss": [], "valid_acc": []                 # 验证指标
    }
    best_acc, no_improve = 0.0, 0  # 最优准确率与未提升计数器
    
    warnings.filterwarnings(
                            "ignore",
                            category=UserWarning,
                            module=r".*torch\.optim\.lr_scheduler"
                        )

    for epoch in range(1, num_epochs + 1):  # 遍历每个 epoch（1 开始）
    
        # # 3) 在每轮末做调度
        # optimizer.zero_grad(set_to_none=True)  # 确保没有残留梯度
        # optimizer.step()                       # 无梯度，不会更新参数（no-op）
        # if scheduler is not None:
        #     scheduler.step()  # epoch 级调度步进
    
        current_lr = optimizer.param_groups[0]['lr']  # 读取当前学习率（主参数组）
        logging.info(f"[Epoch {epoch}/{num_epochs}] lr = {current_lr:.3e}")  # 记录学习率

        # 1) 训练
        tr_loss, tr_acc = run_one_epoch(
            train_dl, model, loss_fn, device,
            mode="train", optimizer=optimizer,
            max_norm=max_norm, log_steps=log_steps,
            show_progress=True,
            mixup_alpha = mixup_alpha
        )  # 执行一个训练 epoch，返回训练损失与准确率

        # 2) 验证
        val_loss, val_acc, _, _ = run_one_epoch(
            valid_dl, model, loss_fn, device,
            mode="eval", optimizer=None,
            max_norm=None, log_steps=log_steps,
            show_progress=False
        )  # 执行一个验证 epoch，返回验证损失与准确率（以及标签与预测，未使用）
        
        # 3) 在每轮末做调度
        if scheduler is not None:
            scheduler.step()  # epoch 级调度步进


        # 4) 记录
        history["epoch"].append(epoch)         # 记录 epoch
        history["train_loss"].append(tr_loss)  # 记录训练损失
        history["train_acc"].append(tr_acc)    # 记录训练准确率
        history["valid_loss"].append(val_loss) # 记录验证损失
        history["valid_acc"].append(val_acc)   # 记录验证准确率

        # 5) 保存最优 & 早停
        if epoch == 1:
            # 第1个 epoch 后直接初始化 best，并保存
            best_acc, best_loss = val_acc, val_loss  # 初始化最佳指标
            no_improve = 0                           # 重置未提升计数
            tmp = save_dir / "best_model.pth.tmp"    # 临时文件路径
            torch.save(model.state_dict(), tmp)      # 保存权重到临时文件
            tmp.replace(save_dir / "best_model.pth") # 原子替换为正式文件
        else:
            is_better = False
            if best_by == 'acc':
                # 先看 acc（大优先），再以更小的 loss 打破平局
                if (val_acc > best_acc) or (val_acc == best_acc and val_loss < best_loss):
                    is_better = True
            else:  # best_by == 'loss'
                # 先看 loss（小优先），再以更大的 acc 打破平局
                if (val_loss < best_loss) or (val_loss == best_loss and val_acc > best_acc):
                    is_better = True

            if is_better:
                best_acc, best_loss = val_acc, val_loss
                no_improve = 0
                tmp = save_dir / "best_model.pth.tmp"
                torch.save(model.state_dict(), tmp)
                tmp.replace(save_dir / "best_model.pth")
            else:
                no_improve += 1
                if early_stopping_patience is not None and no_improve >= early_stopping_patience:
                    logging.info(f"Early stopping at epoch {epoch} (best_by={best_by})")
                    break

    # —— 导出 & 绘图 ——  
    hist_df = pd.DataFrame(history)  # 将历史记录转为 DataFrame
    hist_df.to_csv(save_dir / "train_log.csv", index=False)  # 保存训练日志为 CSV

    # 组装 note：参数说明（可选）+ figure_note（若提供）
    note_parts = []  # 存放注释片段
    if is_parameters_plot:
        note_parts.extend([
            f"Model: {model_cls.__name__}",         # 模型类名
            f"model_kwargs={model_kwargs}",         # 模型参数
            f"Optimizer: {optimizer_cls.__name__}", # 优化器类名
            f"optimizer_kwargs={optimizer_kwargs}", # 优化器参数
        ])
    if figure_note is not None:
        note_parts.append(str(figure_note))  # 追加外部注释文本
    
    note = "\n".join(note_parts)  # 用换行拼接注释文本，显示更清晰
    
    if is_plot:
        plot_acc_loss(
            epochs=hist_df["epoch"].values,            # x 轴：epoch
            train_loss=hist_df["train_loss"].values,   # 训练损失
            val_loss=hist_df["valid_loss"].values,     # 验证损失
            train_acc=hist_df["train_acc"].values,     # 训练准确率
            val_acc=hist_df["valid_acc"].values,       # 验证准确率
            save_dir=save_dir,                          # 输出目录
            note=note,                                  # 图注
            show=True, save=True                        # 显示并保存图像
        )

    logging.info(f"Training complete. Best valid acc = {best_acc:.1%}")  # 打印最终最佳验证准确率
    return hist_df  # 返回训练历史 DataFrame

def load_model(
    model_cls: Callable[..., torch.nn.Module],
    model_kwargs: Dict[str, Any],
    model_path: Union[str, Path],
    device: torch.device,
) -> torch.nn.Module:
    """
    加载深度学习模型的结构和权重，并将模型移动到指定的设备上（如 CPU 或 GPU）。

    该函数首先实例化模型，并从指定路径加载模型的权重。如果模型的权重文件是完整的 checkpoint，
    则会提取出其中的模型权重并加载到模型中。最后，将模型设置为评估模式。

    参数:
    - model_cls (Callable[..., torch.nn.Module]): 模型类（可调用的对象），用于创建模型实例。
    - model_kwargs (Dict[str, Any]): 用于初始化模型的参数（字典）。
    - model_path (Union[str, Path]): 保存模型权重的文件路径，可以是字符串或 Path 对象。
    - device (torch.device): PyTorch 设备对象（例如 'cuda' 或 'cpu'），指定加载模型的目标设备。

    返回:
    - model (torch.nn.Module): 加载并配置好的模型，已移动到指定设备，设置为评估模式。

    异常:
    - RuntimeError: 如果加载模型或权重过程中发生错误，会抛出该异常并提供详细的错误信息。
    """
    try:
        # 1) 实例化模型并移动到指定设备
        model = model_cls(**model_kwargs).to(device)

        # 2) 加载 checkpoint（模型权重）
        ckpt = torch.load(str(model_path), map_location=device, weights_only=True)

        # 3) 兼容“完整 checkpoint”保存格式
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']  # 从 checkpoint 中提取模型参数
        else:
            state_dict = ckpt  # 直接使用加载的对象作为 state_dict

        # 4) 加载模型权重
        model.load_state_dict(state_dict)

        # 5) 设置模型为评估模式（关闭 dropout 和 batch normalization 的训练行为）
        model.eval()

    except FileNotFoundError:
        raise RuntimeError(f"模型文件未找到：{model_path}")
    except KeyError as e:
        raise RuntimeError(f"加载模型权重时出错，缺少关键字段：{e}")
    except Exception as e:
        raise RuntimeError(f"加载模型失败，错误信息: {e}")

    return model

def EEG_test(
    test_ds: torch.utils.data.Dataset,
    model_cls: Callable[..., torch.nn.Module],
    model_kwargs: Dict[str, Any],
    model_path: Union[str, Path],
    loss_fn: torch.nn.Module,
    batch_size: int,
    device: torch.device,
    max_norm: Optional[float] = None,
    log_steps: int = 100
):
    """加载已保存的模型并在测试集上评估，返回真实标签和预测标签。

    该函数会：
    1. 实例化模型并加载指定路径的权重；
    2. 构建测试 DataLoader；
    3. 调用 `run_one_epoch` 在 `eval` 模式下运行测试集推理；
    4. 返回测试集所有样本的真实标签和预测标签。

    Args:
        test_ds (torch.utils.data.Dataset): 测试数据集，每个元素为 (X, y)。
        model_cls (Callable[..., torch.nn.Module]): 模型构造函数。
        model_kwargs (Dict[str, Any]): 实例化模型所需的关键字参数。
        model_path (Union[str, Path]): 模型权重文件路径（.pth）。
        loss_fn (torch.nn.Module): 损失函数，用于计算测试集 loss。
        batch_size (int): 测试批大小。
        device (torch.device): 运行设备（CPU 或 GPU）。
        max_norm (Optional[float], optional): 梯度裁剪阈值（测试时一般不使用）。默认为 None。
        log_steps (int, optional): 日志打印间隔（通常测试时用不到）。默认为 100。

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - y_true: 测试集真实标签，形状 (n_samples,)。
            - y_pred: 测试集预测标签，形状 (n_samples,)。
    """
    # 1) 加载模型结构与权重
    model = load_model(model_cls=model_cls,
                       model_kwargs=model_kwargs,
                       model_path=model_path,
                       device=device)

    # 2) DataLoader
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,  # 测试批大小
        shuffle=False,          # 测试集不打乱
        pin_memory=True,        # 固定内存以加速 GPU 数据传输
        num_workers=0           # 为保证可复现性，这里不启用多进程加载
    )

    # 3) 测试：run_one_epoch 内部会 no_grad + model.eval()
    _, _, y_true, y_pred = run_one_epoch(
        dataloader=test_dl,
        model=model,
        loss_fn=loss_fn,
        device=device,
        mode="eval",       # 评估模式
        optimizer=None,    # 无优化器（不反向传播）
        max_norm=max_norm, # 通常 None
        log_steps=log_steps
    )

    return y_true, y_pred  # 返回真实标签和预测标签



def EEG_train_validate_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_cls: Callable[..., torch.nn.Module],
    model_kwargs: dict,
    optimizer_cls: Callable[..., torch.optim.Optimizer],
    optimizer_kwargs: dict,
    loss_fn: torch.nn.Module,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
    save_dir: Union[str, Path],
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    warmup_epochs: Optional[int] = None,
    mixup_alpha: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    max_norm: Optional[float] = None,
    log_steps: int = 100,
    train_aug_transformer: Optional[TransformerMixin] = None,
    is_zscore_first: bool = True,
    is_plot: bool = True,
    is_parameters_plot: bool = True,
    figure_note: Optional[str] = None,
    best_by: Literal['acc', 'loss'] = 'acc',
) -> pd.DataFrame:
    """使用重复分层交叉验证 (Repeated Stratified CV) 训练并验证 EEG 模型。

    对输入的 EEG 数据 (X, y) 执行重复 K 折分层交叉验证：
    - 每折内可选先做 Z-score 标准化（基于该折训练集统计量）。
    - 可选地对训练集应用符合 scikit-learn Transformer 接口的数据增强。
    - 将每折数据包装为 PyTorch Dataset，调用 `EEG_train_validate` 执行训练与验证。
    - 记录每折的最佳验证性能及对应的模型路径。

    Args:
        X (np.ndarray): EEG 数据，形状 (n_samples, n_channels, n_times)。
        y (np.ndarray): 标签数组，形状 (n_samples,)。
        model_cls (Callable[..., torch.nn.Module]): 模型类构造函数。
        model_kwargs (dict): 模型初始化的关键字参数。
        optimizer_cls (Callable[..., torch.optim.Optimizer]): 优化器类构造函数。
        optimizer_kwargs (dict): 优化器初始化的关键字参数。
        loss_fn (torch.nn.Module): 损失函数实例。
        batch_size (int): DataLoader 的批大小。
        num_epochs (int): 每折的最大训练轮数。
        device (torch.device): 训练/验证使用的设备。
        save_dir (Union[str, Path]): 所有折结果保存的根目录。
        n_splits (int, optional): 每次重复的折数，默认 5。
        n_repeats (int, optional): 重复次数，默认 1。
        random_state (int, optional): 随机种子，默认 42。
        warmup_epochs (Optional[int]): 学习率 warmup 数量，None 表示不使用。
        mixup_alpha (Optional[float]): Mixup 数据增强的 α 参数，None 表示不使用。
        early_stopping_patience (Optional[int]): 早停耐心值，None 表示不早停。
        max_norm (Optional[float]): 梯度裁剪阈值，None 表示不裁剪。
        log_steps (int, optional): 训练日志的步频，默认 100。
        train_aug_transformer (Optional[TransformerMixin]): 训练集数据增强器（scikit-learn 风格）。
        is_zscore_first (bool, optional): 是否在增强前进行 Z-score，默认 True。
        is_plot (bool, optional): 是否绘制训练/验证曲线，默认 True。
        is_parameters_plot (bool, optional): 是否绘制参数分布，默认 True。
        figure_note (Optional[str]): 曲线图的额外注释。

    Returns:
        pd.DataFrame: 汇总 DataFrame，包含每折的最佳 epoch、验证准确率、验证损失及模型路径。
    """
    # 1. 创建根目录
    top_dir = Path(save_dir)  # 转换为 Path 对象
    top_dir.mkdir(parents=True, exist_ok=True)  # 创建保存目录（包含父目录）

    # 2. 构造重复分层交叉验证拆分器
    rkf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state
    )  # Repeated Stratified K-Fold 拆分器


    records = []  # 存储每折的结果记录
    # 3. 依次遍历每一次重复和每一折
    for rep_idx, (train_idx, valid_idx) in enumerate(rkf.split(X, y), start=1):
        rep = (rep_idx - 1) // n_splits + 1
        fold = (rep_idx - 1) %  n_splits + 1
    
        # 3.1 固定本折的随机种子，保证可复现
        seed = random_state + rep * 100 + fold
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 3.2 拆分训练/验证索引
        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_valid_raw, y_valid = X[valid_idx], y[valid_idx]

        # 3.3 折内 Z-score：只基于本折训练集计算均值和方差
        if is_zscore_first:
            scaler = ChannelZScoreScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(X_train_raw)
            X_valid = scaler.transform(X_valid_raw)
        else:
            X_train, X_valid = X_train_raw, X_valid_raw

        # 3.4 可选增强：仅对训练集调用 fit_transform
        if train_aug_transformer is not None:
            X_train, y_train = train_aug_transformer.fit_transform(
                X_train, y_train
            )

        # 3.5 构造 PyTorch Dataset
        train_ds = EEGDataset(X_train, y_train)
        valid_ds = EEGDataset(X_valid, y_valid)

        # 3.6 本折结果保存目录
        fold_dir = top_dir / f"repeat_{rep}" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        # 3.7 调用训练-验证主流程
        history_df = EEG_train_validate(
            train_ds=train_ds,
            valid_ds=valid_ds,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_epochs=num_epochs,
            device=device,
            save_dir=str(fold_dir),
            random_state=seed,
            warmup_epochs=warmup_epochs,
            mixup_alpha=mixup_alpha,
            early_stopping_patience=early_stopping_patience,
            max_norm=max_norm,
            log_steps=log_steps,
            is_plot=is_plot,
            is_parameters_plot=is_parameters_plot,
            figure_note=figure_note,
            best_by=best_by,
        )

        # 3.8 从历史记录中提取本折最佳指标
        if best_by == 'acc':
            sort_cols, ascending = ["valid_acc", "valid_loss"], [False, True]
        else:  # 'loss'
            sort_cols, ascending = ["valid_loss", "valid_acc"], [True, False]
        
        best_idx = history_df.sort_values(by=sort_cols, ascending=ascending, kind="mergesort").index[0]
        best_row = history_df.loc[best_idx]
        best_epoch  = int(best_row["epoch"])              # 最佳 epoch
        best_acc    = float(best_row["valid_acc"])        # 最佳验证准确率
        best_loss   = float(best_row["valid_loss"])       # 最佳验证损失
        model_path  = fold_dir / "best_model.pth"         # 最佳模型路径

        # 保存本折记录
        records.append({
            "repeat":          rep,
            "fold":            fold,
            "best_epoch":      best_epoch,
            "best_valid_acc":  best_acc,
            "best_valid_loss": best_loss,
            "model_path":      str(model_path),
        })

        # 3.9 清理 CUDA 缓存（若使用 GPU）
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # 4. 汇总所有折结果，保存 CSV
    summary_df  = pd.DataFrame(records)
    summary_csv = top_dir / "cv_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    # 5. 复制全局最优模型到根目录
    overall = summary_df.loc[summary_df["best_valid_acc"].idxmax()]
    shutil.copy(overall["model_path"], top_dir / "best_overall_model.pth")

    # 6. 打印总体最优折信息
    print(
        f">>> Overall best: rep {overall['repeat']}, fold {overall['fold']}, "
        f"acc {overall['best_valid_acc']:.1%}, "
        f"loss {overall['best_valid_loss']:.4f} @ epoch {overall['best_epoch']}.\n"
        f"Summary saved to {summary_csv}\n"
        f"Best model copied to {top_dir/'best_overall_model.pth'}"
    )

    return summary_df



def EEG_hyperparameter_grid_search_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_cls: Callable[..., torch.nn.Module],
    model_params: Dict[str, list],
    optimizer_cls: Callable[..., torch.optim.Optimizer],
    optimizer_params: Dict[str, list],
    loss_fn: torch.nn.Module,
    batch_size: int,
    num_epochs: int,
    device: torch.device,
    save_dir: Union[str, Path],
    save_every_model: bool = True,
    n_splits: int = 5,
    n_repeats: int = 1,
    random_state: int = 42,
    warmup_epochs: Optional[int] = None,
    mixup_alpha: Optional[float] = None,
    early_stopping_patience: Optional[int] = None,
    max_norm: Optional[float] = None,
    log_steps: int = 100,
    train_aug_transformer: Optional[TransformerMixin] = None,
    is_zscore_first: bool = True,
    is_plot: bool = True,
    is_parameters_plot: bool = True,
    figure_note: Optional[str] = None,
    show_progress: bool = False,
    best_by: Literal['acc', 'loss'] = 'acc',
) -> pd.DataFrame:
    """使用重复分层交叉验证对模型与优化器超参数执行网格搜索，并汇总表现。

    对 `model_params` 与 `optimizer_params` 的每一组组合，调用
    `EEG_train_validate_cv` 在同一数据集上完成重复分层交叉验证，
    统计该组合在各折的验证准确率/损失的均值与标准差，并可选保存该组合的最优模型。

    过程概览：
      1) 在 `save_dir/grid_search/exp_{i}` 下运行一次 `EEG_train_validate_cv`。
      2) 汇总该组合的 mean/std 验证指标。
      3) 可选：复制该组合的最优模型到 `grid_search/models/best_model_{i}.pth`。
      4) 汇总所有组合结果，挑选总体最佳并复制到 `grid_search/best_model.pth`。

    Args:
        X (np.ndarray): EEG 原始数据，形状为 (n_samples, n_channels, n_times)。
        y (np.ndarray): 分类标签，形状为 (n_samples,)。
        model_cls (Callable[..., torch.nn.Module]): 模型类（构造函数）。
        model_params (Dict[str, list]): 模型超参数搜索空间（每个键对应一个候选列表）。
        optimizer_cls (Callable[..., torch.optim.Optimizer]): 优化器类（构造函数）。
        optimizer_params (Dict[str, list]): 优化器超参数搜索空间（每个键对应一个候选列表）。
        loss_fn (torch.nn.Module): 损失函数实例（如 `nn.CrossEntropyLoss()`）。
        batch_size (int): DataLoader 批大小。
        num_epochs (int): 每折训练的最大 epoch 数。
        device (torch.device): 训练/验证所用设备（CPU/GPU）。
        save_dir (Union[str, Path]): 网格搜索的根目录；结果都保存在其下的 `grid_search/` 中。
        save_every_model (bool, optional): 是否将每个组合的最优模型另存到 `grid_search/models/`。默认为 True。
        n_splits (int, optional): 每次交叉验证的折数，默认为 5。
        n_repeats (int, optional): 交叉验证重复轮数，默认为 1。
        random_state (int, optional): 基础随机种子，默认为 42。
        warmup_epochs (Optional[int]): 传给 `EEG_train_validate` 的 warmup 数量；None 表示不使用。
        mixup_alpha (Optional[float]): 传给 `EEG_train_validate` 的 MixUp α；None 表示不使用。
        early_stopping_patience (Optional[int]): 早停耐心值；None 表示不早停。
        max_norm (Optional[float]): 梯度裁剪阈值；None 表示不裁剪。
        log_steps (int, optional): 训练日志步频，默认为 100。
        train_aug_transformer (Optional[TransformerMixin]): 可选的 scikit-learn 风格增强器（需实现 `fit_transform`）。
        is_zscore_first (bool, optional): 是否在每折前执行折内 Z-score，默认为 True。
        is_plot (bool, optional): 是否在 `EEG_train_validate` 中绘图，默认为 True。
        is_parameters_plot (bool, optional): 是否绘制参数信息，默认为 True。
        figure_note (Optional[str]): 图像中的额外注释。
        show_progress (bool, optional): 是否显示网格搜索进度条（tqdm），默认为 False。

    Returns:
        pd.DataFrame: 每个超参组合一行的结果表，包含：
            - model_params: dict，该组合的模型参数
            - opt_params: dict，该组合的优化器参数
            - mean_valid_acc / std_valid_acc: 验证准确率的均值/标准差
            - mean_valid_loss / std_valid_loss: 验证损失的均值/标准差
            - exp_dir: 该组合的结果目录
            - model_path: 该组合最终最优模型的文件路径
    """
    # 根目录： save_dir/grid_search
    top_dir = Path(save_dir) / "grid_search"  # 设定网格搜索的根目录
    top_dir.mkdir(parents=True, exist_ok=True)  # 创建目录（含父目录）

    # 如果需要保存所有模型，则准备 models 子目录
    if save_every_model:
        models_dir = top_dir / "models"  # 放置每个组合的最优模型
        models_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

    records = []  # 存放各组合统计结果的列表
    exp_idx = 0   # 实验编号（用于命名 exp_1, exp_2, ...）

    # 1. 遍历所有模型参数组合
    outer_iter = ParameterGrid(model_params)  # 外层网格：模型参数
    if show_progress:
        outer_iter = tqdm(outer_iter, desc="Model grid", leave=False)  # 可选进度条

    for model_kwargs in outer_iter:  # 逐个模型参数组合
        # 2. 遍历所有优化器参数组合
        inner_iter = ParameterGrid(optimizer_params)  # 内层网格：优化器参数
        if show_progress:
            inner_iter = tqdm(
                inner_iter,
                desc=f"Opt grid (model {exp_idx})",
                leave=False
            )  # 可选进度条（嵌套）

        for optimizer_kwargs in inner_iter:  # 逐个优化器参数组合
            exp_idx += 1  # 实验编号自增
            exp_dir = top_dir / f"exp_{exp_idx}"  # 本组合的结果目录
            exp_dir.mkdir(parents=True, exist_ok=True)  # 创建目录

            # 3. 调用已有的 CV 函数，评估此超参组合
            summary_df = EEG_train_validate_cv(
                X=X, y=y,
                model_cls=model_cls,
                model_kwargs=model_kwargs,
                optimizer_cls=optimizer_cls,
                optimizer_kwargs=optimizer_kwargs,
                loss_fn=loss_fn,
                batch_size=batch_size,
                num_epochs=num_epochs,
                device=device,
                save_dir=str(exp_dir),
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
                warmup_epochs=warmup_epochs,
                mixup_alpha=mixup_alpha,
                early_stopping_patience=early_stopping_patience,
                max_norm=max_norm,
                log_steps=log_steps,
                train_aug_transformer=train_aug_transformer,
                is_zscore_first=is_zscore_first,
                is_plot=is_plot,
                is_parameters_plot=is_parameters_plot,
                figure_note=figure_note,
                best_by=best_by,
            )  # 运行重复分层交叉验证，产出该组合的每折最佳指标

            # 4. 统计该组合在所有折上的平均和标准差
            mean_acc  = summary_df["best_valid_acc"].mean()  # 验证准确率均值
            std_acc   = summary_df["best_valid_acc"].std()   # 验证准确率标准差
            mean_loss = summary_df["best_valid_loss"].mean() # 验证损失均值
            std_loss  = summary_df["best_valid_loss"].std()  # 验证损失标准差

            # 5. 确定该组合的最优模型路径
            src_model = exp_dir / "best_overall_model.pth"  # 每个 exp 的总体最优模型
            if save_every_model:
                dst = models_dir / f"best_model_{exp_idx}.pth"  # 拷贝目标位置
                shutil.copy(src_model, dst)  # 复制该组合的最优模型
                model_path = str(dst)  # 记录保存后的路径
            else:
                model_path = str(src_model)  # 直接使用原路径

            # 6. 将结果记录到列表
            records.append({
                "model_params":     model_kwargs,   # 该组合的模型参数
                "opt_params":       optimizer_kwargs,  # 该组合的优化器参数
                "mean_valid_acc":   mean_acc,       # 均值准确率
                "std_valid_acc":    std_acc,        # 标准差准确率
                "mean_valid_loss":  mean_loss,      # 均值损失
                "std_valid_loss":   std_loss,       # 标准差损失
                "exp_dir":          str(exp_dir),   # 实验目录
                "model_path":       model_path      # 最优模型路径
            })

    # 7. 汇总所有超参组合的结果
    results_df = pd.DataFrame(records)  # 转为 DataFrame

    # 8. 复制全局最佳（最高 mean_valid_acc）到 grid_search/best_model.pth
    best_idx = results_df["mean_valid_acc"].idxmax()  # 选出均值准确率最高的组合
    best_row = results_df.loc[best_idx]               # 对应行
    best_dest = top_dir / "best_model.pth"            # 全局最佳模型的目标路径
    shutil.copy(best_row["model_path"], best_dest)    # 复制该最佳模型

    # 9. 打印最佳结果摘要
    bm = best_row["model_params"]  # 最优模型参数
    bo = best_row["opt_params"]    # 最优优化器参数
    param_str = ", ".join(
        [f"{k}={v}" for k, v in bm.items()] +
        [f"{k}={v}" for k, v in bo.items()]
    )  # 组装参数字符串，便于查看
    print(f">>> Best mean_valid_acc: {best_row['mean_valid_acc']:.2%}")  # 输出最佳均值准确率
    print(f"    Params: {{{param_str}}}")  # 输出对应参数
    print(f"    Saved to {best_dest}")     # 输出保存路径

    return results_df  # 返回汇总结果表


def get_checkpoint_data(X, model, checkpoint_name, is_backbone=False):
    """
    从指定模型的某个 checkpoint 获取其 forward 输出数据。

    该函数会将输入的 EEG 数据（4D）转换为 PyTorch 张量并传递给模型，
    然后，模型在指定 checkpoint 上注册一个 forward hook，该 hook 会捕获每次前向传播时的输出。
    最终返回 hook 捕获到的第一个输出数据。

    参数:
    - X (torch.Tensor): 输入的 EEG 数据张量，形状可以是 4D (n_samples, n_channels, n_times) 或 3D (n_samples, n_trials, n_channels, n_times)。
    - model (torch.nn.Module): 目标模型，该模型应具有 `backbone` 子模块。
    - checkpoint_name (str): 指定模型中 checkpoint 的名称，如 'ckp_1', 'ckp_2' 等。
    - is_backbone (bool, optional): 如果为 True，表示 checkpoint 位于模型的 `backbone` 子模块中；默认为 False，表示 checkpoint 位于模型的根模块。

    返回:
    - hook_output (np.ndarray): 从注册的 hook 捕获的第一个输出结果，已转换为 NumPy 数组。
    """

    # 用于存储 hook 捕获的输出
    hook_output_list = []

    def hook_fn(module, input, output):
        """
        该 hook 函数在模型的 forward 过程中被调用，并将输出的张量转换为 NumPy 数组后存储。
        
        参数:
        - module (torch.nn.Module): 被注册 hook 的模块
        - input (tuple): 输入到该模块的张量
        - output (torch.Tensor): 该模块的输出张量
        """
        # 将输出从计算图中分离出来，并转换为 NumPy 数组，存储到 hook_output_list 中
        hook_output_list.append(output.detach().cpu().numpy())

    try:
        # 根据 is_backbone 判断是否从 backbone 子模块中获取 checkpoint
        if is_backbone:
            checkpoint = getattr(model.backbone, checkpoint_name, None)
        else:
            checkpoint = getattr(model, checkpoint_name, None)

        # 如果没有找到指定的 checkpoint，抛出异常
        if checkpoint is None:
            raise AttributeError(f"Module '{checkpoint_name}' not found in model.")

        # 注册 forward hook 到指定 checkpoint
        hook = checkpoint.register_forward_hook(hook_fn)

        # 关闭梯度计算，进行推理
        with torch.no_grad():
            output = model(X)  # 进行一次前向传播，触发 hook

        # 返回 hook 捕获的第一个输出结果
        hook_output = hook_output_list[0]

    finally:
        # 确保在钩子注册后移除钩子，避免内存泄漏
        hook.remove()

    return hook_output

