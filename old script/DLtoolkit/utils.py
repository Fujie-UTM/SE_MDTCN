# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 14:47:56 2025

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
from typing import Callable, Optional, Union, Tuple, Dict, Any, List, Iterable

# Third-party libraries
import h5py
import numpy as np
import pandas as pd
import torch



import torch.profiler

from torch.utils.data import Dataset
import matplotlib.pyplot as plt


from pathlib import Path
# Scikit-learn

from sklearn.base import TransformerMixin, BaseEstimator

import ast

import warnings
import shutil
#%%

def to_sorted_json(v):
    """把参数对象稳定化为 JSON 字符串（键排序）。"""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    if isinstance(v, str):
        # 兼容 JSON 字符串 / Python 字面量字符串
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(v)
                if isinstance(parsed, (dict, list)):
                    return json.dumps(parsed, sort_keys=True, ensure_ascii=False)
            except Exception:
                pass
    return str(v)

def save_csv(
    df: pd.DataFrame, 
    path: Path, 
    encoding: str = "utf-8", 
    overwrite: bool = True
):
    """
    最简单的 CSV 保存：确保目录存在 → 一次性写入。
    overwrite=False 时若已存在就不覆盖。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and path.exists():
        return
    df.to_csv(path, index=False, encoding=encoding)
    

def merge_subject_results(
    base_dir: str | Path,
    out_csv: str | Path,
    pattern: str = "GSresults_sub*.csv",
    drop_duplicates: bool = True,
):
    """
    汇总 deep_learning/<data_folder>/sub*/GSresults_sub*.csv
    - 若无匹配文件：抛错（或你改成返回空 DF）
    - 可选去重：按 (sub_i, model_params, opt_params) 去重
    """
    base = Path(base_dir)
    files = sorted(base.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {base}/**/{pattern}")

    frames = []
    for p in files:
        try:
            frames.append(pd.read_csv(p))
        except Exception as e:
            print(f"[WARN] skip {p}: {e}")

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if drop_duplicates and {"sub_i","model_params","opt_params"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["sub_i","model_params","opt_params"], keep="last")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False, encoding="utf-8")
    return merged


def _maybe_eval(x):
    """
    如果输入是字符串，并且字符串符合字典或列表的格式，则解析为字典或列表。
    """
    if isinstance(x, str) and x.strip().startswith(("{", "[")):
        try:
            # 如果是字符串，尝试将单引号替换为双引号并解析
            x = x.replace("'", '"')
            return ast.literal_eval(x)
        except Exception as e:
            print(f"Error evaluating: {x}\n{e}")
            return x
    return x

def select_best_hyperparams(
    results_csv: Path | str,
    out_best_filename: str = "best_params_GSresults.csv",
    out_ranked_filename: str = "params_sorted_GSresults.csv",
    model_col: str = "model_params",
    opt_col: str = "opt_params",
    mean_acc_col: str = "mean_valid_acc",
    std_acc_col: str = "std_valid_acc",
):
    """
    读取超参数搜索结果 CSV，对 (model_params, opt_params) 组合做中位数聚合，  
    按 (mean_valid_acc 降序, std_valid_acc 升序) 排序，选取最佳组合，  
    并将最佳与完整排序结果写回 CSV。

    返回：  
    ranked, best_model_params, best_opt_params, out_best_path, out_ranked_path
    """
    results_csv = Path(results_csv)
    save_dir = results_csv.parent

    # 1) 读取，并把字符串形式的 dict 安全还原
    df = pd.read_csv(
        results_csv, converters={ model_col: _maybe_eval, opt_col: _maybe_eval},
        )

    # 2) 将 dict 列转成稳定可 hash 的 JSON 字符串（保证分组一致）
    df[model_col] = df[model_col].apply(to_sorted_json)
    df[opt_col] = df[opt_col].apply(to_sorted_json)

    # 3) 聚合（中位数更稳健）
    ranked = (
        df.groupby([model_col, opt_col], as_index=False)
        .agg({mean_acc_col: "median", std_acc_col: "median"})
    )

    # 4) 排序：高准确率优先、低波动优先
    ranked = ranked.sort_values(
        by=[mean_acc_col, std_acc_col],
        ascending=[False, True],
        kind="mergesort",  # 稳定排序
    )

    # 5) 取最佳一行（仍保留 DataFrame）
    best_row = ranked.iloc[0]

    # 6) 反序列化回 dict
    best_model_params = json.loads(best_row[model_col])
    best_opt_params = json.loads(best_row[opt_col])

    # 7) 将 ranked 转换为字典表示
    ranked_dict = []
    for _, row in ranked.iterrows():
        ranked_dict.append({
            model_col: json.loads(row[model_col]),  # 解析 JSON 字符串为字典
            opt_col: json.loads(row[opt_col]),      # 解析 JSON 字符串为字典
            mean_acc_col: row[mean_acc_col],
            std_acc_col: row[std_acc_col]
        })

    # 8) 将 ranked_dict 转换为 DataFrame，然后保存为 CSV 文件
    ranked_df = pd.DataFrame(ranked_dict)


    # 将 best_payload 保存为 CSV（单行数据）
    best_payload = pd.DataFrame([{
        model_col: best_model_params,
        opt_col: best_opt_params,
        mean_acc_col: best_row[mean_acc_col],
        std_acc_col: best_row[std_acc_col],
    }])
    
    # 9) 保存结果
    out_best_path = save_dir / out_best_filename
    out_ranked_path = save_dir / out_ranked_filename

    best_payload.to_csv(out_best_path, index=False, encoding="utf-8")

    # 将 ranked_dict 保存为 CSV（多行数据）
    ranked_df.to_csv(out_ranked_path, index=False, encoding="utf-8")

    return best_model_params, best_opt_params


def _coerce_best(best_model_params, best_opt_params) -> tuple[str, str]:
    """
    将直接提供的 model_params 和 opt_params 转换为稳定的 JSON 字符串。

    参数：
    - best_model_params (dict): 模型超参数字典
    - best_opt_params (dict): 优化器超参数字典

    返回：
    - (best_mp_str, best_op_str): 对应的稳定 JSON 字符串
    """
    if not isinstance(best_model_params, dict) or not isinstance(best_opt_params, dict):
        raise TypeError("best_model_params 和 best_opt_params 必须是字典类型。")
    
    # 将字典转换为稳定的 JSON 字符串
    best_mp_str = to_sorted_json(best_model_params)
    best_op_str = to_sorted_json(best_opt_params)

    return best_mp_str, best_op_str

def load_best_params_from_csv(best_params_csv: str | Path, 
                              model_col: str="model_params", 
                              opt_col: str="opt_params"):
    """
    从 CSV 文件中加载最佳模型参数和优化器参数，并确保它们是字典类型。

    参数：
    - best_params_csv (str | Path): CSV 文件路径
    - model_col (str): CSV 文件中存储模型参数的列名
    - opt_col (str): CSV 文件中存储优化器参数的列名
    - _maybe_eval (function): 用于将字符串转换为字典的函数

    返回：
    - tuple: 返回一个包含最佳模型参数和优化器参数字典的元组
    """
    # 读取 CSV 文件
    best_params_df = pd.read_csv(best_params_csv)

    # 假设 CSV 文件中有 best_model_params 和 best_opt_params 列
    best_mp_str = best_params_df[model_col].iloc[0]
    best_op_str = best_params_df[opt_col].iloc[0]

    # 确保它们是字典类型，通过 _maybe_eval 转换
    best_model_params = _maybe_eval(best_mp_str)
    best_opt_params = _maybe_eval(best_op_str)

    if not isinstance(best_model_params, dict):
        raise TypeError(f"model_params 不是 dict，实际类型: {type(best_model_params)}")
    if not isinstance(best_opt_params, dict):
        raise TypeError(f"opt_params 不是 dict，实际类型: {type(best_opt_params)}")
    return best_model_params, best_opt_params


def build_model_lookup(
    train_results: str | Path | pd.DataFrame,
    best_save_path: str | Path = "saved_models",  # 新增 save_path 参数
    best_model_prefix: str = "sub",
    best_params_csv: Optional[str | Path] = None,  # 新增 best_params_csv 参数
    best_model_params: dict = None,  # 默认为 None，若为 None，则从 CSV 读取
    best_opt_params: dict = None,    # 默认为 None，若为 None，则从 CSV 读取
    sub_list: Optional[Iterable] = None,
    sub_col: str = "sub_i",
    model_col: str = "model_params",
    opt_col: str = "opt_params",
    path_col: str = "model_path",
    prefer_last: bool = True,
    order_by_col: Optional[str] = None,
):
    """
    基于最佳超参 (model_params,opt_params) 在 train_results 中为每个 subject 找到 model_path，
    并将找到的模型文件复制到指定文件夹，并按 sub_i 重命名，同时保存 mapping, chosen, missing 为文本文件。

    参数：
    - best_save_path: 指定保存模型文件的文件夹路径 (可以是相对路径或绝对路径)
    - best_params_csv: 用于指定包含最佳模型参数的 CSV 文件路径。如果为 None，函数会使用传入的 best_model_params 和 best_opt_params
    """

    # 1) 读取并标准化参数列为“稳定 JSON 字符串”
    if isinstance(train_results, (str, Path)):
        df = pd.read_csv(train_results, converters={model_col: _maybe_eval, opt_col: _maybe_eval})
    else:
        df = train_results.copy()

    if model_col not in df.columns or opt_col not in df.columns or sub_col not in df.columns:
        raise KeyError(f"train_results 需要包含列: {sub_col}, {model_col}, {opt_col}")

    if path_col not in df.columns:
        raise KeyError(f"train_results 缺少模型路径列: {path_col}")

    # 将模型参数和优化器参数转换为稳定的 JSON 字符串
    df[model_col] = df[model_col].apply(to_sorted_json)
    df[opt_col] = df[opt_col].apply(to_sorted_json)

    # 2) 如果 best_model_params 或 best_opt_params 为 None，从 CSV 文件中读取它们
    if best_model_params is None or best_opt_params is None:
        if best_params_csv is None:
            raise ValueError("best_model_params 和 best_opt_params 为空时，需要提供 best_params_csv 参数！")
        
        best_model_params, best_opt_params =load_best_params_from_csv(best_params_csv=best_params_csv,
                                                                      model_col=model_col, 
                                                                      opt_col=opt_col)

    # 通过 _coerce_best 将字典转化为 JSON 字符串
    best_mp_str, best_op_str = _coerce_best(best_model_params, best_opt_params)

    # 3) 过滤出与最佳超参一致的行
    m = (df[model_col] == best_mp_str) & (df[opt_col] == best_op_str)
    filtered = df.loc[m].copy()

    if filtered.empty:
        return

    # 4) 仅考虑 sub_list（如果提供）
    if sub_list is not None:
        sub_set = set(sub_list)
        filtered = filtered[filtered[sub_col].isin(sub_set)]
        target_subjects = list(sub_list)
    else:
        target_subjects = sorted(filtered[sub_col].unique().tolist())

    # 5) 在每个 subject 内挑选一条记录
    if order_by_col and order_by_col in filtered.columns:
        chosen = (
            filtered.sort_values([sub_col, order_by_col], ascending=[True, False])
                    .groupby(sub_col, as_index=False)
                    .head(1)
        )
    else:
        if prefer_last:
            chosen = filtered.groupby(sub_col, as_index=False).tail(1)
        else:
            chosen = filtered.groupby(sub_col, as_index=False).head(1)

    # 6) 复制模型文件到目标文件夹并重命名
    best_save_path = Path(best_save_path)
    best_save_path.mkdir(parents=True, exist_ok=True)  # 创建文件夹（如果不存在）

    mapping = {}
    for _, row in chosen.iterrows():
        sub_id = row[sub_col]
        model_path = row[path_col]

        # 确保 model_path 是有效的文件路径
        model_path = Path(model_path)
        if model_path.exists():
            # 创建目标路径，使用 sub_id 重命名
            target_path = best_save_path / f"{best_model_prefix}{sub_id}{model_path.suffix}"

            # 复制文件
            shutil.copy(model_path, target_path)

            # 更新映射
            mapping[sub_id] = str(target_path)
        else:
            print(f"Warning: Model file for subject {sub_id} not found at {model_path}")

    # 7) 保存 mapping, chosen, missing 为文本文件
    summary_file = best_save_path / "model_lookup_summary.txt"

    with open(summary_file, 'w') as f:
        # 写入 mapping
        f.write("Mapping (sub_id to model path):\n")
        for sub_id, model_path in mapping.items():
            f.write(f"{sub_id}: {model_path}\n")
        
        # 写入 chosen
        f.write("\nChosen Records:\n")
        for _, row in chosen.iterrows():
            f.write(f"sub_i: {row[sub_col]}, model_params: {row[model_col]}, opt_params: {row[opt_col]}, model_path: {row[path_col]}\n")
        
        # 写入 missing subjects
        missing = [s for s in target_subjects if s not in mapping]
        f.write("\nMissing Subjects:\n")
        for sub in missing:
            f.write(f"{sub}\n")

    print(f"Model lookup summary saved to {summary_file}")



def set_random_seed(seed: int, benchmark: bool = True) -> None:
    """设置全局随机种子。

    该函数会同时为 Python、NumPy 和 PyTorch（CPU & GPU）设置随机种子。
    在 GPU 可用时，会关闭 deterministic 模式并按需开启 cudnn.benchmark
    来加速固定尺寸输入的计算。

    Args:
        seed (int): 随机种子值。
        benchmark (bool): 是否启用 torch.backends.cudnn.benchmark
            加速（仅当输入尺寸固定时建议开启）。

    Returns:
        None
    """
    # —— Python 环境 ——
    os.environ["PYTHONHASHSEED"] = str(seed)  # 固定 Python 哈希种子，保证哈希结果可复现
    random.seed(seed)  # 设置 Python 内置随机数生成器的种子
    np.random.seed(seed)  # 设置 NumPy 随机数生成器的种子

    # —— PyTorch CPU & GPU ——
    torch.manual_seed(seed)  # 设置 PyTorch CPU 随机数种子
    if torch.cuda.is_available():  # 如果检测到可用 GPU
        torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设备设置相同的随机种子
        torch.use_deterministic_algorithms(False)  # 关闭确定性算法，允许使用最快算法
        torch.backends.cudnn.deterministic = False  # 禁用 cudnn 的确定性模式
        torch.backends.cudnn.benchmark = benchmark  # 按需开启 cudnn.benchmark 提升性能

    logging.info(f"随机种子已设置: seed={seed}, benchmark={benchmark}")  # 记录日志
    print(f"随机种子已设置: seed={seed}")  # 打印提示信息



def torch_seed_worker(worker_id):
    """为 PyTorch DataLoader 的 worker 设置随机种子。

    该函数会根据当前 PyTorch 的初始种子生成 worker 的随机种子，
    并将其应用到 NumPy 和 Python 内置的随机数生成器中。

    Args:
        worker_id (int): DataLoader 中 worker 的 ID。

    Returns:
        None
    """
    worker_seed = torch.initial_seed() % 2**32  # 生成 worker 级别的种子（32 位）
    np.random.seed(worker_seed)  # 设置 NumPy 随机数种子
    random.seed(worker_seed)  # 设置 Python 内置随机数种子
   
    

def load_data(train_dir, sub_i):
    """从 HDF5 文件加载数据。

    该函数会在指定目录中查找符合命名规则的 HDF5 文件，
    读取其中的 `data` 和 `label` 数据集，并返回为 NumPy 数组。

    Args:
        train_dir (str | Path): 存储数据文件的目录路径。
        sub_i (int): 子集编号，函数会查找以 "sub{sub_i}_" 开头的 `.h5` 或 `.H5` 文件。

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - X: 特征数据数组。
            - y: 标签数组。

    Raises:
        FileNotFoundError: 如果未找到符合条件的文件。
    """
    train_dir = Path(train_dir)  # 将目录路径转换为 Path 对象，方便路径操作

    # 遍历目录下所有文件，筛选符合前缀和后缀条件（大小写不敏感）
    candidates = [
        p for p in train_dir.iterdir()  # 遍历目录中的每个文件
        if p.is_file()  # 确保是文件（不是文件夹）
           and p.name.lower().startswith(f"sub{sub_i}_")  # 文件名前缀匹配
           and p.suffix.lower() == '.h5'  # 后缀为 .h5（不区分大小写）
    ]

    if not candidates:  # 如果没有匹配的文件
        raise FileNotFoundError(
            f"No file matching 'sub{sub_i}_*.h5' in {train_dir}"
        )

    if len(candidates) > 1:  # 如果匹配到多个文件
        # 这里默认取第一个，并给出提醒
        print(f"Warning: found multiple files for sub{sub_i}, using {candidates[0].name}")

    file_path = candidates[0]  # 选择第一个匹配的文件

    # 打开 HDF5 文件并读取数据
    with h5py.File(file_path, 'r') as f:
        X = f['data'][:]  # 读取数据集 'data' 为 NumPy 数组
        y = f['label'][:]  # 读取数据集 'label' 为 NumPy 数组

    return X, y  # 返回特征数据和标签


def pad_channel_dimension(
    data: np.ndarray,
    target_num_chs: int,
    pad_value: float = 0.0
) -> np.ndarray:
    """在通道维度上补齐数组到指定通道数。

    给定形状为 (N, Chs, T) 的三维数组，本函数会在第二维（通道维）末尾补齐，
    直到通道数达到 `target_num_chs`。若当前通道数已经大于等于目标通道数，
    则直接返回原数组。

    Args:
        data (numpy.ndarray): 输入数组，形状为 (n_samples, current_num_chs, n_timepoints)。
        target_num_chs (int): 目标通道数。
        pad_value (float, optional): 填充值，默认为 0.0。

    Returns:
        numpy.ndarray: 补齐后的数组，形状为
            (n_samples, max(current_num_chs, target_num_chs), n_timepoints)。
    """
    n_samples, current_num_chs, n_channels = data.shape  # 解包数组的维度
    pad_size = target_num_chs - current_num_chs  # 计算需要补的通道数（可能为负）

    if pad_size > 0:  # 只有当当前通道数小于目标通道数时才需要补齐
        # 在通道维（第二维）后面补 `pad_size` 个通道
        padded = np.pad(
            data,  # 原始数据
            pad_width=((0, 0), (0, pad_size), (0, 0)),  # 每个维度的 pad 宽度设置
            mode='constant',  # 常数模式补值
            constant_values=pad_value  # 填充值
        )
        return padded  # 返回补齐后的数组
    else:
        # 当前通道数已满足或超过目标通道数，直接返回原数组
        return data



class EEGDataset(Dataset):
    """优化后的 EEG 数据 PyTorch Dataset。

    该数据集类支持 EEG 数据的批量加载，并在初始化时将 NumPy 数组
    一次性转换为 PyTorch Tensor，从而减少训练过程中的数据转换开销。

    Attributes:
        data (torch.Tensor): EEG 数据张量。
        label (torch.Tensor): 对应的标签张量。
    """

    def __init__(self, EEGdata, EEGlabel):
        """初始化 EEG 数据集。

        Args:
            EEGdata (numpy.ndarray): EEG 数据，形状可以是
                - (n_samples, n_channels, n_times)  [3D]
                - (n_samples, n_trials, n_channels, n_times) [4D]
            EEGlabel (numpy.ndarray): EEG 标签，1D 或与数据样本数相匹配。

        """
        # 批量转换为 torch.Tensor，只做一次，减少训练时的转换开销
        if len(EEGdata.shape) == 4:  # 如果输入是 4D 数据
            self.data = torch.tensor(EEGdata, dtype=torch.float32)  # 直接转为 float32 张量
        if len(EEGdata.shape) == 3:  # 如果输入是 3D 数据
            # 扩展一个维度（作为通道维），再转为 float32 张量
            self.data = torch.tensor(np.expand_dims(EEGdata, axis=1), dtype=torch.float32)

        # 将标签转换为 long 类型张量（用于分类任务）
        self.label = torch.tensor(EEGlabel, dtype=torch.long)

        # 打印数据与标签的形状，方便调试
        print(f"EEG Dataset: data shape {self.data.shape}, label shape {self.label.shape}")

    def __len__(self):
        """返回数据集样本数量。

        Returns:
            int: 数据集中样本的数量。
        """
        return len(self.data)  # 样本数量等于 data 的第一维大小

    def __getitem__(self, idx):
        """获取指定索引的 EEG 样本及对应标签。

        Args:
            idx (int): 样本索引。

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (EEG 数据张量, 标签张量)。
        """
        return self.data[idx], self.label[idx]  # 返回指定样本及其标签
    
    def get_all_data(self):
        """返回所有的数据和标签."""
        return self.data, self.label



def plot_acc_loss(
    epochs: Sequence[float],
    train_loss: Sequence[float],
    val_loss: Sequence[float],
    train_acc: Sequence[float],
    val_acc: Sequence[float],
    save_dir: Union[str, Path],
    *,
    note: Optional[str] = None,               # 用 fig.text 绘制的注释
    figsize: Tuple[int, int] = (12, 5),
    dpi: int = 100,
    show: bool = True,
    save: bool = True,
    filename: str = "acc_loss.png",
    markers: Tuple[str, str] = ('.', 'o'),
    colors: Optional[Tuple[str, str]] = None,
    linestyles: Tuple[str, str] = ('-', '--'),
    grid: bool = True,
) -> Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]:
    """绘制训练/验证集的 Loss 与 Accuracy 曲线，并可在图中添加文本注释。

    本函数在同一画布上创建两个子图：
    左图显示训练与验证的 Loss 曲线，右图显示训练与验证的 Accuracy 曲线。
    可选地，在图中添加文字说明（自动换行），并选择是否保存或显示图片。

    Args:
        epochs (Sequence[float]): 训练的 epoch 序列。
        train_loss (Sequence[float]): 每个 epoch 对应的训练集 loss。
        val_loss (Sequence[float]): 每个 epoch 对应的验证集 loss。
        train_acc (Sequence[float]): 每个 epoch 对应的训练集准确率。
        val_acc (Sequence[float]): 每个 epoch 对应的验证集准确率。
        save_dir (Union[str, Path]): 保存图像的文件夹路径。
        note (Optional[str], optional): 要在图中显示的注释文字，支持换行。默认为 None。
        figsize (Tuple[int, int], optional): 图像大小（宽, 高），单位为英寸。默认为 (12, 5)。
        dpi (int, optional): 图像分辨率（每英寸点数）。默认为 100。
        show (bool, optional): 是否在屏幕上显示图像。默认为 True。
        save (bool, optional): 是否保存图像到文件。默认为 True。
        filename (str, optional): 保存的文件名（仅当 save=True 时有效）。默认为 "acc_loss.png"。
        markers (Tuple[str, str], optional): 两条曲线的点标记样式。默认为 ('.', 'o')。
        colors (Optional[Tuple[str, str]], optional): 两条曲线的颜色。若为 None，则使用 Matplotlib 默认颜色。
        linestyles (Tuple[str, str], optional): 两条曲线的线型。默认为 ('-', '--')。
        grid (bool, optional): 是否在子图中显示网格线。默认为 True。

    Returns:
        Tuple[plt.Figure, Tuple[plt.Axes, plt.Axes]]: 
            返回 Matplotlib Figure 对象和包含两个 Axes 对象的元组（左: Loss 子图, 右: Accuracy 子图）。

    Raises:
        AssertionError: 当输入序列长度不一致时抛出。
    """
    # ---- 输入长度校验 ----
    n = len(epochs)  # 获取 epochs 的长度
    assert len(train_loss) == n == len(val_loss) == len(train_acc) == len(val_acc), \
        "All input sequences must have the same length"  # 确保所有输入序列长度一致

    save_dir = Path(save_dir)  # 确保 save_dir 为 Path 对象
    if save and not save_dir.exists():  # 若需要保存且路径不存在
        save_dir.mkdir(parents=True, exist_ok=True)  # 创建文件夹（含父目录）

    # ---- 创建画布 ----
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)  # 创建 1 行 2 列子图

    # ---- Loss 子图 ----
    kw1 = {'marker': markers[0], 'linestyle': linestyles[0]}  # 训练曲线样式参数
    kw2 = {'marker': markers[1], 'linestyle': linestyles[1]}  # 验证曲线样式参数
    if colors:  # 如果用户指定了颜色
        kw1['color'], kw2['color'] = colors[0], colors[1]  # 设置颜色

    ax_loss.plot(epochs, train_loss, label="Train Loss", **kw1)  # 绘制训练 Loss 曲线
    ax_loss.plot(epochs, val_loss,   label="Val Loss",   **kw2)  # 绘制验证 Loss 曲线
    ax_loss.set(xlabel="Epoch", ylabel="Loss", title="Train vs. Val Loss")  # 设置坐标轴与标题
    ax_loss.legend()  # 显示图例
    if grid:  # 如果需要显示网格
        ax_loss.grid(True)

    # ---- Accuracy 子图 ----
    ax_acc.plot(epochs, train_acc, label="Train Acc", **kw1)  # 绘制训练 Accuracy 曲线
    ax_acc.plot(epochs, val_acc,   label="Val Acc",   **kw2)  # 绘制验证 Accuracy 曲线
    ax_acc.set(xlabel="Epoch", ylabel="Accuracy", title="Train vs. Val Accuracy")  # 设置坐标轴与标题
    ax_acc.legend()  # 显示图例
    if grid:  # 如果需要显示网格
        ax_acc.grid(True)

    # ---- 用 fig.text() 添加 note ----
    if note:  # 如果用户传入了注释文字
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，预留顶部空间
        fig.canvas.draw()  # 触发绘制，以便获取坐标信息
    
        axes_tops = [ax.get_position().y1 for ax in fig.axes]  # 获取所有子图的顶部位置（归一化坐标）
        top_axes_frac = max(axes_tops)  # 找到最高的子图顶部位置
    
        note_lines = (len(note) // 80) + 1  # 粗略估计注释的行数（假设每行 80 字符）
        line_spacing = 0.03  # 每行注释的垂直间距
    
        # 原计算
        y_pos = top_axes_frac + 0.01 + (note_lines - 1) * line_spacing  # 注释文本的 Y 坐标位置
    
        # 限制最大值，避免太高
        y_pos = min(y_pos, 0.98)  # 防止注释超出图像顶部
    
        fig.text(0.5, y_pos, note, ha='center', va='bottom', fontsize=10, wrap=True)  # 居中绘制注释
    else:
        fig.tight_layout()  # 如果没有注释，则直接自动调整布局

    # ---- 保存 & 显示 ----
    if save:  # 如果需要保存图像
        out_path = save_dir / filename  # 拼接保存路径
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')  # 保存图像，确保边界完整
        print(f"[INFO] Figure saved to {out_path}")  # 打印保存路径

    if show:  # 如果需要显示图像
        plt.show()
    else:  # 如果不显示，则关闭以释放内存
        plt.close(fig)

    return fig, (ax_loss, ax_acc)  # 返回 Figure 和两个 Axes



class ChannelZScoreScaler(TransformerMixin, BaseEstimator):
    """对多通道时间序列按通道做 Z-score 标准化。

    标准化公式为 (X - μ) / σ，其中 μ 和 σ 是在所有 epoch 和所有 time point 上计算的，
    每个通道分别计算自己的均值和标准差。

    Attributes:
        with_mean (bool): 是否执行去中心化（减去通道均值）。
        with_std (bool): 是否执行缩放（除以通道标准差）。
        mean_ (np.ndarray): 每个通道的均值，shape=(n_channels,)。
        scale_ (np.ndarray): 每个通道的标准差，shape=(n_channels,)。
    """

    def __init__(self, with_mean=True, with_std=True):
        """初始化 Z-score 标准化参数。

        Args:
            with_mean (bool, optional): 是否做去中心化（减去通道均值）。默认为 True。
            with_std (bool, optional): 是否做缩放（除以通道标准差）。默认为 True。
        """
        self.with_mean = with_mean  # 是否减去均值
        self.with_std = with_std    # 是否除以标准差

    def fit(self, X, y=None):
        """计算每个通道的均值 (μ) 和标准差 (σ)。

        Args:
            X (array-like): 输入数据，shape 为 (n_epochs, n_channels, n_times) 或 (n_epochs, n_channels)。
            y (ignored): 保留参数，为与 sklearn API 兼容。

        Returns:
            ChannelZScoreScaler: 返回自身。
        
        Raises:
            ValueError: 当输入数据既不是 2D 也不是 3D 时抛出。
        """
        X = np.asarray(X)  # 转为 NumPy 数组
        # 如果是二维数据，则当作每个通道只有一个 time point
        if X.ndim == 2:
            X = X[:, :, np.newaxis]  # 扩展出时间维度
        if X.ndim != 3:  # 检查维度
            raise ValueError(f"Scaler 只接受 2D 或 3D 输入，got shape {X.shape}")

        # axis=(0, 2) 表示跨 epoch 和 time 维度计算
        if self.with_mean:
            self.mean_ = X.mean(axis=(0, 2))  # 计算每个通道的均值
        else:
            self.mean_ = np.zeros(X.shape[1], dtype=X.dtype)  # 均值设为 0

        if self.with_std:
            self.scale_ = X.std(axis=(0, 2), ddof=0)  # 计算标准差（总体标准差）
            self.scale_[self.scale_ == 0.0] = 1.0     # 避免除以零
        else:
            self.scale_ = np.ones(X.shape[1], dtype=X.dtype)  # 标准差设为 1

        return self

    def transform(self, X):
        """对输入数据执行 Z-score 标准化。

        标准化公式：(X - μ) / σ

        Args:
            X (array-like): 输入数据，shape 为 (n_epochs, n_channels, n_times) 或 (n_epochs, n_channels)。

        Returns:
            np.ndarray: 标准化后的数据，与输入形状相同。
        
        Raises:
            ValueError: 当输入数据既不是 2D 也不是 3D 时抛出。
        """
        X = np.asarray(X)  # 转为 NumPy 数组
        squeeze = False  # 标记是否需要在最后去掉时间维度
        if X.ndim == 2:
            X = X[:, :, np.newaxis]  # 扩展时间维度
            squeeze = True
        if X.ndim != 3:
            raise ValueError(f"Scaler 只接受 2D 或 3D 输入，got shape {X.shape}")

        # 广播方式执行 (X - μ) / σ
        X_out = (X - self.mean_[None, :, None]) / self.scale_[None, :, None]

        if squeeze:
            X_out = X_out[:, :, 0]  # 去掉时间维度
        return X_out

    def inverse_transform(self, X):
        """将标准化后的数据反变换回原始尺度。

        反变换公式：X * σ + μ

        Args:
            X (array-like): 标准化后的数据，shape 为 (n_epochs, n_channels, n_times) 或 (n_epochs, n_channels)。

        Returns:
            np.ndarray: 反变换后的数据，与输入形状相同。
        
        Raises:
            ValueError: 当输入数据既不是 2D 也不是 3D 时抛出。
        """
        X = np.asarray(X)  # 转为 NumPy 数组
        squeeze = False  # 标记是否需要去掉时间维度
        if X.ndim == 2:
            X = X[:, :, np.newaxis]  # 扩展时间维度
            squeeze = True
        if X.ndim != 3:
            raise ValueError(f"Scaler 只接受 2D 或 3D 输入，got shape {X.shape}")

        # 广播方式执行反变换
        X_out = X * self.scale_[None, :, None] + self.mean_[None, :, None]
        if squeeze:
            X_out = X_out[:, :, 0]  # 去掉时间维度
        return X_out

    def fit_transform(self, X, y=None):
        """等同于先调用 fit 再调用 transform。

        Args:
            X (array-like): 输入数据。
            y (ignored): 保留参数，为与 sklearn API 兼容。

        Returns:
            np.ndarray: 标准化后的数据。
        """
        return self.fit(X, y).transform(X)  # 先 fit 再 transform

































