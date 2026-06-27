# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:59:02 2025

@author: Fujie
"""
#%% 1. 设置工作地址

import os
from pathlib import Path
root = Path(r'C:\Users\vipuser\Documents')
os.chdir(root)
current_path = Path.cwd()
print("当前路径为：", current_path)

#%% 2. 导入包
# —— Python 标准库 —— 
import sys
import io
import time
import logging
import json
import ast

# —— 第三方库 —— 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import mne
import h5py
import joblib
from joblib import parallel_backend
from PIL import Image

from scipy.stats import wilcoxon, binom
from statsmodels.stats.multitest import multipletests


from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score
)

import torch
from torch import optim
from torchinfo import summary

# —— 本地模块 —— 
from DLtoolkit_A5.utils import (
    save_csv,
    merge_subject_results,
    select_best_hyperparams,
    set_random_seed,
    load_data,
    EEGDataset,
    build_model_lookup)
from DLtoolkit_A5.DLtoolkits import (
    EEG_hyperparameter_grid_search_cv, 
    EEG_test)

from BCIC2020Track3_config import Config

from EEGNet_MOD import EEGNet
#%% 3. 设置预处理参数

n_jobs=10 #设置并行核数
random_state = 99 #随机数种子
set_random_seed(random_state) 

CONFIG=Config()

sub_list = CONFIG.sub_list
path_list = CONFIG.path_list
data_folder_list = CONFIG.data_folder_list
BG_name_list = CONFIG.BG_name_list
BG_list = CONFIG.BG_list
timewin_name_list = CONFIG.timewin_name_list
timewin_list = CONFIG.timewin_list
artifact_type = CONFIG.artifact_type
prob_critera_ic = CONFIG.prob_critera_ic
NF_freq_list = CONFIG.NF_freq_list
num_chs = CONFIG.num_chs
sfreq = CONFIG.sfreq
event_id=CONFIG.event_id
chance_level = 1.0/len(event_id)
num_conds = len(event_id)

ica_type_list=['raw', 'raw_ic', 'clean', 'clean_ic']
ica_type=ica_type_list[0]
timewin_name=timewin_name_list[-1]
BG_name=BG_name_list[-1]

train_data_dir = root/ path_list[7]/data_folder_list[0]/timewin_name/BG_name/ica_type
valid_data_dir = root/ path_list[7]/data_folder_list[1]/timewin_name/BG_name/ica_type 
test_data_dir = root/ path_list[7]/data_folder_list[2]/timewin_name/BG_name/ica_type 

#%% 4. 配置深度学习

model_cls =  EEGNet
optimizer_cls = optim.AdamW
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
batch_size = 64
num_epochs = 500 #这是深度学习的epoch，不是EEG的epoch
warmup_epochs = 50
n_splits = 5
n_repeats = 1


model_params = { 'input_channels': [1],
                 'input_electrodes': [64],
                 'num_classes': [5],
                 'input_times':[641],
                 'fs':[256],
                 'F1':[8],
                 'D':[2],
                 'F2':[16],
                 'fc_in_channels':[320],
                 'p_drop':[0.5],
                 'is_zscore':[True, False]
                 }

optimizer_params ={"lr": [1e-2, 1e-3]}

figure_note = f"{ica_type}; {timewin_name}; {BG_name};bz={batch_size}; warmup={warmup_epochs}"


# 设置保存路径
train_dir = root/ 'deep_learning' / model_cls.__name__ / f'{ica_type}_{timewin_name}_{BG_name}/train'
train_dir.mkdir(parents=True, exist_ok=True) 
test_dir = root/ 'deep_learning' / model_cls.__name__ /  f'{ica_type}_{timewin_name}_{BG_name}/test'
test_dir.mkdir(parents=True, exist_ok=True)  
        
#%% 6. 交叉验证训练神经网络：训练集
      
train_results = pd.DataFrame()

for sub_i in sub_list:
        
    output_dir =train_dir / f'sub{sub_i}'
    subject_csv = output_dir / f"GSresults_sub{sub_i}.csv"

    X, y = load_data(train_data_dir , sub_i)
    
    
    sub_result = EEG_hyperparameter_grid_search_cv( 
                    X = X, y = y, model_cls = model_cls,
                    model_params = model_params,
                    optimizer_cls = optimizer_cls,
                    optimizer_params = optimizer_params,
                    loss_fn = loss_fn,
                    batch_size = batch_size,
                    num_epochs = num_epochs,
                    device = device,
                    save_dir = output_dir,
                    save_every_model=True,
                    n_splits = n_splits,
                    n_repeats = n_repeats,
                    random_state = random_state,
                    warmup_epochs = warmup_epochs,
                    mixup_alpha = 0.5,
                    early_stopping_patience = None,
                    max_norm = None,
                    log_steps = 100,
                    train_aug_transformer = None,
                    is_zscore_first = False,
                    is_plot = True,
                    is_parameters_plot = True,
                    figure_note = figure_note,
                    ) 
    sub_result.insert(0, "sub_i", sub_i)

    save_csv(sub_result, subject_csv, overwrite=True)
    
# 任何时候都可以汇总
merge_subject_results(base_dir = train_dir , 
                      out_csv = train_dir / "GSresults.csv",
                      pattern = "GSresults_sub*.csv",
                      drop_duplicates = True,)

#%% 7. 统计超参数性能，选取最佳值
# 1. 读取
train_result_csv = train_dir / "GSresults.csv"

# 2. 选取最佳值
best_model_params, best_opt_params = select_best_hyperparams(results_csv=train_result_csv,
                                                             out_best_filename = "best_params_GSresults.csv",
                                                             out_ranked_filename = "params_sorted_GSresults.csv",)

# 3. 选取最佳值对应的模型
build_model_lookup(
    train_results=train_result_csv,
    best_params_csv=train_dir/"best_params_GSresults.csv",
    best_save_path=train_dir/"best models",
    sub_list=sub_list,          
    best_model_prefix="sub",
    order_by_col="mean_valid_acc", 
)
#%% 8. 测试训练神经网络：测试集

test_results = []
for sub_i in sub_list:
    
    model_path = train_dir/"best models"/f"sub{sub_i}.pth"
    
    if not model_path.exists():
        print(f"sub_i={sub_i} 没有匹配到对应的超参组合，跳过。")
        continue
        
    X_test_set, y_test_true = load_data(test_data_dir, sub_i)
    X_valid_set, y_valid_true = load_data(valid_data_dir, sub_i)
    
    X_test = np.concatenate((X_test_set, X_valid_set), axis=0)
    y_true = np.concatenate((y_test_true, y_valid_true), axis=0)
  
    test_ds = EEGDataset(X_test, y_true)
    y_true, y_pred = EEG_test(test_ds = test_ds,
                              model_cls = model_cls,
                              model_kwargs = best_model_params,
                              model_path = model_path,
                              loss_fn = loss_fn,
                              batch_size = 64,
                              device = device,
                              max_norm = None,
                              log_steps = 100)
        
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro=f1_score(y_true, y_pred, average='macro')
    kappa=cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred) 
    
    # 12.8 打印测试结果
    print(f'Sub_{sub_i} test_results:{balanced_acc}')   
   
    # 12.9 汇总测试结果
    new_row = {
        "sub_i": sub_i,
        "acc": float(balanced_acc),
        "f1_macro": float(f1_macro),
        "cohen_kappa": float(kappa),  # 列名建议不用空格
        "confusion_matrix": json.dumps(cm.tolist()),  # 列表→JSON，CSV 友好
        "model_params": best_model_params,  # dict，便于后续内存里直接用
        "opt_params": best_opt_params,  # dict
    }
    
    # 7. 将新行添加到结果列表中
    test_results.append(new_row)
    
test_results = pd.DataFrame(test_results)    
test_results.to_csv(test_dir / "test_results.csv", index=False, encoding="utf-8")
        
#%% 13. 定义一个无损保存tiff的函数

# # 将厘米转换为英寸的函数
# def cm_to_inch(value):
#     return value / 2.54

# # 设置全局的图形尺寸，以厘米为单位
# # https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample
# width_cm = 6
# height_cm = 6
# plt.rcParams['figure.figsize'] = (cm_to_inch(width_cm), cm_to_inch(height_cm))
# plt.rcParams['figure.autolayout']=True
# plt.rcParams['figure.constrained_layout.use']=False
# plt.rcParams['font.family'] = 'Times New Roman'
# plt.rcParams['font.size'] = 5
# plt.rcParams['axes.labelsize'] = 10  # x和y轴标签的字体大小
# plt.rcParams['axes.titlesize'] = 8  # 标题的字体大小
# plt.rcParams['xtick.labelsize'] = 6  # x轴刻度标签的字体大小
# plt.rcParams['ytick.labelsize'] = 6  # y轴刻度标签的字体大小
# plt.rcParams['legend.fontsize'] = 6  # 图例的字体大小
# plt.rcParams['figure.titlesize'] = 0  # 图形标题的字体大小
# plt.rcParams['legend.title_fontsize'] = 0  # 强制不显示图例标题   
# plt.rcParams['legend.fontsize'] = 6 
# plt.rcParams['legend.markerscale'] = 0.8
# plt.rcParams['legend.columnspacing'] = 0.5 
# plt.rcParams['legend.borderaxespad'] = 0.5
# plt.rcParams['legend.borderpad']=0
# plt.rcParams['legend.framealpha']=0
# plt.rcParams['legend.labelspacing']=0.1
# plt.rcParams['legend.handlelength']=1.0
# plt.rcParams['figure.dpi']=600
# plt.rcParams['savefig.dpi']=600
# plt.rcParams['savefig.format']='svg'
# plt.rcParams['savefig.bbox']='standard'    

#%%
#%% 15. 绘制结果_测试集部分

# 15.1 设置本级目录
plot_dir=test_dir / 'report_drawing'
plot_dir.mkdir(parents=True, exist_ok=True)  

# 15.2 读取测试结果
test_results=pd.read_csv(test_dir / 'test_results.csv')



# 15.4 画单一时频窗下各个被试的性能柱状图
test_df = test_results.copy()
test_df['acc'] = test_df['acc']*100

# 计算均值和标准差
mean_acc = test_df['acc'].mean()
std_acc  = test_df['acc'].std()

# 准备文字内容
textstr = f"Mean = {mean_acc:.2f}%\nSD   = {std_acc:.2f}%"
    
# 画图
fig, ax=plt.subplots()
plt.rcParams['font.family'] = 'Times New Roman'
sns.barplot(data=test_df, x='sub_i', y='acc', order=None,
            units=None, weights=None, orient=None, 
            color=None, palette=None, saturation=0.75, 
            fill=True, hue_norm=None, width=0.8, dodge='auto', 
            gap=0, log_scale=None, native_scale=False, 
            formatter=None, legend='auto', capsize=0.3, 
            err_kws={'color':'black', 'linewidth':1},  ax=ax,)
ax.text(
    0.95, 0.95, textstr,
    transform=ax.transAxes,
    ha='right', va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
)

# 图例修改
legend = ax.legend()
legend._loc=2
legend.set_title('')

# 添加机会概率线
ax.axhline(y=chance_level*100, color="black", linewidth=0.5)
ax.text(x=14.5, y=chance_level*100 + 1, s="Chance="+str(chance_level*100)+'%',
        color="black", ha='right')
# 坐标轴修改
ax.set_ylim(bottom=0, top=120)
ax.set_yticks(ticks=np.arange(0, 101, 10), 
              labels=np.arange(0, 101, 10))
ax.set_xticks(ticks=sub_list, 
              labels=sub_list,)
ax.set_xlabel("Participant ID", )
ax.set_ylabel("Accuracy (%)", )
plt.show()
filename = 'group classification results'
fig.savefig(plot_dir / filename)
plt.close()
        
    
# 15.6 画测试集的混淆矩阵
test_df = test_results.copy()

def find_keys_by_values(dict_obj, value_list):
    keys = []
    for value in value_list:
        for k, v in dict_obj.items():
            if v == value:
                keys.append(k)
    return keys


# ---- 1) 准备标签（按 event_id 的 value 排序）
# 假设 event_id: dict[str -> int]，例如 {"classA":0, "classB":1, ...}
mc_labels = find_keys_by_values(event_id, event_id.values())
num_conds = len(mc_labels)

# ---- 2) 累加每个被试的混淆矩阵
BG_CM = np.zeros((num_conds, num_conds), dtype=int)

# 统一列名：你在构建 test_results 时建议使用 "confusion_matrix"
col_cm = "confusion_matrix"  
if col_cm not in test_df.columns:
    raise KeyError(f"列 {col_cm} 不存在。当前列有：{list(test_df.columns)}")

for cell in test_df[col_cm]:
    # cell 可能是 JSON 字符串、list[list]、np.array
    if isinstance(cell, str):
        try:
            arr = np.array(json.loads(cell), dtype=int)
        except Exception:
            # 兼容万一是 Python 字面量字符串
            import ast
            arr = np.array(ast.literal_eval(cell), dtype=int)
    else:
        arr = np.array(cell, dtype=int)
    if arr.shape != (num_conds, num_conds):
        raise ValueError(f"混淆矩阵形状不匹配：得到 {arr.shape}，期望 {(num_conds, num_conds)}")
    BG_CM += arr

# ---- 3) 归一化（按行），转百分比
row_sums = BG_CM.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # 防 0
norm_BG_CM = np.round(BG_CM / row_sums * 100, 0).astype(int)

# 用字符串百分比作为覆盖文本
percent_text = np.vectorize(lambda x: f"{x:d}%")(norm_BG_CM)

# ---- 4) 绘图
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=norm_BG_CM, display_labels=mc_labels)
disp.plot(include_values=False, cmap=plt.cm.Blues, ax=ax, colorbar=False)

# 在格子里写百分比
for i in range(norm_BG_CM.shape[0]):
    for j in range(norm_BG_CM.shape[1]):
        ax.text(
            j, i, percent_text[i, j],
            ha="center", va="center",
            color=("white" if i == j else "black"),
            fontweight="bold", fontsize=7
        )

# 美化
ax.legend([], [], frameon=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="center", rotation_mode="anchor")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

plt.tight_layout()
plt.show()

# 保存
filename = 'Group_Confusion_Matrix.png'  # 加上后缀更明确
fig.savefig(plot_dir / filename, dpi=300)
plt.close()





























    