# -*- coding: utf-8 -*-
"""
Created on Thu May  8 18:59:02 2025

@author: Fujie
"""
#%% 1. Set working directory

import os
from pathlib import Path
root = Path(r'C:\Users\vipuser\Documents')
os.chdir(root)
current_path = Path.cwd()
print("Current path:", current_path)

#%% 2. Import packages
# --- Python standard library ---

import json
import ast

# --- Third-party libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    cohen_kappa_score
)

import torch
from torch import optim

# --- Local modules ---
from DLtoolkit.utils import (
    save_csv,
    merge_subject_results,
    select_best_hyperparams,
    set_random_seed,
    load_data,
    EEGDataset,
    build_model_lookup
)
from DLtoolkit.DLtoolkits import (
    EEG_hyperparameter_grid_search_cv,
    EEG_test
)

from BCIC2020Track3_config import Config
from SE_MDTCN import (
    SE_MDTCN,
    SE_MDTCN_woElectrodeNormalization,
    SE_MDTCN_woSE_SpatialFilter,
    SE_MDTCN_woSE_DWTCNBlock,
    SE_MDTCN_woSE_DWTCNBlock2,
    SE_MDTCN_wZScoreNormalization,
    SE_MDTCN_wDWConvSpatialFilter2D,
    SE_MDTCN_wTemporalFilter2D,
)
#%% 3. Set preprocessing parameters

n_jobs = 10  # Set the number of parallel workers
random_state = 99  # Random seed
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

#%% 4. Configure deep learning

model_cls =  SE_MDTCN
optimizer_cls = optim.AdamW
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
batch_size = 64
num_epochs = 500 
warmup_epochs = 50
n_splits = 5
n_repeats = 1

branch_params = [
    {'n_layers': 4, 'kernel_size': 7, 'dropout': 0.25, 'n_branches': 1},
    {'n_layers': 4, 'kernel_size': 5, 'dropout': 0.25, 'n_branches': 1},
    {'n_layers': 4, 'kernel_size': 3, 'dropout': 0.25, 'n_branches': 1},
]


model_params = { 'input_channels': [1],
                 'input_electrodes': [64],
                 'num_classes': [5],
                 'input_times':[641],
                 'temporal_branch_params': [branch_params],
                 'spatial_filter_mode': ['SE'],
                 'spatial_filter_factor':[2],
                 'spatial_filter_factor_learnable':[False],
                 'fc_expand_dim':[4],
                 'fc_dropout':[0],
                 }

optimizer_params ={"lr": [1e-2],"weight_decay":[0]}

figure_note = f"{ica_type}; {timewin_name}; {BG_name};bz={batch_size}; warmup={warmup_epochs}"


# 设置保存路径
train_dir = root/ 'deep_learning' / model_cls.__name__ / f'{ica_type}_{timewin_name}_{BG_name}/train'
train_dir.mkdir(parents=True, exist_ok=True) 
test_dir = root/ 'deep_learning' / model_cls.__name__ /  f'{ica_type}_{timewin_name}_{BG_name}/test'
test_dir.mkdir(parents=True, exist_ok=True)  
        
#%% 5. Train the neural network with cross-validation: training set
      
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
    
merge_subject_results(base_dir = train_dir , 
                      out_csv = train_dir / "GSresults.csv",
                      pattern = "GSresults_sub*.csv",
                      drop_duplicates = True,)

#%% 6. Summarize hyperparameter performance and select the best values
# 1. Load results
train_result_csv = train_dir / "GSresults.csv"

# 2. Select the best values
best_model_params, best_opt_params = select_best_hyperparams(
    results_csv=train_result_csv,
    out_best_filename="best_params_GSresults.csv",
    out_ranked_filename="params_sorted_GSresults.csv",
)

# 3. Select the model corresponding to the best values
build_model_lookup(
    train_results=train_result_csv,
    best_params_csv=train_dir/"best_params_GSresults.csv",
    best_save_path=train_dir/"best models",
    sub_list=sub_list,          
    best_model_prefix="sub",
    order_by_col="mean_valid_acc", 
)
#%% 7. Test the trained neural network: test set

test_results = []
for sub_i in sub_list:

    model_path = train_dir / "best models" / f"sub{sub_i}.pth"

    if not model_path.exists():
        print(f"No matching hyperparameter combination found for sub_i={sub_i}, skipping.")
        continue

    X_test_set, y_test_true = load_data(test_data_dir, sub_i)
    X_valid_set, y_valid_true = load_data(valid_data_dir, sub_i)

    X_test = np.concatenate((X_test_set, X_valid_set), axis=0)
    y_true = np.concatenate((y_test_true, y_valid_true), axis=0)

    test_ds = EEGDataset(X_test, y_true)
    y_true, y_pred = EEG_test(
        test_ds=test_ds,
        model_cls=model_cls,
        model_kwargs=best_model_params,
        model_path=model_path,
        loss_fn=loss_fn,
        batch_size=64,
        device=device,
        max_norm=None,
        log_steps=100
    )

    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    # Print test results
    print(f'Sub_{sub_i} test_results: {balanced_acc}')

    # Aggregate test results
    new_row = {
        "sub_i": sub_i,
        "acc": float(balanced_acc),
        "f1_macro": float(f1_macro),
        "cohen_kappa": float(kappa),  # Avoid spaces in column names
        "confusion_matrix": json.dumps(cm.tolist()),  # list -> JSON for CSV-friendliness
        "model_params": best_model_params,  # dict for easy reuse in memory
        "opt_params": best_opt_params,      # dict
    }

    # Append the new row to the results list
    test_results.append(new_row)

test_results = pd.DataFrame(test_results)
test_results.to_csv(test_dir / "test_results.csv", index=False, encoding="utf-8")


#%% 8. Plot results: test set

# Set the directory for this level
plot_dir = test_dir / 'report_drawing'
plot_dir.mkdir(parents=True, exist_ok=True)

# Load test results
test_results = pd.read_csv(test_dir / 'test_results.csv')


# Plot per-subject performance bar chart under a single time-frequency window
test_df = test_results.copy()
test_df['acc'] = test_df['acc'] * 100

# Compute mean and standard deviation
mean_acc = test_df['acc'].mean()
std_acc = test_df['acc'].std()

# Prepare annotation text
textstr = f"Mean = {mean_acc:.2f}%\nSD   = {std_acc:.2f}%"

# Plot
fig, ax = plt.subplots()
plt.rcParams['font.family'] = 'Times New Roman'
sns.barplot(
    data=test_df, x='sub_i', y='acc', order=None,
    units=None, weights=None, orient=None,
    color=None, palette=None, saturation=0.75,
    fill=True, hue_norm=None, width=0.8, dodge='auto',
    gap=0, log_scale=None, native_scale=False,
    formatter=None, legend='auto', capsize=0.3,
    err_kws={'color': 'black', 'linewidth': 1}, ax=ax,
)
ax.text(
    0.95, 0.95, textstr,
    transform=ax.transAxes,
    ha='right', va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
)

# Modify legend
legend = ax.legend()
legend._loc = 2
legend.set_title('')

# Add chance-level line
ax.axhline(y=chance_level * 100, color="black", linewidth=0.5)
ax.text(
    x=14.5, y=chance_level * 100 + 1,
    s="Chance=" + str(chance_level * 100) + '%',
    color="black", ha='right'
)

# Axis formatting
ax.set_ylim(bottom=0, top=120)
ax.set_yticks(
    ticks=np.arange(0, 101, 10),
    labels=np.arange(0, 101, 10)
)
ax.set_xticks(
    ticks=sub_list,
    labels=sub_list,
)
ax.set_xlabel("Participant ID")
ax.set_ylabel("Accuracy (%)")
plt.show()

filename = 'group classification results'
fig.savefig(plot_dir / filename)
plt.close()


# 15.6 Plot the test-set confusion matrix
test_df = test_results.copy()

def find_keys_by_values(dict_obj, value_list):
    keys = []
    for value in value_list:
        for k, v in dict_obj.items():
            if v == value:
                keys.append(k)
    return keys


# ---- 1) Prepare labels (sorted by the values in event_id)
# Assume event_id: dict[str -> int], e.g., {"classA": 0, "classB": 1, ...}
mc_labels = find_keys_by_values(event_id, event_id.values())
num_conds = len(mc_labels)

# ---- 2) Accumulate confusion matrices across subjects
BG_CM = np.zeros((num_conds, num_conds), dtype=int)

# Standardize column name: you suggested using "confusion_matrix" when building test_results
col_cm = "confusion_matrix"
if col_cm not in test_df.columns:
    raise KeyError(f"Column {col_cm} does not exist. Current columns: {list(test_df.columns)}")

for cell in test_df[col_cm]:
    # cell may be a JSON string, list[list], or np.array
    if isinstance(cell, str):
        try:
            arr = np.array(json.loads(cell), dtype=int)
        except Exception:
            # Fallback: handle Python literal strings just in case
            import ast
            arr = np.array(ast.literal_eval(cell), dtype=int)
    else:
        arr = np.array(cell, dtype=int)

    if arr.shape != (num_conds, num_conds):
        raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {(num_conds, num_conds)}")
    BG_CM += arr

# ---- 3) Row-normalize to percentages
row_sums = BG_CM.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
norm_BG_CM = np.round(BG_CM / row_sums * 100, 0).astype(int)

# Use percentage strings as overlay text
percent_text = np.vectorize(lambda x: f"{x:d}%")(norm_BG_CM)

# ---- 4) Plot
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=norm_BG_CM, display_labels=mc_labels)
disp.plot(include_values=False, cmap=plt.cm.Blues, ax=ax, colorbar=False)

# Write percentages inside each cell
for i in range(norm_BG_CM.shape[0]):
    for j in range(norm_BG_CM.shape[1]):
        ax.text(
            j, i, percent_text[i, j],
            ha="center", va="center",
            color=("white" if i == j else "black"),
            fontweight="bold", fontsize=7
        )

# Cosmetics
ax.legend([], [], frameon=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="center", rotation_mode="anchor")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

plt.tight_layout()
plt.show()

# Save
filename = 'Group_Confusion_Matrix.png'  # Include an extension for clarity
fig.savefig(plot_dir / filename, dpi=300)
plt.close()





























    