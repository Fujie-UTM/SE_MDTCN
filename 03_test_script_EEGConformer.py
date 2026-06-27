# -*- coding: utf-8 -*-
"""Runs cross-validation, hyperparameter selection, and testing for EEGConformer

This script configures data paths, preprocessing settings, model variants,
training parameters, cross-validation, hyperparameter selection, and final test
evaluation for the EEGConformer model family.
"""
#%% 1. Set working directory

import os
from pathlib import Path
root = Path(r'C:\Users\vipuser\Documents')
os.chdir(root)
current_path = Path.cwd()
print("Current path:", current_path)

#%% 2. Import dependencies

import json
import numpy as np
import pandas as pd
import torch
from torch import optim

from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    cohen_kappa_score,
)

from mne.decoding import Scaler
from joblib import dump, load

# Import project utilities for data loading, training, result saving, and plotting.
from DLtoolkit.utils import (
    save_csv,
    merge_subject_results,
    select_best_hyperparams,
    set_random_seed,
    load_data,
    EEGDataset,
    build_model_lookup,
    event_labels,
    prepare_param_index_and_folders,
    save_test_results_by_param,
    save_subject_results_by_param,
    plot_one_param_results,
)

# Import project training and testing routines.
from DLtoolkit.DLtoolkits import (
    EEG_hyperparameter_grid_search_cv,
    EEG_test,
)

# Import experiment configuration.
from BCIC2020Track3_config import Config

# Import EEGNet model variants.
from models.EEGConformer_model import EEGConformer

#%% 3. Set preprocessing parameters

n_jobs = 10  # Number of parallel workers.
random_state = 99  # Random seed for reproducibility.
set_random_seed(random_state)

CONFIG=Config()

# Read dataset and preprocessing configuration values.
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

# Select whether to use raw or cleaned ICA-preprocessed data.
ica_type_list=['raw', 'clean']
ica_type="raw"

# Define data directories for training, validation, and test sets.
train_data_dir = root/ r"7_h5py/Training set/-500_2000/Gamma"/ica_type
valid_data_dir = root/ r"7_h5py/Validation set/-500_2000/Gamma"/ica_type 
test_data_dir = root/ r"7_h5py/Test set/-500_2000//Gamma"/ica_type 

#%% 4. Configure deep learning

# Select model, optimizer, loss function, device, and training schedule.
model_cls =  EEGConformer
optimizer_cls = optim.AdamW
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
batch_size = 64
num_epochs = 500 
warmup_epochs = 50
n_splits = 5
n_repeats = 1

# Enable preprocessing z-score only for the z-score normalization model variant.
if model_cls.__name__ == "EEGConformer":
    is_preprocess_zscore=True
else: 
    is_preprocess_zscore=False

# Set output directories for training results, test results, and preprocessing files.
train_dir = root/ 'DL_results' / model_cls.__name__ / f'{ica_type}_-500_2000_Gamma/train'
train_dir.mkdir(parents=True, exist_ok=True) 
test_dir = root/ 'DL_results' / model_cls.__name__ /  f'{ica_type}_-500_2000_Gamma/test'
test_dir.mkdir(parents=True, exist_ok=True)  
preprocess_dir = train_dir/"preprocess"
preprocess_dir.mkdir(parents=True, exist_ok=True)          


# Define model hyperparameter search space.
model_params = { 'input_channels': [1],
                 'input_electrodes': [64],
                 'input_times':[641],
                 'num_classes': [5],
                 'temp_conv_ks':[25], 
                 'temp_pool_ks':[75],
                 'k':[40],
                 'emb_dim':[40],
                 'encoder_depth':[6], 
                 'num_heads':[10],
                 'ffn_expansion':[4],     
                 'dropout_prepatch':[0.5],
                 'dropout_mha':[0.5],
                 'dropout_ffn':[0.5],              
                 'dropout_encoder':[0.5],
                 'fc_in_channels':[1480],
                 }
optimizer_params ={"lr": [1e-3, 1e-4],"weight_decay":[0, 1e-2]} #grid search


figure_note = f"{ica_type}; -500_2000; Gamma; bz={batch_size}; warmup={warmup_epochs}"



#%% 5. Train the neural network with cross-validation on the training set

train_results = pd.DataFrame()

for sub_i in sub_list:

    output_dir =train_dir / f'sub{sub_i}'
    subject_csv = output_dir / f"GSresults_sub{sub_i}.csv"

    X, y = load_data(train_data_dir , sub_i)

    # Fit and save a subject-specific z-score scaler when z-score preprocessing is enabled.
    if is_preprocess_zscore is True:
        zscore =Scaler(scalings='mean', with_mean=True, with_std=True)
        X = zscore.fit_transform(X)
        dump(zscore, preprocess_dir/f"sub_{sub_i}_zscore.pkl")

    # Run repeated cross-validation grid search for the current subject.
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

# Merge all subject-level grid-search result files into one summary CSV.
merge_subject_results(base_dir = train_dir , 
                      out_csv = train_dir / "GSresults.csv",
                      pattern = "GSresults_sub*.csv",
                      drop_duplicates = True,)

#%% 6. Summarize hyperparameter performance and select the best values

train_result_csv = train_dir / "GSresults.csv"

# Select the best model and optimizer hyperparameters according to the grid-search results.
best_model_params, best_opt_params = select_best_hyperparams(
    results_csv=train_result_csv,
    out_best_filename="best_params_GSresults.csv",
    out_ranked_filename="params_sorted_GSresults.csv",
)

# Keep the original best-model copy logic to make it easier to reproduce the
# test results corresponding to the best cross-validation hyperparameters.
build_model_lookup(
    train_results=train_result_csv,
    best_params_csv=train_dir / "best_params_GSresults.csv",
    best_save_path=train_dir / "best models",
    sub_list=sub_list,
    best_model_prefix="sub",
    order_by_col="mean_valid_acc",
)

#%% 7. Test all hyperparameter combinations on the test set



train_results_df, param_index_df, param_meta = prepare_param_index_and_folders(
    train_result_csv=train_result_csv,
    test_dir=test_dir,
    best_model_params=best_model_params,
    best_opt_params=best_opt_params,
    ranked_params_csv=train_dir / "params_sorted_GSresults.csv",
)

mc_labels, class_values = event_labels(event_id)


test_records = []

for sub_i in sub_list:
    sub_rows = train_results_df.loc[train_results_df["sub_i"] == sub_i].copy()

    if sub_rows.empty:
        print(f"No grid-search results found for sub_i={sub_i}, skipping.")
        continue

    # Load each subject's test and validation data once to avoid repeated I/O.
    X_test_set, y_test_true = load_data(test_data_dir, sub_i)
    X_valid_set, y_valid_true = load_data(valid_data_dir, sub_i)

    # Preserve the original setting: use validation data and test data together
    # as the final evaluation set.
    X_test = np.concatenate((X_test_set, X_valid_set), axis=0)
    y_true_full = np.concatenate((y_test_true, y_valid_true), axis=0)

    # Apply the saved subject-specific z-score scaler when preprocessing is enabled.
    if is_preprocess_zscore is True:
        zscore = load(preprocess_dir/f"sub_{sub_i}_zscore.pkl")
        X_test = zscore.transform(X_test)  

    test_ds = EEGDataset(X_test, y_true_full)

    for _, row in sub_rows.iterrows():
        model_path = Path(row["model_path"])

        if not model_path.exists():
            print(f"Model file not found for sub_i={sub_i}: {model_path}, skipping.")
            continue

        meta = param_meta.get(row["combo_key"])
        if meta is None:
            print(f"No param_id found for sub_i={sub_i}, combo={row['combo_key']}, skipping.")
            continue

        param_id = meta["param_id"]
        param_dir = Path(meta["param_dir"])
        model_kwargs = row["model_params_dict"]

        # Evaluate the saved model for the current subject and hyperparameter combination.
        y_true, y_pred = EEG_test(
            test_ds=test_ds,
            model_cls=model_cls,
            model_kwargs=model_kwargs,
            model_path=model_path,
            loss_fn=loss_fn,
            batch_size=batch_size,
            device=device,
            max_norm=None,
            log_steps=100,
        )

        # Compute test metrics and the confusion matrix.
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average="macro")
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=class_values)

        print(
            f"{param_id}, Sub_{sub_i}, "
            f"is_cv_best={bool(meta['is_cv_best_hyperparams'])}, "
            f"test_balanced_acc={balanced_acc:.4f}"
        )

        # Store one test record for the current subject and hyperparameter combination.
        record = {
            "param_id": param_id,
            "cv_rank": int(meta["cv_rank"]),
            "sub_i": sub_i,
            "acc": float(balanced_acc),
            "f1_macro": float(f1_macro),
            "cohen_kappa": float(kappa),
            "confusion_matrix": json.dumps(cm.tolist()),
            "model_params": row["model_params_key"],
            "opt_params": row["opt_params_key"],
            "is_cv_best_hyperparams": bool(meta["is_cv_best_hyperparams"]),
            "cv_mean_valid_acc": float(row["mean_valid_acc"]),
            "cv_std_valid_acc": float(row["std_valid_acc"]),
            "cv_mean_valid_loss": float(row["mean_valid_loss"]),
            "cv_std_valid_loss": float(row["std_valid_loss"]),
            "model_path": str(model_path),
            "exp_dir": row.get("exp_dir", ""),
            "param_dir": str(param_dir),
        }
        test_records.append(record)

test_results = pd.DataFrame(test_records)
test_results.to_csv(test_dir / "test_results_all_hyperparams_raw.csv", index=False, encoding="utf-8")

# Save per-parameter test results and summary files.
test_results, param_test_summary, param_index_out = save_test_results_by_param(
    test_records=test_records,
    test_dir=test_dir,
    param_index_df=param_index_df,
)

save_subject_results_by_param(
    test_records=test_records,
    test_dir=test_dir,
    param_index_df=param_index_df,
    )
#%% 8. Plot all hyperparameter test results

mc_labels, class_values = event_labels(event_id)
param_index_for_plot = pd.read_csv(test_dir / "param_index.csv")

for _, row in param_index_for_plot.sort_values("cv_rank", kind="mergesort").iterrows():
    param_id = row["param_id"]
    param_dir = Path(row["param_dir"]) if "param_dir" in row.index and pd.notna(row["param_dir"]) else test_dir / param_id
    param_test_csv = param_dir / "test_results.csv"

    if not param_test_csv.exists():
        print(f"{param_id}: {param_test_csv} not found, skip plotting.")
        continue

    # Plot and save evaluation figures for the current hyperparameter setting.
    param_test_df = pd.read_csv(param_test_csv)
    plot_one_param_results(
        param_test_df=param_test_df,
        param_dir=param_dir,
        param_id=param_id,
        mc_labels=mc_labels,
        chance_level=chance_level,
    )

