# -*- coding: utf-8 -*-
"""Generate statistical summaries and publication figures for EEG model results.

This script aggregates model comparison results, exports summary CSV files,
extracts checkpoint activations, computes t-SNE embeddings, and draws
classification, confusion matrix, t-SNE, and scalp topography figures.

Created on Mon Dec  8 00:49:38 2025

Author:
    Fujie
"""

import os
from pathlib import Path

root = Path(r'C:\Users\vipuser\Documents')
os.chdir(root)
current_path = Path.cwd()
print("当前路径为：", current_path)


import numpy as np
import h5py
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import ast
import json
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from scipy.stats import wilcoxon

from statsmodels.stats.multitest import multipletests


from DLtoolkit.utils import (
    load_best_params_from_csv,
    save_csv,
    merge_subject_results,
    select_best_hyperparams,
    set_random_seed,
    load_data,
    EEGDataset,
    build_model_lookup,
    to_sorted_json)
from DLtoolkit.DLtoolkits import (
    load_model,
    get_checkpoint_data
    
    )
from models.SE_MDTCN_model import SE_MDTCN
#%% 1. Model comparison

#%% 1.1. Set the working directory and parameters

# Create the output results directory path.
output_dir = root / 'output_results'  # Set the output directory path.
output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory without raising an error if it already exists.

# Define the list of folder names for different models.
folder_name_list = [
    "SE_MDTCN", 
    "SE_MDTCN_woElectrodeNormalization",  # Model without electrode normalization.
    "SE_MDTCN_woSpatialFilter",  # Model without the spatial filter.
    "SE_MDTCN_woTemporalFilter",  # Model without the spatial filter and DWTCN block.
    "SE_MDTCN_wZScoreNormalization",  # Model using Z-score normalization.
    "SE_MDTCN_wDWConvSpatialFilter2D",  # Model using a 2D DWConv spatial filter.
    "SE_MDTCN_wTemporalFilter2D"  # Model using a 2D temporal filter.
]

# Define paths for the training, validation, and test data.
train_data_dir = root / '7_h5py/Training set/-500_2000/Gamma/raw'  # Training data path.
valid_data_dir = root / '7_h5py/Validation set/-500_2000/Gamma/raw'  # Validation data path.
test_data_dir = root / '7_h5py/Test set/-500_2000/Gamma/raw'  # Test data path.

# Define the training output path.
train_dir = root / 'DL_results/SE_MDTCN/raw_-500_2000_Gamma/train'  # Training output path.

# Define and create the model checkpoint output directory.
ckpt_output_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/checkpoint_output'  # Checkpoint output path.
ckpt_output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory without raising an error if it already exists.

# Define and create the directory for spatial filter results.
spatial_filter_save_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/SpatialFilter'  # Spatial filter output path.
spatial_filter_save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory without raising an error if it already exists.

# Load the best model and optimizer parameters from a CSV file.
best_model_params, best_opt_params = load_best_params_from_csv(
    best_params_csv=train_dir / "best_params_GSresults.csv",  # CSV file path.
    model_col="model_params",  # Model parameter column.
    opt_col="opt_params"  # Optimizer parameter column.
)

# Define and create the t-SNE result directory.
tsne_save_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/TSNE'  # t-SNE result path.
tsne_save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory without raising an error if it already exists.


#%%

# Define the participant list with integers from 0 to 14 (15 participants total).
sub_list = list(range(15))

# Define the event label dictionary that maps event names to integer IDs.
event_id = {'Hello': 0, 'Help me': 1, 'Stop': 2, 'Thank you': 3, 'Yes': 4}

# Define a helper function to find dictionary keys for the given values.
def find_keys_by_values(dict_obj, value_list):
    """Find dictionary keys whose values match a list of target values.

    Args:
        dict_obj: Dictionary to search.
        value_list: Iterable of values to match against dictionary values.

    Returns:
        A list of keys whose corresponding values are present in ``value_list``.
    """
    keys = []  # Store the matched keys.
    for value in value_list:  # Iterate over all given values.
        for k, v in dict_obj.items():  # Iterate over the dictionary key-value pairs.
            if v == value:  # If the value matches.
                keys.append(k)  # Add the corresponding key to the result list.
    return keys

# Get all event labels.
mc_labels = find_keys_by_values(event_id, event_id.values())

# Count the number of event conditions.
num_conds = len(mc_labels)  # Number of event conditions.

# Electrode location dictionary that maps electrode IDs to electrode names.
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

# Create an electrode-name list by sorting the electrode dictionary keys.
electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]

#%% 1.2-1.3. Summarize best-parameter valid/test results from test_records outputs and run Wilcoxon automatically




# -----------------------------
# Configuration
# -----------------------------
DATA_TAG = "raw_-500_2000_Gamma"
CLEAN_DATA_TAG = "clean_-500_2000_Gamma"

# Valid-set kappa is approximated from balanced accuracy with chance level = 1 / num_classes.
# For this 5-class task, chance level is 0.2 and kappa ~= (acc - 0.2) / (1 - 0.2).
CHANCE_LEVEL = 1.0 / num_conds

# alternative="greater" tests whether the target model has larger acc than each compared model.
WILCOXON_ALTERNATIVE = "greater"
WILCOXON_ALPHA = 0.05

# Ablation: full SE_MDTCN is the target model, compared with its ablated variants.
ablation_model_configs = [
    {
        "model_name": folder_name_i,
        "result_dir": root / "DL_results" / folder_name_i / DATA_TAG,
    }
    for folder_name_i in folder_name_list
]

# Model comparison: compare SE_MDTCN raw/clean with other published/backbone models.
compared_model_name_list = [
    "EEGNet", "EEGTCNet", "EEGConformer", "HS_STDCN", "TSception_Full", "DGCNN"
]

model_comparison_model_configs = [
    {
        "model_name": "SE_MDTCN_raw",
        "result_dir": root / "DL_results" / "SE_MDTCN" / DATA_TAG,
    },
    {
        "model_name": "SE_MDTCN_clean",
        "result_dir": root / "DL_results" / "SE_MDTCN" / CLEAN_DATA_TAG,
    },
] + [
    {
        "model_name": compared_model_i,
        "result_dir": root / "DL_results" / compared_model_i / DATA_TAG,
    }
    for compared_model_i in compared_model_name_list
]

# Used by later plotting blocks.
all_model_name_list = [cfg["model_name"] for cfg in model_comparison_model_configs]


# -----------------------------
# Helper functions
# -----------------------------
def _find_first_existing_col(df, candidates, *, required=True, context=""):
    """Find the first column name that exists in df from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise KeyError(
            f"{context} cannot find any of these columns: {candidates}. "
            f"Current columns: {list(df.columns)}"
        )
    return None


def _as_bool_series(s):
    """Convert common CSV boolean representations to a bool Series."""
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(int).astype(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def _kappa_from_accuracy(acc, chance_level=CHANCE_LEVEL):
    """Approximate Cohen's kappa from accuracy under a fixed chance level."""
    return (np.asarray(acc, dtype=float) - chance_level) / (1.0 - chance_level)


def _read_confusion_matrix_cell(cell):
    """Parse a confusion-matrix cell saved as JSON, Python literal, list, or ndarray."""
    if cell is None:
        return None
    if isinstance(cell, float) and np.isnan(cell):
        return None
    if isinstance(cell, str):
        text = cell.strip()
        if text == "" or text.lower() == "nan":
            return None
        for loader in (json.loads, ast.literal_eval):
            try:
                return np.array(loader(text), dtype=int)
            except Exception:
                pass
        raise ValueError(f"Cannot parse confusion_matrix cell: {cell}")
    return np.array(cell, dtype=int)


def _confusion_matrix_to_json(cell, expected_n_classes):
    """Return a standardized JSON string for one subject-level confusion matrix."""
    arr = _read_confusion_matrix_cell(cell)
    if arr is None:
        return ""
    expected_shape = (expected_n_classes, expected_n_classes)
    if arr.shape != expected_shape:
        raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {expected_shape}")
    return json.dumps(arr.tolist(), ensure_ascii=False)


def _load_best_param_test_table(model_name, result_dir):
    """Load subject rows under the CV-best hyperparameter from test_records-derived files only.

    Preferred source:
        test/test_results_best_params.csv

    Fallbacks:
        test/test_results_all_hyperparams.csv filtered by is_cv_best_hyperparams
        test/test_results_all_hyperparams_raw.csv filtered by is_cv_best_hyperparams
        test/param_XXX/test_results.csv, where XXX is read from cv_best_param_id.txt

    This function intentionally does not read train/GSresults.csv.
    """
    result_dir = Path(result_dir)
    test_dir_i = result_dir / "test"

    preferred = test_dir_i / "test_results_best_params.csv"
    if preferred.exists():
        df = pd.read_csv(preferred).copy()
        source_file = preferred
    else:
        df = None
        source_file = None

        for candidate in [
            test_dir_i / "test_results_all_hyperparams.csv",
            test_dir_i / "test_results_all_hyperparams_raw.csv",
        ]:
            if candidate.exists():
                all_df = pd.read_csv(candidate).copy()
                if "is_cv_best_hyperparams" in all_df.columns:
                    mask = _as_bool_series(all_df["is_cv_best_hyperparams"])
                    df = all_df.loc[mask].copy()
                    source_file = candidate
                    break

        if df is None:
            best_id_file = test_dir_i / "cv_best_param_id.txt"
            if best_id_file.exists():
                best_param_id = best_id_file.read_text(encoding="utf-8").strip()
                candidate = test_dir_i / best_param_id / "test_results.csv"
                if candidate.exists():
                    df = pd.read_csv(candidate).copy()
                    source_file = candidate

        if df is None:
            raise FileNotFoundError(
                f"Cannot locate test_records-derived best-parameter results for {model_name}. "
                f"Checked under: {test_dir_i}"
            )

    if df.empty:
        raise ValueError(f"Empty best-parameter test table for {model_name}: {source_file}")

    if "sub_i" not in df.columns:
        raise KeyError(f"{source_file} has no 'sub_i' column. Current columns: {list(df.columns)}")

    if "model_name" in df.columns:
        df["model_name"] = model_name
    else:
        df.insert(0, "model_name", model_name)
    df["source_file"] = str(source_file)
    df = df.drop_duplicates("sub_i", keep="last").sort_values("sub_i", kind="mergesort")
    return df


def _split_valid_and_test_from_best_table(experiment_name, model_name, result_dir):
    """Create subject-level valid/test tables from one model's best-parameter test output."""
    best_df = _load_best_param_test_table(model_name, result_dir)

    valid_acc_col = _find_first_existing_col(
        best_df,
        ["cv_mean_valid_acc", "mean_valid_acc", "cv_valid_bacc_mean", "valid_bacc_mean", "valid_acc"],
        context=f"{model_name} valid acc",
    )
    test_acc_col = _find_first_existing_col(
        best_df,
        ["acc", "test_acc", "test_bacc", "balanced_accuracy", "bacc"],
        context=f"{model_name} test acc",
    )
    test_kappa_col = _find_first_existing_col(
        best_df,
        ["cohen_kappa", "test_kappa", "kappa"],
        required=False,
        context=f"{model_name} test kappa",
    )
    f1_col = _find_first_existing_col(
        best_df,
        ["f1_macro", "test_f1_macro", "macro_f1"],
        required=False,
        context=f"{model_name} test macro F1",
    )
    cm_col = _find_first_existing_col(
        best_df,
        ["confusion_matrix", "test_confusion_matrix"],
        required=False,
        context=f"{model_name} confusion matrix",
    )

    valid_acc = pd.to_numeric(best_df[valid_acc_col], errors="coerce").astype(float)
    valid_subject = pd.DataFrame({
        "experiment": experiment_name,
        "split": "valid",
        "model_name": model_name,
        "sub_i": best_df["sub_i"].values,
        "acc": valid_acc.values,
        "cohen_kappa": _kappa_from_accuracy(valid_acc.values, CHANCE_LEVEL),
        "kappa_source": f"computed_from_acc_with_chance_level_{CHANCE_LEVEL:.6g}",
        "metric_source": valid_acc_col,
        "source_file": best_df["source_file"].values,
    })

    test_acc = pd.to_numeric(best_df[test_acc_col], errors="coerce").astype(float)
    if test_kappa_col is not None:
        test_kappa = pd.to_numeric(best_df[test_kappa_col], errors="coerce").astype(float)
        test_kappa_source = test_kappa_col
    else:
        test_kappa = pd.Series(_kappa_from_accuracy(test_acc.values, CHANCE_LEVEL), index=best_df.index, dtype=float)
        test_kappa_source = f"computed_from_acc_with_chance_level_{CHANCE_LEVEL:.6g}"

    test_subject = pd.DataFrame({
        "experiment": experiment_name,
        "split": "test",
        "model_name": model_name,
        "sub_i": best_df["sub_i"].values,
        "acc": test_acc.values,
        "cohen_kappa": test_kappa.values,
        "kappa_source": test_kappa_source,
        "f1_macro": pd.to_numeric(best_df[f1_col], errors="coerce").astype(float).values if f1_col is not None else np.nan,
        "confusion_matrix": best_df[cm_col].apply(lambda x: _confusion_matrix_to_json(x, num_conds)).values if cm_col is not None else "",
        "metric_source": test_acc_col,
        "source_file": best_df["source_file"].values,
    })

    return valid_subject, test_subject


def _make_subject_acc_wide(metric_df):
    """Create a subject × model wide accuracy table for paired Wilcoxon tests."""
    wide = (
        metric_df
        .pivot_table(index="sub_i", columns="model_name", values="acc", aggfunc="last")
        .reset_index()
        .sort_values("sub_i", kind="mergesort")
    )
    wide.columns.name = None
    return wide


def _summarize_subject_results(experiment_name, split_name, subject_df):
    """Summarize acc_mean, acc_std, and kappa_mean for one experiment/split."""
    summary = (
        subject_df.groupby("model_name", as_index=False)
        .agg(
            n_subjects=("sub_i", "nunique"),
            acc_mean=("acc", "mean"),
            acc_std=("acc", "std"),
            kappa_mean=("cohen_kappa", "mean"),
        )
    )

    if split_name == "valid":
        # Required behavior: derive valid kappa_mean from acc_mean, rather than from a saved CV kappa column.
        summary["kappa_mean"] = _kappa_from_accuracy(summary["acc_mean"].values, CHANCE_LEVEL)
        summary["kappa_source"] = f"computed_from_acc_mean_with_chance_level_{CHANCE_LEVEL:.6g}"
    else:
        summary["kappa_source"] = "cohen_kappa_from_test_records"

    summary.insert(0, "split", split_name)
    summary.insert(0, "experiment", experiment_name)
    return summary


def _aggregate_group_confusion_matrix(test_subject_df, expected_n_classes):
    """Aggregate subject-level confusion matrices for each model."""
    rows = []
    for model_name, model_df in test_subject_df.groupby("model_name", sort=False):
        group_cm = np.zeros((expected_n_classes, expected_n_classes), dtype=int)
        has_any_cm = False
        for cell in model_df["confusion_matrix"]:
            arr = _read_confusion_matrix_cell(cell)
            if arr is None:
                continue
            if arr.shape != (expected_n_classes, expected_n_classes):
                raise ValueError(
                    f"{model_name}: confusion matrix shape mismatch: got {arr.shape}, "
                    f"expected {(expected_n_classes, expected_n_classes)}"
                )
            group_cm += arr
            has_any_cm = True

        rows.append({
            "model_name": model_name,
            "confusion_matrix": json.dumps(group_cm.tolist(), ensure_ascii=False) if has_any_cm else "",
        })
    return pd.DataFrame(rows)


def _bonferroni_correction(p_values, alpha=0.05):
    """Bonferroni correction with a statsmodels fallback."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    if multipletests is not None:
        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method="bonferroni")
        return reject, p_corrected

    p_corrected = np.minimum(p_values * len(p_values), 1.0)
    reject = p_corrected < alpha
    return reject, p_corrected


def _run_wilcoxon_from_wide_acc(wide_df, target_model, split_name, alternative="greater", alpha=0.05):
    """Run paired Wilcoxon signed-rank tests on subject-level accuracy.

    With alternative="greater", this tests whether target_model has larger
    accuracy than each compared model, i.e. target - compared > 0.
    """
    if target_model not in wide_df.columns:
        raise ValueError(
            f"Target model '{target_model}' is not in the subject-level table. "
            f"Available columns: {list(wide_df.columns)}"
        )

    model_cols = [c for c in wide_df.columns if c != "sub_i"]
    rows = []

    for compared_model in model_cols:
        if compared_model == target_model:
            continue

        paired = wide_df[["sub_i", target_model, compared_model]].dropna().copy()
        n_pairs = len(paired)

        if n_pairs == 0:
            stat, p_value = np.nan, np.nan
        else:
            diff = paired[target_model] - paired[compared_model]
            if np.allclose(diff.values, 0.0):
                # scipy.wilcoxon raises for all-zero differences with zero_method="wilcox".
                stat, p_value = 0.0, 1.0
            else:
                stat, p_value = wilcoxon(
                    paired[target_model],
                    paired[compared_model],
                    alternative=alternative,
                    zero_method="wilcox",
                    method="auto",
                )

        rows.append({
            "split": split_name,
            "target_model": target_model,
            "compared_model": compared_model,
            "alternative": alternative,
            "n_pairs": int(n_pairs),
            "mean_target": float(paired[target_model].mean()) if n_pairs else np.nan,
            "mean_compared": float(paired[compared_model].mean()) if n_pairs else np.nan,
            "mean_diff_target_minus_compared": float((paired[target_model] - paired[compared_model]).mean()) if n_pairs else np.nan,
            "W_stat": float(stat) if pd.notna(stat) else np.nan,
            "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        })

    res = pd.DataFrame(rows)
    if not res.empty:
        valid_p = res["p_value"].notna()
        res["p_bonferroni"] = np.nan
        res["reject_bonf_0.05"] = False
        if valid_p.any():
            reject, p_bonf = _bonferroni_correction(res.loc[valid_p, "p_value"].values, alpha=alpha)
            res.loc[valid_p, "p_bonferroni"] = p_bonf
            res.loc[valid_p, "reject_bonf_0.05"] = reject
        res = res.sort_values(["split", "p_value"], ascending=[True, True], kind="mergesort").reset_index(drop=True)
    return res


def _aggregate_one_experiment(experiment_name, model_configs, target_model):
    """Aggregate valid/test subject results, write CSVs, and run Wilcoxon tests."""
    valid_parts = []
    test_parts = []

    for cfg in model_configs:
        valid_i, test_i = _split_valid_and_test_from_best_table(
            experiment_name=experiment_name,
            model_name=cfg["model_name"],
            result_dir=cfg["result_dir"],
        )
        valid_parts.append(valid_i)
        test_parts.append(test_i)

    valid_subject = pd.concat(valid_parts, axis=0, ignore_index=True)
    test_subject = pd.concat(test_parts, axis=0, ignore_index=True)

    # The two main subject-level CSV files for each experiment.
    # test_results_by_subject includes confusion_matrix for later plotting.
    valid_subject.to_csv(output_dir / f"{experiment_name}_best_param_valid_results_by_subject.csv", index=False, encoding="utf-8")
    test_subject.to_csv(output_dir / f"{experiment_name}_best_param_test_results_by_subject.csv", index=False, encoding="utf-8")

    valid_summary = _summarize_subject_results(experiment_name, "valid", valid_subject)
    test_summary = _summarize_subject_results(experiment_name, "test", test_subject)

    # Add group-level confusion matrices to the test summary for direct confusion-matrix plotting if needed.
    group_cm = _aggregate_group_confusion_matrix(test_subject, expected_n_classes=num_conds)
    test_summary = test_summary.merge(group_cm, on="model_name", how="left")
    valid_summary["confusion_matrix"] = ""

    summary = pd.concat([valid_summary, test_summary], axis=0, ignore_index=True)
    summary.to_csv(output_dir / f"{experiment_name}_summary_stats.csv", index=False, encoding="utf-8")

    valid_wide = _make_subject_acc_wide(valid_subject)
    test_wide = _make_subject_acc_wide(test_subject)

    valid_wilcoxon = _run_wilcoxon_from_wide_acc(
        valid_wide,
        target_model=target_model,
        split_name="valid",
        alternative=WILCOXON_ALTERNATIVE,
        alpha=WILCOXON_ALPHA,
    )
    test_wilcoxon = _run_wilcoxon_from_wide_acc(
        test_wide,
        target_model=target_model,
        split_name="test",
        alternative=WILCOXON_ALTERNATIVE,
        alpha=WILCOXON_ALPHA,
    )
    wilcoxon_results = pd.concat([valid_wilcoxon, test_wilcoxon], axis=0, ignore_index=True)
    wilcoxon_results.insert(0, "experiment", experiment_name)
    wilcoxon_results.to_csv(output_dir / f"{experiment_name}_wilcoxon_acc.csv", index=False, encoding="utf-8")

    return {
        "valid_subject": valid_subject,
        "test_subject": test_subject,
        "summary": summary,
        "wilcoxon": wilcoxon_results,
    }


# -----------------------------
# Run aggregation for both experiments
# -----------------------------
ablation_outputs = _aggregate_one_experiment(
    experiment_name="ablation",
    model_configs=ablation_model_configs,
    target_model="SE_MDTCN",
)

model_comparison_outputs = _aggregate_one_experiment(
    experiment_name="model_comparison",
    model_configs=model_comparison_model_configs,
    target_model="SE_MDTCN_raw",
)

combined_summary_stats = pd.concat(
    [ablation_outputs["summary"], model_comparison_outputs["summary"]],
    axis=0,
    ignore_index=True,
)
combined_summary_stats.to_csv(output_dir / "all_experiments_summary_stats.csv", index=False, encoding="utf-8")

combined_wilcoxon_acc = pd.concat(
    [ablation_outputs["wilcoxon"], model_comparison_outputs["wilcoxon"]],
    axis=0,
    ignore_index=True,
)
combined_wilcoxon_acc.to_csv(output_dir / "all_experiments_wilcoxon_acc.csv", index=False, encoding="utf-8")

# Convenience variables for later blocks in this script.
ablation_valid_results_by_subject = ablation_outputs["valid_subject"].copy()
ablation_test_results_by_subject = ablation_outputs["test_subject"].copy()
model_comparison_valid_results_by_subject = model_comparison_outputs["valid_subject"].copy()
model_comparison_test_results_by_subject = model_comparison_outputs["test_subject"].copy()
all_model_df = model_comparison_test_results_by_subject.copy()

print("Saved best-parameter subject-level valid/test results, summary statistics, and Wilcoxon results to:", output_dir)


#%% 3. Extract model outputs at each checkpoint for the training-set model

# Define checkpoint names for different model checkpoints.
check_point_name_list = ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3', 'ckpt_4', 'classifier']

# Use itertools.product to generate all participant-checkpoint combinations.
for sub_i, ckpt_name in itertools.product(sub_list, check_point_name_list):
    
    # Load training, validation, and test data.
    X_train, y_train = load_data(train_data_dir, sub_i)
    X_valid, y_valid = load_data(valid_data_dir, sub_i)
    X_test, y_test = load_data(test_data_dir, sub_i)

    # Concatenate training, validation, and test data.
    X = np.concatenate((X_train, X_valid, X_test), axis=0)
    y = np.concatenate((y_train, y_valid, y_test), axis=0)
    
    # Create an EEGDataset instance with the loaded data X and labels y.
    eeg_dataset = EEGDataset(EEGdata=X, EEGlabel=y)
    
    # Get all data in tensor format.
    X_tensor, _ = eeg_dataset.get_all_data()

    # Set the model path, assuming each participant's best model is saved in the "best models" directory.
    model_path = train_dir / "best models" / f"sub{sub_i}.pth"

    # Load the pretrained model with `load_model` using the model class, best parameters, and path.
    model = load_model(
        model_cls=SE_MDTCN,  # Model class used.
        model_kwargs=best_model_params,  # Best model parameters.
        model_path=model_path,  # Model path.
        device='cpu'  # Use the CPU device for inference.
    )

    # Set the current checkpoint name.
    check_point_name = f"{ckpt_name}"
    
    # Get the current checkpoint data, assuming `get_checkpoint_data` processes model data.
    # `is_backbone=False` may indicate that the model backbone is not used.
    if ckpt_name == 'raw':
        ckpt_data = X  # If the checkpoint is "raw", use the raw data directly.
    else:
        ckpt_data = get_checkpoint_data(X_tensor, model, check_point_name, is_backbone=False)

    # Create an HDF5 file to save data, using one folder per participant.
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist.
    
    # Set the HDF5 file name with the checkpoint name.
    filename = f'{ckpt_name}.h5'
    
    # Open the HDF5 file for writing.
    with h5py.File(h5_dir / filename, 'w') as f:
        # Save the processed data to the HDF5 file with gzip compression.
        f.create_dataset('data', data=ckpt_data, dtype='float32', compression="gzip", compression_opts=9)
        f.create_dataset('label', data=y, dtype='int32', compression="gzip", compression_opts=9)


#%% 3. tsne participant 7

sub_i = 7  # Set the current participant ID to 7.

# Create a folder to save t-SNE images, with the participant ID in the path.
sub_tsne_dir = tsne_save_dir / f"sub{sub_i}"
sub_tsne_dir.mkdir(parents=True, exist_ok=True)  # Create the folder if it does not exist.

# Initialize data scalers.
normal_1 = StandardScaler()  # StandardScaler shifts data to zero mean and unit variance.
normal_2 = MinMaxScaler(feature_range=(-1, 1))  # MinMaxScaler scales data to the interval [-1, 1].

# Initialize a t-SNE instance with the specified parameters.
tsne = TSNE(n_components=2, random_state=99, method='exact', 
            init='pca', max_iter=10000, n_iter_without_progress=500,
            n_jobs=8, perplexity=50)  # Use PCA initialization, 10000 maximum iterations, and n_jobs=8 for parallel computation on 8 CPU cores.

# Iterate over each checkpoint name.
for ckpt_name in check_point_name_list:
    
    # Create a participant folder to save this participant's t-SNE data.
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the file name and read the data.
    filename = f'{ckpt_name}.h5'
    X = []
    with h5py.File(h5_dir / filename, 'r') as f:
        X = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
    
    # Compute variance for selected checkpoint data; axis=-1 indicates the last dimension (time).
    if ckpt_name in ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3']:
        X = np.var(X, axis=-1)  # Compute variance for raw data and selected checkpoint data.
    else:
        X = X  # Do not apply variance processing to other checkpoints.

    # Standardize the data.
    X_norm = normal_1.fit_transform(X)  # Standardize the data with normal_1.
    X_tsne = tsne.fit_transform(X_norm)  # Use t-SNE to reduce the data to two dimensions.
    X_tsne_norm = normal_2.fit_transform(X_tsne)  # Further scale the two-dimensional t-SNE output with MinMaxScaler.

    # Set the new file name for saving t-SNE data.
    filename = f'{ckpt_name}.h5'
    
    # Open the HDF5 file for writing.
    with h5py.File(sub_tsne_dir / filename, 'w') as f:
        # Save the processed t-SNE data to the HDF5 file with gzip compression.
        f.create_dataset('data', data=X_tsne_norm, dtype='float32', compression="gzip", compression_opts=9)
        f.create_dataset('label', data=y, dtype='int32', compression="gzip", compression_opts=9)

#%%

# Convert centimeters to inches.
def cm_to_inch(value):
    """Convert centimeters to inches.

    Args:
        value: Length in centimeters.

    Returns:
        The corresponding length in inches.
    """
    return value / 2.54  # 1 inch equals 2.54 centimeters; return the value in inches.

# Set global figure properties with sizes specified in centimeters.
# https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample

plt.rcParams['figure.autolayout'] = True  # Enable automatic layout to adjust subplot positions and sizes.
plt.rcParams['figure.constrained_layout.use'] = False  # Disable constrained layout so that figure layout is not forced.
plt.rcParams['font.family'] = 'Arial'  # Set the global font to Arial.
plt.rcParams['font.size'] = 9  # Set the global font size to 8.
plt.rcParams['axes.labelsize'] = 8  # Set the x- and y-axis label font size to 8.
plt.rcParams['axes.titlesize'] = 8  # Set the figure title font size to 8.
plt.rcParams['xtick.labelsize'] = 6  # Set the x-axis tick-label font size to 6.
plt.rcParams['ytick.labelsize'] = 6  # Set the y-axis tick-label font size to 6.
plt.rcParams['xtick.direction'] = 'in'  # Set x-axis ticks to point inward.
plt.rcParams['ytick.direction'] = 'in'  # Set y-axis ticks to point inward.
plt.rcParams['legend.fontsize'] = 6  # Set the legend font size to 6.
plt.rcParams['figure.titlesize'] = 0  # Disable figure titles.

# Set legend styles in rcParams.
plt.rcParams['legend.title_fontsize'] = 0  # Disable the legend title.
plt.rcParams['legend.fontsize'] = 8  # Set the legend font size to 6.
plt.rcParams['legend.markerscale'] = 0.8  # Set the legend marker-size scale to 0.8.
plt.rcParams['legend.columnspacing'] = 0.5  # Set the spacing between legend columns to 0.5.
plt.rcParams['legend.borderaxespad'] = 0.5  # Set the padding between the legend and axes to 0.5.
plt.rcParams['legend.borderpad'] = 0  # Set the inner padding of the legend frame to 0.
plt.rcParams['legend.framealpha'] = 0  # Set the legend alpha to 0 so that the legend frame is invisible.
plt.rcParams['legend.labelspacing'] = 0.1  # Set the vertical spacing between legend labels to 0.1.
plt.rcParams['legend.handlelength'] = 1.0  # Set the legend handle length to 1.0.
plt.rcParams['legend.loc'] = 'upper right'  # Legend location.

plt.rcParams['legend.handletextpad'] = 0.5  # Adjust the horizontal distance between markers and text.
plt.rcParams['legend.frameon'] = True  # Show the legend frame.
plt.rcParams['legend.facecolor'] = 'lightgray'  # Set the legend background color to light gray.

# Set figure resolution (DPI).
plt.rcParams['figure.dpi'] = 600  # Set the figure resolution to 600 DPI.
plt.rcParams['savefig.dpi'] = 600  # Set the saved-figure resolution to 600 DPI.
plt.rcParams['savefig.format'] = 'tiff'  # Set the saved-figure format to TIFF.
plt.rcParams['savefig.bbox'] = 'standard'  # Set the saved-figure bounding box to standard.

# Set the color list for plotting.
colors = ['red', 'orange', 'green', 'blue', 'purple']

#%% 1.4 Plot test-set performance

# 15.1 Set the current-level directory and create a folder for saving figures.
output_plot_dir = output_dir / 'report_drawing'
output_plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist.

# 15.2 Read model test results.
all_model_df = pd.read_csv(output_dir / 'model_comparison_best_param_test_results_by_subject.csv')  # Read subject-level test results for model comparison.

# Iterate over all model names.
for model_i in all_model_name_list:
    
    # 1. Filter the corresponding test data by model name.
    test_df = all_model_df[all_model_df['model_name'] == model_i].copy()

    # Convert accuracy to percentages.
    test_df['acc'] = test_df['acc'] * 100
    
    # Compute the mean and standard deviation of accuracy.
    mean_acc = test_df['acc'].mean()
    std_acc = test_df['acc'].std()
    
    # Draw a bar plot of accuracy for each model.
    fig, ax = plt.subplots(figsize=(cm_to_inch(7), cm_to_inch(5.5)))  # Create the figure and axes.
    sns.barplot(data=test_df, x='sub_i', y='acc', ax=ax,  # Draw the bar plot with Seaborn.
                saturation=0.75, width=0.8, err_kws={'color': 'black', 'linewidth': 1})
    
    # Prepare the displayed text containing the mean and standard deviation of accuracy.
    textstr = f"Mean = {mean_acc:.1f}%\nSD = {std_acc:.1f}%"
    ax.text(
        0.95, 0.95, textstr,  # Place the text in the upper-right corner of the axes.
        fontsize=6,  # Set the font size to 6.
        transform=ax.transAxes,  # Use relative axes coordinates.
        ha='right', va='top',  # Right-align and top-align the text.
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),  # Set the background box.
    )
    
    # Add the chance-level line (20%).
    ax.axhline(y=20, color="firebrick", linewidth=0.6, linestyle='--')  # The red horizontal line indicates chance level.
    # ax.text(x=14.5, y=21, s="chance level", fontsize=6, color="red", ha='right', fontweight='bold')
    
    # Adjust the axes.
    ax.set_ylim(bottom=0, top=120)  # Set the y-axis display range.
    ax.set_yticks(ticks=np.arange(0, 101, 20), labels=np.arange(0, 101, 20))  # Set the y-axis ticks.
    ax.set_xticks(ticks=sub_list, labels=sub_list)  # Set the x-axis ticks.
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)  # Hide the top ticks.
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)  # Show the left ticks.
    ax.set_xlabel("Participant ID")  # Set the x-axis label.
    ax.set_ylabel("Accuracy (%)")  # Set the y-axis label.
    
    plt.show()  # Show the figure.
    
    # Save the figure as a TIFF file.
    filename = f'{model_i}_group_classification_results.tiff'
    fig.savefig(output_plot_dir / filename)
    plt.close()  # Close the figure.
    
    
    
    # 2. Draw the confusion matrix.
    group_CM = np.zeros((num_conds, num_conds), dtype=int)  # Initialize the confusion matrix.
    
    # Standardize the column name by assuming "confusion_matrix" is used.
    col_cm = "confusion_matrix"
    if col_cm not in test_df.columns:
        raise KeyError(f"列 {col_cm} 不存在。当前列有：{list(test_df.columns)}")
    
    # Accumulate confusion matrices for all models.
    for cell in test_df[col_cm]:
        # The cell can be a JSON string, list[list], or np.array.
        if isinstance(cell, str):
            try:
                arr = np.array(json.loads(cell), dtype=int)
            except Exception:
                arr = np.array(ast.literal_eval(cell), dtype=int)
        else:
            arr = np.array(cell, dtype=int)
        if arr.shape != (num_conds, num_conds):
            raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {(num_conds, num_conds)}")
        group_CM += arr  # Accumulate each model confusion matrix.
    
    # Normalize by row and convert to percentages.
    row_sums = group_CM.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero.
    norm_group_CM = group_CM / row_sums * 100

    # Display values in percentage format.
    percent_text = np.vectorize(lambda x: f"{x:.1f}%")(norm_group_CM)

    # Draw the confusion matrix.
    fig, ax = plt.subplots(figsize=(cm_to_inch(7), cm_to_inch(5.5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=norm_group_CM, display_labels=mc_labels)
    disp.plot(include_values=False, cmap=plt.cm.Blues, ax=ax, colorbar=False)
    
    # Write the percentage in each cell.
    for i in range(norm_group_CM.shape[0]):
        for j in range(norm_group_CM.shape[1]):
            ax.text(
                j, i, percent_text[i, j],  # Display the percentage text at the corresponding position.
                ha="center", va="center",  # Center horizontally and vertically.
                color=("white" if i == j else "black"),  # Use white for diagonal cells and black for other cells.
                fontweight="bold", fontsize=6
            )

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="center", rotation_mode="anchor")
    ax.tick_params(axis='x', pad=5)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.show()  # Show the confusion matrix figure.

    # Save the confusion matrix figure.
    filename = f'{model_i}_Group_Confusion_Matrix.tiff'
    fig.savefig(output_plot_dir / filename)
    plt.close()  # Close the figure.

#%% 4. Plot checkpoint outputs on scalp topographies
            
import mne

# Specify the checkpoint list to output; only output the first two checkpoints.
ckpt_list = [1, 2]

# Create an MNE info object containing electrode names, electrode type (EEG), and sampling rate.
info = mne.create_info(ch_names=electrode_list, sfreq=256, ch_types='eeg')

# Set electrode positions with the standard montage provided by EEGLAB.
montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
# Alternatively, use the standard 10-05 montage by commenting out the line above and using the line below.
# montage = mne.channels.make_standard_montage('standard_1005')

# Apply the montage to the EEG info object.
info.set_montage(montage)

# Iterate over each participant, checkpoint, and event condition.
for sub_i, ckpt_i, cond_i in itertools.product(sub_list, ckpt_list, list(range(num_conds))):
    
    # Create the output folder for each participant.
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the file name and read the data.
    filename = f'ckpt_{ckpt_i}.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
    
    # Select data by event condition.
    indices = np.where(y == cond_i)[0]
    X_select = X[indices, :, :]
    
    # Get the current event name.
    event_name = mc_labels[cond_i]
    
    # Compute the variance of selected data as an approximation of power.
    power = np.var(X_select, axis=-1)
    
    # Compute the minimum and maximum power for each electrode.
    power_min = np.min(power, axis=1, keepdims=True)
    power_max = np.max(power, axis=1, keepdims=True)
    
    # Apply Min-Max normalization to the power data.
    normalized_power = (power - power_min) / (power_max - power_min)
    
    # Average the normalized power.
    average_normalized_power = np.mean(normalized_power, axis=0)
    
    # Create a figure and axes, and draw the scalp map with MNE topomap visualization.
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(6)))  # Set the figure size.
    mne.viz.plot_topomap(
        data=average_normalized_power,  # Normalized power data.
        pos=info,  # Electrode position information.
        ch_type='eeg',  # Electrode type.
        sensors=True,  # Show sensor positions.
        outlines='head',  # Draw the head outline.
        sphere=0.080,  # Head radius.
        vlim=(min(average_normalized_power), max(average_normalized_power)),  # Data range.
        res=600,  # Resolution.
        contours=10,  # Number of contour lines.
        cmap='RdYlBu_r',  # Colormap (red-blue).
        axes=ax  # Specify the axes.
    )

    plt.tight_layout()  # Adjust the layout to avoid clipping figure content.
    # Save the figure.
    fig_dir = spatial_filter_save_dir / f"sub{sub_i}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    filename = f'sub_{sub_i}_ckpt_{ckpt_i}_{event_name}.tiff'  # Set the file name.
    fig.savefig(fig_dir / filename, bbox_inches='tight')  # Save the figure file.
    
    # Close the figure.
    plt.close(fig)

    
#%% Plot activation-map outputs on scalp topographies

# Iterate over each participant and event condition.
for sub_i, cond_i in itertools.product(sub_list, list(range(num_conds))):
    
    # Create the output folder for each participant.
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the file name and read data from the first checkpoint.
    filename = f'ckpt_1.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X_1 = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
        
    # Set the file name and read data from the second checkpoint.
    filename = f'ckpt_2.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X_2 = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
    
    # Compute the difference between the two checkpoint outputs.
    X = X_2 - X_1
    
    # Select data by event condition.
    indices = np.where(y == cond_i)[0]  # Get indices matching the current event condition.
    X_select = X[indices, :, :]  # Select data by index.
    
    # Get the current event name.
    event_name = mc_labels[cond_i]
    
    # Compute the variance of selected data as an approximation of power.
    power = np.var(X_select, axis=-1)
    
    # Compute the minimum and maximum power for each electrode.
    power_min = np.min(power, axis=1, keepdims=True)
    power_max = np.max(power, axis=1, keepdims=True)
    
    # Apply Min-Max normalization to the power data.
    normalized_power = (power - power_min) / (power_max - power_min)
    
    # Average the normalized power.
    average_normalized_power = np.mean(normalized_power, axis=0)
    
    # Create a figure and axes, and draw the scalp map with MNE topomap visualization.
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(6)))  # Set the figure size.
    mne.viz.plot_topomap(
        data=average_normalized_power,  # Normalized power data.
        pos=info,  # Electrode position information.
        ch_type='eeg',  # Electrode type.
        sensors=True,  # Show sensor positions.
        outlines='head',  # Draw the head outline.
        sphere=0.080,  # Head radius.
        vlim=(min(average_normalized_power), max(average_normalized_power)),  # Data range.
        res=600,  # Resolution.
        contours=10,  # Number of contour lines.
        cmap='RdYlBu_r',  # Colormap (red-blue).
        axes=ax  # Specify the axes.
    )

    plt.tight_layout()  # Adjust the layout to avoid clipping figure content.
    # Save the figure.
    fig_dir = spatial_filter_save_dir / f"sub{sub_i}_filter_active"
    fig_dir.mkdir(parents=True, exist_ok=True)  # Create the folder for saving figures.
    filename = f'sub_{sub_i}_{event_name}.tiff'  # Set the file name.
    fig.savefig(fig_dir / filename, bbox_inches='tight')  # Save the figure file.
    
    # Close the figure.
    plt.close(fig)


          
#%% 5. Plot checkpoint difference outputs (mean minus each group) on scalp topographies
            
# Select the checkpoints to process, assuming the first two are selected.
ckpt_list = [1, 2]

# Iterate over each participant, checkpoint, and event condition.
for sub_i, ckpt_i, cond_i in itertools.product(sub_list, ckpt_list, list(range(num_conds))):
    
    # Create the folder for each participant.
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)
    
    # Set the file name and read the corresponding checkpoint data.
    filename = f'ckpt_{ckpt_i}.h5'
    
    # Read data from the HDF5 file.
    with h5py.File(h5_dir / filename, 'r') as f:
        X = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
    
    # Compute the reference power for the whole dataset as variance over time.
    ref_power = np.var(X, axis=-1)

    # Compute the mean reference power.
    avgref_power = np.mean(ref_power, axis=0)
    
    # Select samples for the specified condition.
    indices = np.where(y == cond_i)[0]  # Select samples by event condition.
    X_select = X[indices, :, :]  # Get the selected samples.
    
    # Get the current event name.
    event_name = mc_labels[cond_i]
    
    # Compute variance of the selected data as power.
    power = np.var(X_select, axis=-1)
    
    # Compute the difference from the reference power.
    diff_power = power - avgref_power
    
    # Compute the minimum and maximum for each row (electrode).
    diff_power_min = np.min(diff_power, axis=1, keepdims=True)
    diff_power_max = np.max(diff_power, axis=1, keepdims=True)
    
    # Apply Min-Max normalization to scale the differences to [-1, 1].
    normalized_diff_power = 2 * (diff_power - diff_power_min) / (diff_power_max - diff_power_min) - 1
    
    # Average the normalized difference power across all samples for each electrode.
    average_normalized_diff_power = np.mean(normalized_diff_power, axis=0)
    
    # Create a figure and axes to display the scalp map.
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(6)))  # Set the figure size.
    mne.viz.plot_topomap(
        data=average_normalized_diff_power,  # Normalized difference power data.
        pos=info,  # Electrode position information.
        ch_type='eeg',  # Electrode type.
        sensors=True,  # Show sensor positions.
        outlines='head',  # Draw the head outline.
        vlim=(min(average_normalized_diff_power), max(average_normalized_diff_power)),  # Set the data range.
        sphere=0.080,  # Head radius.
        res=600,  # Resolution.
        contours=10,  # Number of contour lines.
        cmap='RdYlBu_r',  # Colormap (red-blue).
        axes=ax  # Specify the axes.
    )
    
    # Create a folder for saving figures.
    fig_dir = spatial_filter_save_dir / f"sub{sub_i}_diff"
    fig_dir.mkdir(parents=True, exist_ok=True)  # Create the directory for saving figures.
    
    # Set the file name and save the figure.
    filename = f'sub_{sub_i}_ckpt_{ckpt_i}_{event_name}_diff.tiff'  # Set the file name.
    fig.savefig(fig_dir / filename, bbox_inches='tight')  # Save the figure file.
    
    # Close the figure and release resources.
    plt.close(fig)



#%% tsne subject 7

# Define the checkpoint name list.
check_point_name_list = ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3', 'ckpt_4', 'classifier']

# Set the folder path for saving t-SNE figures.
sub_tsne_dir = tsne_save_dir / "sub7"

# Iterate over each checkpoint name.
for ckpt_name in check_point_name_list:
    
    # Set the file name and read the corresponding HDF5 data.
    filename = f'{ckpt_name}.h5'
    with h5py.File(sub_tsne_dir / filename, 'r') as f:
        X = f['data'][:]  # Read the 'data' dataset as a NumPy array.
        y = f['label'][:]  # Read the 'label' dataset as a NumPy array.
    
    # Create the figure and axes, and set the figure size.
    fig, ax = plt.subplots(figsize=(cm_to_inch(5.2), cm_to_inch(4)))
    
    # Color the data points by class label y.
    for cond_i in range(num_conds):  # Assume there are 5 classes.
        plt.scatter(X[y == cond_i, 0],  # Filter data by class.
                    X[y == cond_i, 1],  # Filter data by class.
                    label=f'{mc_labels[cond_i]}',  # Set the label.
                    alpha=0.6,  # Set the alpha value.
                    s=3,  # Set the point size.
                    linewidth=0,  # Set the line width.
                    edgecolor='none',  # Set no edge color.
                    color=colors[cond_i])  # Set the point color.
    
    # Set the x- and y-axis ranges.
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    # Set the x- and y-axis ticks.
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    
    # Set the x- and y-axis labels.
    ax.set_xlabel('t-SNE dim_1')
    ax.set_ylabel('t-SNE dim_2')
    
    # Set the file name and save the figure.
    filename = f'sub_{sub_i}_{ckpt_name}.tiff'  # Set the output file name.
    fig.savefig(sub_tsne_dir / filename, dpi=600)  # Save the figure at 600 dpi.
    plt.show(fig)  # Show the figure.
    plt.close(fig)  # Close the figure and release resources.




















































