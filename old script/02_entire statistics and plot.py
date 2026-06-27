# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 00:49:38 2025

@author: Fujie
"""

import os
from pathlib import Path

root = Path(r'E:\SE_MDTCN_all_20251126')
os.chdir(root)
current_path = Path.cwd()
print("Current path:", current_path)


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
from SE_MDTCN import SE_MDTCN


#%% 1.1. Set working directory

# Create the output results directory
output_dir = root / 'output_results'  # Set the output folder path
output_dir.mkdir(parents=True, exist_ok=True)  # Create the folder (no error if it already exists)

# List of folder names for different models
folder_name_list = [
    "SE_MDTCN",
    "SE_MDTCN_woElectrodeNormalization",  # Model without electrode normalization
    "SE_MDTCN_woSE_SpatialFilter",        # Model without the spatial filter
    "SE_MDTCN_woSE_DWTCNBlock",           # Model without the spatial filter and DWTCN block
    "SE_MDTCN_wZScoreNormalization",      # Model using Z-score normalization
    "SE_MDTCN_wDWConvSpatialFilter2D",    # Model using a 2D DWConv spatial filter
    "SE_MDTCN_wTemporalFilter2D"          # Model using a 2D temporal filter
]

# Paths for training/validation/test data
train_data_dir = root / '7_h5py/Training set/-500_2000/Gamma/raw'      # Training set data path
valid_data_dir = root / '7_h5py/Validation set/-500_2000/Gamma/raw'    # Validation set data path
test_data_dir = root / '7_h5py/Test set/-500_2000/Gamma/raw'           # Test set data path

# Training output path
train_dir = root / 'SE_MDTCN/raw_-500_2000_Gamma/train'  # Training output path

# Output path for model checkpoints (and create the folder)
ckpt_output_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/checkpoint_output'  # Checkpoint output path
ckpt_output_dir.mkdir(parents=True, exist_ok=True)  # Create the folder (no error if it already exists)

# Path to save spatial filter results (and create the folder)
spatial_filter_save_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/SpatialFilter'  # Spatial filter save path
spatial_filter_save_dir.mkdir(parents=True, exist_ok=True)  # Create the folder (no error if it already exists)

# Load the best model parameters and optimizer parameters from a CSV file
best_model_params, best_opt_params = load_best_params_from_csv(
    best_params_csv=train_dir / "best_params_GSresults.csv",  # CSV file path
    model_col="model_params",  # Model-parameter column
    opt_col="opt_params"       # Optimizer-parameter column
)

# Path to save t-SNE results (and create the folder)
tsne_save_dir = output_dir / 'SE_MDTCN/raw_-500_2000_Gamma/TSNE'  # t-SNE save path
tsne_save_dir.mkdir(parents=True, exist_ok=True)  # Create the folder (no error if it already exists)


#%%  1.2. Set parameters

# Define the subject list containing integers from 0 to 14 (15 subjects in total)
sub_list = list(range(15))

# Define the event label dictionary: keys are event names, values are integer IDs
event_id = {'Hello': 0, 'Help me': 1, 'Stop': 2, 'Thank you': 3, 'Yes': 4}

# Define a function to find dictionary keys corresponding to a given list of values
def find_keys_by_values(dict_obj, value_list):
    keys = []  # Store matched keys here
    for value in value_list:  # Iterate over all given values
        for k, v in dict_obj.items():  # Iterate over dictionary key-value pairs
            if v == value:  # If the value matches
                keys.append(k)  # Append the corresponding key to the result list
    return keys

# Get all event labels
mc_labels = find_keys_by_values(event_id, event_id.values())

# Compute the number of event conditions
num_conds = len(mc_labels)  # Number of event conditions

# Electrode position dictionary: maps electrode index to electrode name
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

# Create a list of electrode names, sorted by electrode index to preserve ordering
electrode_list = [electrode_dict[key] for key in sorted(electrode_dict.keys())]


#%% 2. Summarize test-set results for all comparison models

# Create an empty list to store test results for each folder/model
test_results = []

# Iterate over all model folder names
for folder_name_i in folder_name_list:

    # Path to the test results CSV for each model
    test_result_path = root / folder_name_i / "raw_-500_2000_Gamma/test/test_results.csv"

    # Read the test results into a DataFrame and make a copy
    test_df = pd.read_csv(test_result_path).copy()

    # Compute mean and standard deviation of evaluation metrics
    acc_mean = test_df['acc'].mean()  # Mean accuracy
    acc_std = test_df['acc'].std()    # Std of accuracy
    kappa_mean = test_df['cohen_kappa'].mean()  # Mean Cohen's Kappa
    f1_mean = test_df['f1_macro'].mean()  # Mean F1 score (macro)

    # Initialize an empty group confusion matrix: (num_conds x num_conds)
    group_CM = np.zeros((num_conds, num_conds), dtype=int)

    # Standardize the confusion-matrix column name
    col_cm = "confusion_matrix"

    # If "confusion_matrix" is not in test_df, raise an error
    if col_cm not in test_df.columns:
        raise KeyError(f"Column {col_cm} does not exist. Current columns: {list(test_df.columns)}")

    # Parse and accumulate confusion matrices
    for cell in test_df[col_cm]:
        # cell may be a JSON string, list[list], or np.array
        if isinstance(cell, str):
            try:
                # If JSON string, parse into a NumPy array
                arr = np.array(json.loads(cell), dtype=int)
            except Exception:
                # If parsing fails, try ast.literal_eval for Python literal strings
                arr = np.array(ast.literal_eval(cell), dtype=int)
        else:
            # If already array-like, convert directly
            arr = np.array(cell, dtype=int)

        # Validate confusion matrix shape
        if arr.shape != (num_conds, num_conds):
            raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {(num_conds, num_conds)}")

        # Accumulate into the group confusion matrix
        group_CM += arr

    # Create a row dict for this model and append to the result list
    new_row = {
        "model": folder_name_i,            # Model name
        "acc_mean": float(acc_mean),       # Mean accuracy
        "acc_std": float(acc_std),         # Std accuracy
        "f1_mean": float(f1_mean),         # Mean F1 (macro)
        "kappa_mean": float(kappa_mean),   # Mean Cohen's Kappa
        "confusion_matrix": group_CM       # Accumulated confusion matrix
    }

    # Append current model results
    test_results.append(new_row)

# Convert all model results to a DataFrame and save as CSV
test_results = pd.DataFrame(test_results)
test_results.to_csv(output_dir / "ablation_results.csv", index=False, encoding="utf-8")



#%% 3. Summarize test-set results for all comparison models

# Load test results of the first model (SE_MDTCN_raw) and copy into a DataFrame
all_model_df = pd.read_csv(root / "SE_MDTCN/raw_-500_2000_Gamma/test/test_results.csv").copy()
# Insert a 'model_name' column with value "SE_MDTCN_raw"
all_model_df.insert(0, 'model_name', "SE_MDTCN_raw")

# Load test results of the second model (SE_MDTCN_clean) and copy into a DataFrame
all_model_df2 = pd.read_csv(root / "SE_MDTCN/clean_-500_2000_Gamma/test/test_results.csv").copy()
# Insert a 'model_name' column with value "SE_MDTCN_clean"
all_model_df2.insert(0, 'model_name', "SE_MDTCN_clean")

# Concatenate the two DataFrames row-wise (axis=0) and reset the index
all_model_df = pd.concat([all_model_df, all_model_df2], axis=0, ignore_index=True)

# Directory containing the compared models
compared_models_dir = root / 'compared_models_20251128'

# List of compared model names
compared_model_name_list = ["EEGNet", "EEGTCNet", "EEGConformer", "HS_STDCN", "TSception_Full"]

# Loop over each compared model name
for compared_model_i in compared_model_name_list:

    # Build the test result path for the current compared model
    compared_test_result_path = compared_models_dir / compared_model_i / "raw_-500_2000_Gamma/test/test_results.csv"

    # Read the compared model test results and copy into a DataFrame
    compared_test_df = pd.read_csv(compared_test_result_path).copy()

    # Insert a 'model_name' column for the compared model
    compared_test_df.insert(0, 'model_name', compared_model_i)

    # Append the compared model results to all_model_df
    all_model_df = pd.concat([all_model_df, compared_test_df], axis=0, ignore_index=True)

# Save all model test results to a CSV file
all_model_df.to_csv(output_dir / "model_comparison_results.csv", index=False, encoding="utf-8")

# List of all model names (including SE_MDTCN_raw, SE_MDTCN_clean, and compared models)
all_model_name_list = ["SE_MDTCN_raw", "SE_MDTCN_clean"] + compared_model_name_list

# Initialize a list to store summary results for all models
model_results = []

# Loop over all model names
for model_i in all_model_name_list:

    # Filter rows for the current model
    model_df = all_model_df[all_model_df['model_name'] == model_i]

    # Compute mean metrics for the current model
    acc_mean = model_df['acc'].mean()                 # Mean accuracy
    acc_std = model_df['acc'].std()                   # Std accuracy
    kappa_mean = model_df['cohen_kappa'].mean()       # Mean Cohen's Kappa
    f1_mean = model_df['f1_macro'].mean()             # Mean macro F1

    # Initialize an all-zero confusion matrix (num_conds x num_conds)
    group_CM = np.zeros((num_conds, num_conds), dtype=int)

    # Standardize the confusion-matrix column name
    col_cm = "confusion_matrix"
    if col_cm not in model_df.columns:
        raise KeyError(f"Column {col_cm} does not exist. Current columns: {list(model_df.columns)}")

    # Accumulate confusion matrices for the current model
    for cell in model_df[col_cm]:
        # cell may be a JSON string, list[list], or np.array
        if isinstance(cell, str):
            try:
                # If JSON string, parse into a NumPy array
                arr = np.array(json.loads(cell), dtype=int)
            except Exception:
                # If parsing fails, try ast.literal_eval for Python literal strings
                arr = np.array(ast.literal_eval(cell), dtype=int)
        else:
            # If already array-like, convert directly
            arr = np.array(cell, dtype=int)

        # Validate confusion matrix shape
        if arr.shape != (num_conds, num_conds):
            raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {(num_conds, num_conds)}")

        # Add into the group confusion matrix
        group_CM += arr

    # Create a row dict for this model
    new_row = {
        "model": model_i,              # Model name
        "acc_mean": float(acc_mean),   # Mean accuracy
        "acc_std": float(acc_std),     # Std accuracy
        "f1_mean": float(f1_mean),     # Mean macro F1
        "kappa_mean": float(kappa_mean),  # Mean Cohen's Kappa
        "confusion_matrix": group_CM   # Accumulated confusion matrix
    }

    # Append to summary results
    model_results.append(new_row)

# Convert summary results to a DataFrame and save to CSV
model_results = pd.DataFrame(model_results)
model_results.to_csv(output_dir / "model_comparison_summary_results.csv", index=False, encoding="utf-8")



#%% 4. Extract outputs at each checkpoint for the model trained on the training set

# List of checkpoint names representing different model checkpoints
check_point_name_list = ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3', 'ckpt_4', 'classifier']

# Use itertools.product to generate all combinations of subjects and checkpoints
for sub_i, ckpt_name in itertools.product(sub_list, check_point_name_list):

    # Load training, validation, and test data
    X_train, y_train = load_data(train_data_dir, sub_i)
    X_valid, y_valid = load_data(valid_data_dir, sub_i)
    X_test, y_test = load_data(test_data_dir, sub_i)

    # Concatenate training, validation, and test data
    X = np.concatenate((X_train, X_valid, X_test), axis=0)
    y = np.concatenate((y_train, y_valid, y_test), axis=0)

    # Create an EEGDataset instance with loaded data X and labels y
    eeg_dataset = EEGDataset(EEGdata=X, EEGlabel=y)

    # Get all data as tensors
    X_tensor, _ = eeg_dataset.get_all_data()

    # Set model path (assume the best model per subject is saved under "best models")
    model_path = train_dir / "best models" / f"sub{sub_i}.pth"

    # Load the pretrained model with `load_model`
    model = load_model(
        model_cls=SE_MDTCN,            # Model class
        model_kwargs=best_model_params,  # Best model parameters
        model_path=model_path,         # Model checkpoint path
        device='cpu'                   # Use CPU for inference
    )

    # Current checkpoint name
    check_point_name = f"{ckpt_name}"

    # Get data at the specified checkpoint
    # `is_backbone=False` may indicate not using the backbone-only path
    if ckpt_name == 'raw':
        ckpt_data = X  # If "raw", directly use the raw input data
    else:
        ckpt_data = get_checkpoint_data(X_tensor, model, check_point_name, is_backbone=False)

    # Create a subject-specific folder to store the HDF5 file
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # HDF5 filename includes the checkpoint name
    filename = f'{ckpt_name}.h5'

    # Write to HDF5
    with h5py.File(h5_dir / filename, 'w') as f:
        # Save processed data and labels with gzip compression
        f.create_dataset('data', data=ckpt_data, dtype='float32',
                         compression="gzip", compression_opts=9)
        f.create_dataset('label', data=y, dtype='int32',
                         compression="gzip", compression_opts=9)


#%% 5. t-SNE for participant 7

sub_i = 7  # Set the current subject ID to 7

# Create a folder to save t-SNE outputs (path includes the subject ID)
sub_tsne_dir = tsne_save_dir / f"sub{sub_i}"
sub_tsne_dir.mkdir(parents=True, exist_ok=True)  # Create it if it doesn't exist

# Initialize data normalizers
normal_1 = StandardScaler()  # Standard scaler: zero mean and unit variance
normal_2 = MinMaxScaler(feature_range=(-1, 1))  # Min-max scaler: scale into [-1, 1]

# Initialize t-SNE with specific settings
tsne = TSNE(
    n_components=2,
    random_state=99,
    method='exact',
    init='pca',
    max_iter=10000,
    n_iter_without_progress=500,
    n_jobs=8,
    perplexity=50
)  # PCA init, max_iter=10000, n_jobs=8 uses 8 CPU cores in parallel

# Loop over each checkpoint name
for ckpt_name in check_point_name_list:

    # Create the subject folder for reading checkpoint data
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # Set filename and load data
    filename = f'{ckpt_name}.h5'
    X = []
    with h5py.File(h5_dir / filename, 'r') as f:
        X = f['data'][:]   # Load dataset 'data' as a NumPy array
        y = f['label'][:]  # Load dataset 'label' as a NumPy array

    # For certain checkpoints, compute variance along the last dimension (time)
    if ckpt_name in ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3']:
        X = np.var(X, axis=-1)  # Compute variance for raw and early checkpoints
    else:
        X = X  # For other checkpoints, keep data unchanged

    # Standardize the data
    X_norm = normal_1.fit_transform(X)        # Standardize using normal_1
    X_tsne = tsne.fit_transform(X_norm)       # Run t-SNE to get 2D embeddings
    X_tsne_norm = normal_2.fit_transform(X_tsne)  # Scale embeddings into [-1, 1]

    # Save t-SNE results to an HDF5 file
    filename = f'{ckpt_name}.h5'
    with h5py.File(sub_tsne_dir / filename, 'w') as f:
        # Save the processed t-SNE data with gzip compression
        f.create_dataset(
            'data', data=X_tsne_norm, dtype='float32',
            compression="gzip", compression_opts=9
        )
        f.create_dataset(
            'label', data=y, dtype='int32',
            compression="gzip", compression_opts=9
        )


#%% 6. Plot Setup

# Function to convert centimeters to inches
def cm_to_inch(value):
    return value / 2.54  # 1 inch = 2.54 cm, return the value in inches

# Set global figure properties (sizes are intended to be in centimeters)
# https://matplotlib.org/stable/users/explain/customizing.html#matplotlibrc-sample

plt.rcParams['figure.autolayout'] = True  # Enable auto layout to adjust subplot positions/sizes
plt.rcParams['figure.constrained_layout.use'] = False  # Disable constrained layout
plt.rcParams['font.family'] = 'Arial'  # Set global font to Arial
plt.rcParams['font.size'] = 8  # Global font size = 8
plt.rcParams['axes.labelsize'] = 8  # Font size for x/y axis labels
plt.rcParams['axes.titlesize'] = 8  # Font size for plot titles
plt.rcParams['xtick.labelsize'] = 6  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6  # Font size for y-axis tick labels
plt.rcParams['xtick.direction'] = 'in'  # Point x-axis ticks inward
plt.rcParams['ytick.direction'] = 'in'  # Point y-axis ticks inward
plt.rcParams['legend.fontsize'] = 6  # Legend font size
plt.rcParams['figure.titlesize'] = 0  # Disable figure title

# Legend styling in rcParams
plt.rcParams['legend.title_fontsize'] = 0  # Disable legend title
plt.rcParams['legend.fontsize'] = 8  # Set legend font size
plt.rcParams['legend.markerscale'] = 0.8  # Scale legend marker size
plt.rcParams['legend.columnspacing'] = 0.5  # Spacing between legend columns
plt.rcParams['legend.borderaxespad'] = 0.5  # Padding between legend and axes
plt.rcParams['legend.borderpad'] = 0  # Padding inside the legend border
plt.rcParams['legend.framealpha'] = 0  # Legend transparency (0 means fully transparent)
plt.rcParams['legend.labelspacing'] = 0.1  # Vertical spacing between legend entries
plt.rcParams['legend.handlelength'] = 1.0  # Length of legend handles
plt.rcParams['legend.loc'] = 'upper right'  # Legend location

plt.rcParams['legend.handletextpad'] = 0.5  # Horizontal gap between markers and text
plt.rcParams['legend.frameon'] = True  # Show legend frame
plt.rcParams['legend.facecolor'] = 'lightgray'  # Set legend background to light gray

# Figure resolution (DPI)
plt.rcParams['figure.dpi'] = 600  # Figure DPI
plt.rcParams['savefig.dpi'] = 600  # Saved figure DPI
plt.rcParams['savefig.format'] = 'tiff'  # Save format: TIFF
plt.rcParams['savefig.bbox'] = 'standard'  # Use standard bbox when saving

# Color list for plotting
colors = ['red', 'orange', 'green', 'blue', 'purple']

#%% 7. Plot test-set performance

# Set the output directory for plots and create the folder if needed
output_plot_dir = output_dir / 'report_drawing'
output_plot_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

# Load model test results
all_model_df = pd.read_csv(output_dir / 'model_comparison_results.csv')  # Load model comparison results

# Loop over all model names
for model_i in all_model_name_list:

    # 1. Filter test data by model name
    test_df = all_model_df[all_model_df['model_name'] == model_i].copy()

    # Convert accuracy to percentage
    test_df['acc'] = test_df['acc'] * 100

    # Compute mean and standard deviation of accuracy
    mean_acc = test_df['acc'].mean()
    std_acc = test_df['acc'].std()

    # Plot accuracy bar chart for each model
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(5)))
    sns.barplot(
        data=test_df, x='sub_i', y='acc', ax=ax,
        saturation=0.75, width=0.8,
        err_kws={'color': 'black', 'linewidth': 1}
    )

    # Text annotation: mean and standard deviation
    textstr = f"Mean = {mean_acc:.1f}%\nSD = {std_acc:.1f}%"
    ax.text(
        0.95, 0.95, textstr,
        fontsize=6,
        transform=ax.transAxes,
        ha='right', va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5),
    )

    # Add chance-level line (20%)
    ax.axhline(y=20, color="red", linewidth=0.5)
    ax.text(x=14.5, y=21, s="chance level", fontsize=6, color="red",
            ha='right', fontweight='bold')

    # Axis formatting
    ax.set_ylim(bottom=0, top=120)
    ax.set_yticks(ticks=np.arange(0, 101, 20), labels=np.arange(0, 101, 20))
    ax.set_xticks(ticks=sub_list, labels=sub_list)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=True, right=False, labelleft=True)
    ax.set_xlabel("Participant ID")
    ax.set_ylabel("Accuracy (%)")

    plt.show()

    # Save the figure as a TIFF file
    filename = f'{model_i}_group_classification_results.tiff'
    fig.savefig(output_plot_dir / filename)
    plt.close()

    # 2. Plot confusion matrix
    group_CM = np.zeros((num_conds, num_conds), dtype=int)

    # Standardize confusion-matrix column name
    col_cm = "confusion_matrix"
    if col_cm not in test_df.columns:
        raise KeyError(f"Column {col_cm} does not exist. Current columns: {list(test_df.columns)}")

    # Accumulate confusion matrices
    for cell in test_df[col_cm]:
        # cell may be a JSON string, list[list], or np.array
        if isinstance(cell, str):
            try:
                arr = np.array(json.loads(cell), dtype=int)
            except Exception:
                arr = np.array(ast.literal_eval(cell), dtype=int)
        else:
            arr = np.array(cell, dtype=int)

        if arr.shape != (num_conds, num_conds):
            raise ValueError(f"Confusion matrix shape mismatch: got {arr.shape}, expected {(num_conds, num_conds)}")
        group_CM += arr

    # Row-normalize and convert to percentages
    row_sums = group_CM.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    norm_group_CM = group_CM / row_sums * 100

    # Format percentages for display
    percent_text = np.vectorize(lambda x: f"{x:.1f}%")(norm_group_CM)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(5)))
    disp = ConfusionMatrixDisplay(confusion_matrix=norm_group_CM, display_labels=mc_labels)
    disp.plot(include_values=False, cmap=plt.cm.Blues, ax=ax, colorbar=False)

    # Write percentages in each cell
    for i in range(norm_group_CM.shape[0]):
        for j in range(norm_group_CM.shape[1]):
            ax.text(
                j, i, percent_text[i, j],
                ha="center", va="center",
                color=("white" if i == j else "black"),
                fontweight="bold", fontsize=6
            )

    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=True)
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="center", rotation_mode="anchor")
    ax.tick_params(axis='x', pad=5)
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    plt.show()

    # Save confusion matrix figure
    filename = f'{model_i}_Group_Confusion_Matrix.tiff'
    fig.savefig(output_plot_dir / filename)
    plt.close()


#%% 8. Plot checkpoint outputs on topographic maps

import mne

# Specify the checkpoints to export (only the first two checkpoints)
ckpt_list = [1, 2]

# Create an MNE Info object containing electrode names, channel type (EEG), and sampling rate
info = mne.create_info(ch_names=electrode_list, sfreq=256, ch_types='eeg')

# Set electrode positions using an EEGLAB standard montage
montage = mne.channels.read_custom_montage('standard-10-5-cap385.elp')
# Alternatively, you can use a built-in standard 10-05 montage (comment the line above and use below)
# montage = mne.channels.make_standard_montage('standard_1005')

# Apply the montage to the EEG Info object
info.set_montage(montage)

# Iterate over each subject, each checkpoint, and each event condition
for sub_i, ckpt_i, cond_i in itertools.product(sub_list, ckpt_list, list(range(num_conds))):

    # Create the output folder for each subject
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # Set filename and load data
    filename = f'ckpt_{ckpt_i}.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X = f['data'][:]   # Load dataset 'data' as a NumPy array
        y = f['label'][:]  # Load dataset 'label' as a NumPy array

    # Select samples belonging to the current condition
    indices = np.where(y == cond_i)[0]
    X_select = X[indices, :, :]

    # Get the current event name
    event_name = mc_labels[cond_i]

    # Compute variance as an approximation of power
    power = np.var(X_select, axis=-1)

    # Compute per-sample min and max power across electrodes
    power_min = np.min(power, axis=1, keepdims=True)
    power_max = np.max(power, axis=1, keepdims=True)

    # Min-max normalize the power
    normalized_power = (power - power_min) / (power_max - power_min)

    # Average normalized power across samples
    average_normalized_power = np.mean(normalized_power, axis=0)

    # Create a figure and axis, then plot a scalp topomap using MNE
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(6)))
    mne.viz.plot_topomap(
        data=average_normalized_power,     # Normalized power data
        pos=info,                          # Electrode location info
        ch_type='eeg',                     # Channel type
        sensors=True,                      # Show sensor locations
        outlines='head',                   # Draw head outline
        sphere=0.080,                      # Head radius
        vlim=(min(average_normalized_power), max(average_normalized_power)),  # Data range
        res=600,                           # Resolution
        contours=20,                       # Number of contour lines
        cmap='RdYlBu_r',                   # Colormap (red-blue)
        axes=ax                            # Target axis
    )

    plt.tight_layout()  # Adjust layout to avoid overlaps

    # Save the figure
    fig_dir = spatial_filter_save_dir / f"sub{sub_i}"
    fig_dir.mkdir(parents=True, exist_ok=True)
    filename = f'sub_{sub_i}_ckpt_{ckpt_i}_{event_name}.tiff'
    fig.savefig(fig_dir / filename, bbox_inches='tight')

    # Close the figure
    plt.close(fig)

    
#%% 9. Plot activation map outputs on topographic maps

# Iterate over each subject and each event condition
for sub_i, cond_i in itertools.product(sub_list, list(range(num_conds))):

    # Create the output folder for each subject
    h5_dir = ckpt_output_dir / f"sub{sub_i}"
    h5_dir.mkdir(parents=True, exist_ok=True)

    # Load data from the first checkpoint
    filename = 'ckpt_1.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X_1 = f['data'][:]  # Load dataset 'data' as a NumPy array
        y = f['label'][:]   # Load dataset 'label' as a NumPy array

    # Load data from the second checkpoint
    filename = 'ckpt_2.h5'
    with h5py.File(h5_dir / filename, 'r') as f:
        X_2 = f['data'][:]  # Load dataset 'data' as a NumPy array
        y = f['label'][:]   # Load dataset 'label' as a NumPy array

    # Compute the difference between the two checkpoints
    X = X_2 - X_1

    # Select samples belonging to the current condition
    indices = np.where(y == cond_i)[0]  # Indices matching the current condition
    X_select = X[indices, :, :]         # Select data by indices

    # Get the current event name
    event_name = mc_labels[cond_i]

    # Compute variance as an approximation of power
    power = np.var(X_select, axis=-1)

    # Compute per-sample min and max power across electrodes
    power_min = np.min(power, axis=1, keepdims=True)
    power_max = np.max(power, axis=1, keepdims=True)

    # Min-max normalize the power
    normalized_power = (power - power_min) / (power_max - power_min)

    # Average normalized power across samples
    average_normalized_power = np.mean(normalized_power, axis=0)

    # Create a figure and axis, then plot a scalp topomap using MNE
    fig, ax = plt.subplots(figsize=(cm_to_inch(6), cm_to_inch(6)))
    mne.viz.plot_topomap(
        data=average_normalized_power,     # Normalized power data
        pos=info,                          # Electrode location info
        ch_type='eeg',                     # Channel type
        sensors=True,                      # Show sensor locations
        outlines='head',                   # Draw head outline
        sphere=0.080,                      # Head radius
        vlim=(min(average_normalized_power), max(average_normalized_power)),  # Data range
        res=600,                           # Resolution
        contours=20,                       # Number of contour lines
        cmap='RdYlBu_r',                   # Colormap (red-blue)
        axes=ax                            # Target axis
    )

    plt.tight_layout()  # Adjust layout to avoid overlaps

    # Save the figure
    fig_dir = spatial_filter_save_dir / f"sub{sub_i}_filter_active"
    fig_dir.mkdir(parents=True, exist_ok=True)
    filename = f'sub_{sub_i}_{event_name}.tiff'
    fig.savefig(fig_dir / filename, bbox_inches='tight')

    # Close the figure
    plt.close(fig)

          
#%% 10. t-SNE subject 7

# Define the checkpoint name list
check_point_name_list = ['raw', 'ckpt_1', 'ckpt_2', 'ckpt_3', 'ckpt_4', 'classifier']

# Folder path for saving t-SNE images
sub_tsne_dir = tsne_save_dir / "sub7"

# Loop over each checkpoint name
for ckpt_name in check_point_name_list:

    # Set filename and load data from the corresponding .h5 file
    filename = f'{ckpt_name}.h5'
    with h5py.File(sub_tsne_dir / filename, 'r') as f:
        X = f['data'][:]   # Load dataset 'data' as a NumPy array
        y = f['label'][:]  # Load dataset 'label' as a NumPy array

    # Create figure and axis, set figure size
    fig, ax = plt.subplots(figsize=(cm_to_inch(5.2), cm_to_inch(4)))

    # Color points by class labels (y)
    for cond_i in range(num_conds):  # Assume there are 5 classes
        plt.scatter(
            X[y == cond_i, 0],  # Filter by class
            X[y == cond_i, 1],  # Filter by class
            label=f'{mc_labels[cond_i]}',  # Legend label
            alpha=0.6,          # Transparency
            s=3,                # Point size
            linewidth=0,        # Line width
            edgecolor='none',   # No edge
            color=colors[cond_i]  # Point color
        )

    # Set x/y axis limits
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)

    # Set x/y ticks
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])

    # Set axis labels
    ax.set_xlabel('t-SNE dim_1')
    ax.set_ylabel('t-SNE dim_2')

    # Set filename and save the figure
    filename = f'sub_{sub_i}_{ckpt_name}.tiff'
    fig.savefig(sub_tsne_dir / filename, dpi=600)

    plt.show(fig)
    plt.close(fig)





















































