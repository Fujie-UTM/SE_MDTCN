# -*- coding: utf-8 -*-
"""
Created on Thu Feb  5 14:45:27 2026

@author: Fujie
"""

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

#%%

def run_wilcoxon_for_sheet(sheet_name: str):
    df = pd.read_excel(path, sheet_name=sheet_name)

    # 自动识别模型列（数值列）
    model_cols = [
        c for c in df.columns
        if c not in index_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(model_cols) < 2:
        raise ValueError(f"[{sheet_name}] 检测到的数值模型列少于2列，请检查表格列类型/是否有非模型数值列需要排除。")

    means = df[model_cols].mean().sort_values(ascending=False)
    stds  = df[model_cols].std(ddof=1).reindex(means.index)

    sheet_baseline = baseline if baseline is not None else means.index[0]
    if sheet_baseline not in model_cols:
        raise ValueError(f"[{sheet_name}] baseline='{sheet_baseline}' 不在模型列中：{model_cols}")

    # 基准 vs 其他：Wilcoxon
    rows = []
    for other in model_cols:
        if other == sheet_baseline:
            continue

        x = df[sheet_baseline]
        y = df[other]
        mask = ~(x.isna() | y.isna())
        x2, y2 = x[mask], y[mask]

        stat, p = wilcoxon(
            x2, y2,
            alternative=alternative,
            zero_method="wilcox",
            method="auto"
        )

        rows.append({
            "baseline": sheet_baseline,
            "other": other,
            "n_pairs": int(mask.sum()),
            "mean_baseline": float(x2.mean()),
            "mean_other": float(y2.mean()),
            "mean_diff(baseline-other)": float((x2 - y2).mean()),
            "W_stat": float(stat),
            "p_value": float(p),
        })

    res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)

    # Bonferroni 校正
    reject, p_bonf, _, _ = multipletests(res["p_value"].values, alpha=0.05, method="bonferroni")
    res["p_bonferroni"] = p_bonf
    res["reject_bonf_0.05"] = reject

    return means, stds, res

#%%

# ===== 配置 =====
path = r"./Model_comparison_results.xlsx"
baseline = "SE_MDTCN_RAW"
index_cols = {"sub_i"}  # 视你的表格情况增删
alternative = "greater"  # 与你提供的代码一致

# ===== 分别跑 Train / Test，并写到同一个 xlsx =====
train_means, train_stds, train_res = run_wilcoxon_for_sheet("Train")
test_means,  test_stds,  test_res  = run_wilcoxon_for_sheet("Test")

out_path = r"./wilcoxon_model_comparison_train_test.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    train_summary = pd.concat([train_means, train_stds], axis=1)
    train_summary.columns = ["mean_accuracy", "std_accuracy"]
    train_summary.to_excel(writer, sheet_name="Train_means")

    train_res.to_excel(writer, index=False, sheet_name="Train_wilcoxon")

    test_summary = pd.concat([test_means, test_stds], axis=1)
    test_summary.columns = ["mean_accuracy", "std_accuracy"]
    test_summary.to_excel(writer, sheet_name="Test_means")

    test_res.to_excel(writer, index=False, sheet_name="Test_wilcoxon")

print(f"Saved: {out_path}")

#%%

# ===== 配置 =====
path = r"./Ablation_experiments_results.xlsx"
baseline = "SE_MDTCN_RAW"
index_cols = {"sub_i"}  # 视你的表格情况增删
alternative = "greater"  # 与你提供的代码一致

# ===== 分别跑 Train / Test，并写到同一个 xlsx =====
train_means, train_stds, train_res = run_wilcoxon_for_sheet("Train")
test_means,  test_stds,  test_res  = run_wilcoxon_for_sheet("Test")

out_path = r"./wilcoxon_ablation_experiments_train_test.xlsx"
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    train_summary = pd.concat([train_means, train_stds], axis=1)
    train_summary.columns = ["mean_accuracy", "std_accuracy"]
    train_summary.to_excel(writer, sheet_name="Train_means")

    train_res.to_excel(writer, index=False, sheet_name="Train_wilcoxon")

    test_summary = pd.concat([test_means, test_stds], axis=1)
    test_summary.columns = ["mean_accuracy", "std_accuracy"]
    test_summary.to_excel(writer, sheet_name="Test_means")

    test_res.to_excel(writer, index=False, sheet_name="Test_wilcoxon")

print(f"Saved: {out_path}")
