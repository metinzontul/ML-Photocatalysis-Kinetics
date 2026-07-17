# -*- coding: utf-8 -*-
"""
Random Forest Multi-Seed Robustness Evaluation
Evaluates the RF model across 10 random seeds using various cross-validation methods.
Target variables are unscaled to preserve their physical dimensions (Reviewer adjustment).

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd
import numpy as np
import warnings
import random
import os
from sklearn.model_selection import KFold, LeaveOneOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings('ignore')[cite: 6]

# 1. Load the dataset
df = pd.read_excel('ML_Dataset_Catalist.xlsx')[cite: 6]
X = df[['Cu_Ratio', 'Reaction_Time']].values[cite: 6]
Y = df[['Efficiency', 'Ct_C0', 'ln_C0_Ct']].values[cite: 6]

# Grouping variable: Cu_Ratio column
groups = X[:, 0][cite: 6]

# 2. Multi-Seed Analysis Setup
seeds = [42, 100, 2023, 7, 99, 1234, 55, 777, 88, 2026][cite: 6]
all_results = []

print(f"Starting detailed Random Forest tests for a total of {len(seeds)} random seeds...")[cite: 6]

for seed in seeds:
    os.environ['PYTHONHASHSEED'] = str(seed)[cite: 6]
    random.seed(seed)[cite: 6]
    np.random.seed(seed)[cite: 6]
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=seed)[cite: 6]
    
    cv_methods = {
        '5-Fold CV (Random)': KFold(n_splits=5, shuffle=True, random_state=seed),
        'LOOCV': LeaveOneOut(),
        'GroupKFold (Cu-Based)': GroupKFold(n_splits=5)
    }[cite: 6]
    
    for cv_name, cv in cv_methods.items():
        Y_pred_all = np.zeros_like(Y, dtype=float)[cite: 6]
        Y_true_all = np.zeros_like(Y, dtype=float)[cite: 6]
        
        for train_idx, test_idx in cv.split(X, Y, groups=groups):
            X_train, X_test = X[train_idx], X[test_idx][cite: 6]
            Y_train, Y_test = Y[train_idx], Y[test_idx][cite: 6]
            
            # Scale Inputs (X) only
            scaler_X = StandardScaler()[cite: 6]
            X_train_scaled = scaler_X.fit_transform(X_train)[cite: 6]
            X_test_scaled = scaler_X.transform(X_test)[cite: 6]
            
            # REVIEWER CORRECTION: MinMax scaling for outputs (Y) is cancelled.
            # The model will learn the physical dimensions directly.
            rf_model.fit(X_train_scaled, Y_train)[cite: 6]
            Y_pred_all[test_idx] = rf_model.predict(X_test_scaled)[cite: 6]
            Y_true_all[test_idx] = Y_test[cite: 6]
            
        r2_raw = r2_score(Y_true_all, Y_pred_all, multioutput='raw_values')[cite: 6]
        mae_raw = mean_absolute_error(Y_true_all, Y_pred_all, multioutput='raw_values')[cite: 6]
        mse_raw = mean_squared_error(Y_true_all, Y_pred_all, multioutput='raw_values')[cite: 6]
        
        for i, output_name in enumerate(['Efficiency (%)', 'Ct_C0', 'ln(C0/Ct)']):
            all_results.append({
                'Seed': seed,
                'CV Method': cv_name,
                'Output Variable': output_name,
                'R² Score': r2_raw[i],
                'MAE': mae_raw[i],
                'MSE': mse_raw[i]
            })[cite: 6]

df_all = pd.DataFrame(all_results)[cite: 6]

# 3. Statistical Summarization
df_summary = df_all.groupby(['CV Method', 'Output Variable']).agg(
    R2_Mean=('R² Score', 'mean'),
    R2_Std=('R² Score', 'std'),
    MAE_Mean=('MAE', 'mean'),
    MAE_Std=('MAE', 'std'),
    MSE_Mean=('MSE', 'mean'),
    MSE_Std=('MSE', 'std')
).reset_index()[cite: 6]

# 4. Format for Publication
df_paper = pd.DataFrame({
    'CV Method': df_summary['CV Method'],
    'Output Variable': df_summary['Output Variable'],
    'R² (Mean ± Std)': df_summary.apply(lambda row: f"{row['R2_Mean']:.4f} ± {row['R2_Std']:.4f}", axis=1),
    'MAE (Mean ± Std)': df_summary.apply(lambda row: f"{row['MAE_Mean']:.4f} ± {row['MAE_Std']:.4f}", axis=1),
    'MSE (Mean ± Std)': df_summary.apply(lambda row: f"{row['MSE_Mean']:.4f} ± {row['MSE_Std']:.4f}", axis=1)
})[cite: 6]

df_paper = df_paper.sort_values(by=['CV Method', 'Output Variable'])[cite: 6]

print("\n--- RF Multi-Seed Robustness Analysis (Physical Scales Preserved) ---")[cite: 6]
print(df_paper.to_string(index=False))[cite: 6]

output_file = 'RF_Detailed_with_GroupKFold_Fixed.xlsx'[cite: 6]
df_paper.to_excel(output_file, index=False)[cite: 6]