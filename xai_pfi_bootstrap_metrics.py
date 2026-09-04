# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 20:57:23 2026

"""

"""
Permutation Feature Importance (PFI) and Bootstrap Stability Analysis
-------------------------------------------------------------------
This script computes advanced feature-attribution metrics for the 
strictly regularized (max_depth=5) Random Forest model.

Features:
- Computes 10-seed cross-validated marginal PFI.
- Computes 10-seed Out-Of-Fold (OOF) grouped-conditional PFI.
- Performs a 1000-repetition moving-block bootstrap stability analysis
  to preserve the temporal structure of the kinetic dataset.

Author: Metin Zontul
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('ignore')

def permute_marginal(X, col_idx):
    X_perm = X.copy()
    np.random.shuffle(X_perm[:, col_idx])
    return X_perm

def permute_conditional(X, perm_col_idx, cond_col_idx):
    X_perm = X.copy()
    unique_conds = np.unique(X[:, cond_col_idx])
    for val in unique_conds:
        mask = (X[:, cond_col_idx] == val)
        vals = X_perm[mask, perm_col_idx]
        np.random.shuffle(vals)
        X_perm[mask, perm_col_idx] = vals
    return X_perm

def main():
    print("Loading dataset and initializing max_depth=5 model...")
    try:
        df = pd.read_csv("catalyst_dataset.csv", sep=";")
        if 'Cu_Ratio' in df.columns:
            df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
    except FileNotFoundError:
        print("Dataset not found. Please provide 'catalyst_dataset.csv'.")
        return
        
    X = df[['Cu_Loading', 'Reaction_Time']].values
    y = df['Ct_C0'].values
    
    seeds = range(10)
    n_perms = 50
    
    cv_marg_r2_seeds, cv_marg_mae_seeds = {'Cu': [], 'Time': []}, {'Cu': [], 'Time': []}
    oof_marg_r2_seeds, oof_marg_mae_seeds = {'Cu': [], 'Time': []}, {'Cu': [], 'Time': []}
    oof_cond_r2_seeds, oof_cond_mae_seeds = {'Cu': [], 'Time': []}, {'Cu': [], 'Time': []}
    
    print("Running 10-seed CV and OOF Permutations (Marginal & Conditional)...")
    for seed in seeds:
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        
        oof_base = np.zeros(len(y))
        oof_marg_perms = {'Cu': np.zeros((n_perms, len(y))), 'Time': np.zeros((n_perms, len(y)))}
        oof_cond_perms = {'Cu': np.zeros((n_perms, len(y))), 'Time': np.zeros((n_perms, len(y)))}
        
        cv_marg_r2_folds = {'Cu': [], 'Time': []}
        cv_marg_mae_folds = {'Cu': [], 'Time': []}
        
        for train_idx, test_idx in kf.split(X):
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=seed)
            rf.fit(X[train_idx], y[train_idx])
            
            pred_base = rf.predict(X[test_idx])
            oof_base[test_idx] = pred_base
            r2_base = r2_score(y[test_idx], pred_base)
            mae_base = mean_absolute_error(y[test_idx], pred_base)
            
            for i, feat in enumerate(['Cu', 'Time']):
                r2_drops, mae_incs = [], []
                for p in range(n_perms):
                    # Marginal
                    X_test_marg = permute_marginal(X[test_idx], i)
                    pred_marg = rf.predict(X_test_marg)
                    
                    r2_drops.append(r2_base - r2_score(y[test_idx], pred_marg))
                    mae_incs.append(mean_absolute_error(y[test_idx], pred_marg) - mae_base)
                    oof_marg_perms[feat][p, test_idx] = pred_marg
                    
                    # Conditional
                    cond_idx = 1 if i == 0 else 0
                    X_test_cond = permute_conditional(X[test_idx], i, cond_idx)
                    oof_cond_perms[feat][p, test_idx] = rf.predict(X_test_cond)
                    
                cv_marg_r2_folds[feat].append(np.mean(r2_drops))
                cv_marg_mae_folds[feat].append(np.mean(mae_incs))
                
        for feat in ['Cu', 'Time']:
            cv_marg_r2_seeds[feat].append(np.mean(cv_marg_r2_folds[feat]))
            cv_marg_mae_seeds[feat].append(np.mean(cv_marg_mae_folds[feat]))
            
            base_oof_r2 = r2_score(y, oof_base)
            base_oof_mae = mean_absolute_error(y, oof_base)
            
            oof_marg_r2_drops = [base_oof_r2 - r2_score(y, oof_marg_perms[feat][p]) for p in range(n_perms)]
            oof_marg_mae_incs = [mean_absolute_error(y, oof_marg_perms[feat][p]) - base_oof_mae for p in range(n_perms)]
            
            oof_cond_r2_drops = [base_oof_r2 - r2_score(y, oof_cond_perms[feat][p]) for p in range(n_perms)]
            oof_cond_mae_incs = [mean_absolute_error(y, oof_cond_perms[feat][p]) - base_oof_mae for p in range(n_perms)]
            
            oof_marg_r2_seeds[feat].append(np.mean(oof_marg_r2_drops))
            oof_marg_mae_seeds[feat].append(np.mean(oof_marg_mae_incs))
            
            oof_cond_r2_seeds[feat].append(np.mean(oof_cond_r2_drops))
            oof_cond_mae_seeds[feat].append(np.mean(oof_cond_mae_incs))

    print("\nRunning 1000-Repetition Moving-Block Bootstrap (This may take a minute)...")
    times = np.sort(np.unique(X[:, 1]))
    blocks = [times[i:i+3] for i in range(len(times)-2)]
    
    def boot_rep(seed):
        np.random.seed(seed)
        sampled_times = []
        while len(sampled_times) < len(times):
            idx = np.random.randint(0, len(blocks))
            sampled_times.extend(blocks[idx])
        sampled_times = sampled_times[:len(times)]
        
        boot_dfs = [df[df['Reaction_Time'] == t] for t in sampled_times]
        df_boot = pd.concat(boot_dfs, ignore_index=True)
        
        X_boot = df_boot[['Cu_Loading', 'Reaction_Time']].values
        y_boot = df_boot['Ct_C0'].values
        
        rf_boot = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=seed)
        rf_boot.fit(X_boot, y_boot)
        
        mdi = rf_boot.feature_importances_ * 100
        base_r2 = r2_score(y, rf_boot.predict(X))
        
        marg_r2_drops, cond_r2_drops = {'Cu': [], 'Time': []}, {'Cu': [], 'Time': []}
        for i, feat in enumerate(['Cu', 'Time']):
            for p in range(50):
                X_marg = permute_marginal(X, i)
                marg_r2_drops[feat].append(base_r2 - r2_score(y, rf_boot.predict(X_marg)))
                
                cond_idx = 1 if i == 0 else 0
                X_cond = permute_conditional(X, i, cond_idx)
                cond_r2_drops[feat].append(base_r2 - r2_score(y, rf_boot.predict(X_cond)))
                
        return [mdi[0], mdi[1], np.mean(marg_r2_drops['Cu']), np.mean(marg_r2_drops['Time']), 
                np.mean(cond_r2_drops['Cu']), np.mean(cond_r2_drops['Time'])]

    # Run parallel bootstrap
    boot_results = Parallel(n_jobs=-1)(delayed(boot_rep)(i) for i in range(1000))
    boot_results = np.array(boot_results)
    medians = np.median(boot_results, axis=0)

    print("\n" + "="*60)
    print("MANUSCRIPT OUTPUT SUMMARY (max_depth=5)")
    print("="*60)
    
    print("\n--- Cross-validated Marginal PFI ---")
    print(f"Reaction time R2 drop: {np.mean(cv_marg_r2_seeds['Time']):.3f} ± {np.std(cv_marg_r2_seeds['Time'], ddof=1):.3f}")
    print(f"Cu-loading R2 drop: {np.mean(cv_marg_r2_seeds['Cu']):.3f} ± {np.std(cv_marg_r2_seeds['Cu'], ddof=1):.3f}")
    print(f"Reaction time MAE inc: {np.mean(cv_marg_mae_seeds['Time']):.4f} ± {np.std(cv_marg_mae_seeds['Time'], ddof=1):.4f}")
    print(f"Cu-loading MAE inc: {np.mean(cv_marg_mae_seeds['Cu']):.4f} ± {np.std(cv_marg_mae_seeds['Cu'], ddof=1):.4f}")

    print("\n--- OOF Marginal PFI ---")
    print(f"Reaction time OOF R2 drop: {np.mean(oof_marg_r2_seeds['Time']):.3f} ± {np.std(oof_marg_r2_seeds['Time'], ddof=1):.3f}")
    print(f"Cu-loading OOF R2 drop: {np.mean(oof_marg_r2_seeds['Cu']):.3f} ± {np.std(oof_marg_r2_seeds['Cu'], ddof=1):.3f}")
    print(f"Reaction time OOF MAE inc: {np.mean(oof_marg_mae_seeds['Time']):.4f} ± {np.std(oof_marg_mae_seeds['Time'], ddof=1):.4f}")
    print(f"Cu-loading OOF MAE inc: {np.mean(oof_marg_mae_seeds['Cu']):.4f} ± {np.std(oof_marg_mae_seeds['Cu'], ddof=1):.4f}")

    print("\n--- Grouped-Conditional PFI ---")
    print(f"Reaction time Cond OOF R2 drop: {np.mean(oof_cond_r2_seeds['Time']):.3f} ± {np.std(oof_cond_r2_seeds['Time'], ddof=1):.3f}")
    print(f"Cu-loading Cond OOF R2 drop: {np.mean(oof_cond_r2_seeds['Cu']):.3f} ± {np.std(oof_cond_r2_seeds['Cu'], ddof=1):.3f}")
    print(f"Reaction time Cond OOF MAE inc: {np.mean(oof_cond_mae_seeds['Time']):.4f} ± {np.std(oof_cond_mae_seeds['Time'], ddof=1):.4f}")
    print(f"Cu-loading Cond OOF MAE inc: {np.mean(oof_cond_mae_seeds['Cu']):.4f} ± {np.std(oof_cond_mae_seeds['Cu'], ddof=1):.4f}")

    print("\n--- Bootstrap Medians ---")
    print(f"MDI Median -> Cu-loading: {medians[0]:.2f}% | Reaction time: {medians[1]:.2f}%")
    print(f"Marginal PFI Median R2 -> Cu-loading: {medians[2]:.3f} | Reaction time: {medians[3]:.3f}")
    print(f"Conditional PFI Median R2 -> Cu-loading: {medians[4]:.3f} | Reaction time: {medians[5]:.3f}")
    print("="*60)

if __name__ == '__main__':
    main()