# -*- coding: utf-8 -*-
"""
Phase I-B & II: Target-Specific RF Robustness & Chronological OOT Evaluation
Evaluates the selected regularized RF model (max_depth=5) across 10 controlled
seeds for Ct/C0 and ln(C0/Ct).

Author: Metin Zontul
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_true, y_pred):
    return [r2_score(y_true, y_pred), mean_absolute_error(y_true, y_pred), mean_squared_error(y_true, y_pred)]

def main():
    df = pd.read_csv("catalyst_dataset.csv", sep=";")
    if 'Cu_Ratio' in df.columns:
        df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
        
    X = df[['Cu_Loading', 'Reaction_Time']].values
    targets = {'Ct_C0': df['Ct_C0'].values, 'ln_C0_Ct': df['ln_C0_Ct'].values}

    seeds = list(range(10))
    
    print("--- PHASE I-B: TARGET SPECIFIC ROBUSTNESS (max_depth=5) ---")
    for target_name, y in targets.items():
        metrics_5f = []
        metrics_loo = []
        
        for seed in seeds:
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=seed)
            
            # 5-Fold CV
            cv_5fold = KFold(n_splits=5, shuffle=True, random_state=seed)
            preds_5f = cross_val_predict(rf, X, y, cv=cv_5fold, n_jobs=-1)
            metrics_5f.append(calculate_metrics(y, preds_5f))
            
            # LOOCV
            loo = LeaveOneOut()
            preds_loo = cross_val_predict(rf, X, y, cv=loo, n_jobs=-1)
            metrics_loo.append(calculate_metrics(y, preds_loo))
            
        avg_5f = np.mean(metrics_5f, axis=0)
        std_5f = np.std(metrics_5f, axis=0, ddof=1)
        avg_loo = np.mean(metrics_loo, axis=0)
        std_loo = np.std(metrics_loo, axis=0, ddof=1)
        
        print(f"\nTarget: {target_name}")
        print(f"5-Fold CV -> R2: {avg_5f[0]:.4f}±{std_5f[0]:.4f}, MAE: {avg_5f[1]:.4f}±{std_5f[1]:.4f}")
        print(f"LOOCV     -> R2: {avg_loo[0]:.4f}±{std_loo[0]:.4f}, MAE: {avg_loo[1]:.4f}±{std_loo[1]:.4f}")

    print("\n--- PHASE II: CHRONOLOGICAL OOT EVALUATION (t > 100 min) ---")
    train_mask = df['Reaction_Time'] <= 100
    test_mask = df['Reaction_Time'] > 100
    X_train, X_test = X[train_mask], X[test_mask]

    for target_name, y in targets.items():
        y_train, y_test = y[train_mask], y[test_mask]
        oot_metrics = []
        
        for seed in seeds:
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=seed)
            rf.fit(X_train, y_train)
            preds_test = rf.predict(X_test)
            oot_metrics.append(calculate_metrics(y_test, preds_test))
            
        avg_oot = np.mean(oot_metrics, axis=0)
        std_oot = np.std(oot_metrics, axis=0, ddof=1)
        
        print(f"Target: {target_name} | OOT R2: {avg_oot[0]:.4f}±{std_oot[0]:.4f}, MAE: {avg_oot[1]:.4f}±{std_oot[1]:.4f}")

if __name__ == '__main__':
    main()