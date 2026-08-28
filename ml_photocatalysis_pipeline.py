# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 18:35:19 2026

@author: Metin Zontul
"""

# -*- coding: utf-8 -*-
"""
Machine Learning and XAI Pipeline for Time-Dependent Photocatalytic Degradation
-------------------------------------------------------------------------------
This script implements a chronologically validated explainable machine learning 
framework using Random Forest (RF) regression. It predicts the residual 
concentration ratio (Ct/C0), degradation efficiency, and ln(C0/Ct) based on 
nominal Cu-loading and reaction time.

Key Features (Updated for Reviewer Revisions):
- Regularized Random Forest (max_depth=5, min_samples_leaf=1) to prevent overfitting.
- Target-Specific Robustness Evaluation (5-Fold CV and LOOCV).
- Chronological Out-of-Time (OOT) Diagnostic Validation.
- Explainable AI (XAI) suite including SHAP, MDI, Permutation Feature Importance 
  (to eliminate cardinality bias), and Partial Dependence Plots (PDP).
- Multiprocessing disabled (n_jobs=None) to prevent Windows teardown errors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import warnings
warnings.filterwarnings('ignore') # Gereksiz uyarıları gizler

def calculate_metrics(y_true, y_pred):
    """Calculates evaluation metrics for regression models."""
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred)
    }

def main():
    # ==============================================================================
    # 1. GERÇEK VERİ SETİ YÜKLEME AŞAMASI
    # ==============================================================================
    print("Gerçek veri seti yükleniyor...")
    try:
        # Veriniz noktalı virgül ile ayrıldığı için sep=";" kullanıyoruz
        df = pd.read_csv("catalyst_dataset.csv", sep=";")
        
        # Sütun adını makaledeki ve koddaki standarda uydurmak için yeniden adlandırıyoruz
        if 'Cu_Ratio' in df.columns:
            df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
            
    except FileNotFoundError:
        print("HATA: 'catalyst_dataset.csv' dosyası bulunamadı! Lütfen aynı klasörde olduğundan emin olun.")
        return

    # Özellikler (X) ve Hedefler (y)
    X = df[['Cu_Loading', 'Reaction_Time']]
    y_ct_c0 = df['Ct_C0']
    y_eff = df['Efficiency'] / 100  # Normalize edilmiş verim
    y_ln = df['ln_C0_Ct']

    targets = {'Ct_C0': y_ct_c0, 'Efficiency_Norm': y_eff, 'ln_C0_Ct': y_ln}

    # ==============================================================================
    # 2. HAKEM 2 GÜNCELLEMESİ: REGULARIZED RANDOM FOREST
    # ==============================================================================
    # Windows fatal exception hatasını önlemek için n_jobs=None yapıldı
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        min_samples_leaf=1, 
        random_state=42,
        n_jobs=None  
    )

    # ==============================================================================
    # PHASE I: TARGET-SPECIFIC ROBUSTNESS (5-Fold CV and LOOCV)
    # ==============================================================================
    print("\n--- PHASE I: CROSS-VALIDATION RESULTS ---")
    cv_5fold = KFold(n_splits=5, shuffle=True, random_state=42)
    loo = LeaveOneOut()

    for target_name, y in targets.items():
        preds_5f = cross_val_predict(rf_model, X, y, cv=cv_5fold)
        metrics_5f = calculate_metrics(y, preds_5f)
        
        preds_loo = cross_val_predict(rf_model, X, y, cv=loo)
        metrics_loo = calculate_metrics(y, preds_loo)
        
        print(f"\nTarget: {target_name}")
        print(f"5-Fold CV -> R2: {metrics_5f['R2']:.4f}, MAE: {metrics_5f['MAE']:.4f}, MSE: {metrics_5f['MSE']:.4f}")
        print(f"LOOCV     -> R2: {metrics_loo['R2']:.4f}, MAE: {metrics_loo['MAE']:.4f}, MSE: {metrics_loo['MSE']:.4f}")

    # ==============================================================================
    # PHASE II: CHRONOLOGICAL DIAGNOSTIC EVALUATION (OOT Validation)
    # ==============================================================================
    print("\n--- PHASE II: CHRONOLOGICAL OOT VALIDATION (t > 100 min) ---")
    train_mask = df['Reaction_Time'] <= 100
    test_mask = df['Reaction_Time'] > 100

    X_train, X_test = X[train_mask], X[test_mask]

    for target_name, y in targets.items():
        y_train, y_test = y[train_mask], y[test_mask]
        
        rf_model.fit(X_train, y_train)
        preds_test = rf_model.predict(X_test)
        metrics_oot = calculate_metrics(y_test, preds_test)
        
        print(f"Target: {target_name} | OOT R2: {metrics_oot['R2']:.4f}")

    # ==============================================================================
    # PHASE III: EXPLAINABLE AI (XAI) & DIAGNOSTICS
    # ==============================================================================
    print("\n--- PHASE III: XAI AND DIAGNOSTICS ---")

    # Modeli tüm veriyle nihai olarak eğitiyoruz
    rf_model.fit(X, y_ct_c0)

    # 1. SHAP Analizi
    print("SHAP grafiği üretiliyor...")
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot for Ct/C0")
    plt.tight_layout()
    plt.savefig("Figure_4_SHAP.png", dpi=600)
    plt.close()

    # 2. MDI (Gini) Önem Skorları
    mdi_importances = rf_model.feature_importances_
    print(f"MDI Importances -> Cu Loading: {mdi_importances[0]:.4f}, Reaction Time: {mdi_importances[1]:.4f}")

    # 3. Permutation Feature Importance (Hakem 2 Talebi)
    print("Permütasyon Analizi (Permutation Importance) grafiği üretiliyor...")
    # Windows çökmesini engellemek için n_jobs=None eklendi
    perm_importance = permutation_importance(rf_model, X, y_ct_c0, n_repeats=10, random_state=42, n_jobs=None)
    sorted_idx = perm_importance.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        labels=np.array(['Cu Loading (%)', 'Reaction Time (min)'])[sorted_idx]
    )
    ax.set_title("Permutation Feature Importance (Unbiased Evaluation)")
    ax.set_xlabel("Decrease in R2 Score")
    fig.tight_layout()
    plt.savefig("Figure_6_Permutation_Importance.png", dpi=600)
    plt.close()

    # 4. Partial Dependence Plots (Hakem 1 Talebi)
    print("PDP (Partial Dependence Plots) grafiği üretiliyor...")
    fig, ax = plt.subplots(figsize=(10, 4))
    display = PartialDependenceDisplay.from_estimator(
        rf_model, 
        X, 
        features=[0, 1], 
        feature_names=['Cu Loading (%)', 'Reaction Time (min)'],
        grid_resolution=50,
        ax=ax
    )
    plt.suptitle("Partial Dependence Plots (PDP) for Ct/C0")
    fig.tight_layout()
    plt.savefig("Figure_7_PDP.png", dpi=600)
    plt.close()

    print("\nAnaliz tamamlandı. 600 DPI çözünürlüğündeki grafikler kaydedildi.")

# ==============================================================================
# ENTRY POINT SHIELD (Windows Çökme Kalkanı)
# ==============================================================================
if __name__ == '__main__':
    main()
