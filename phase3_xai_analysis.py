# -*- coding: utf-8 -*-
"""
Phase III: Explainable AI (XAI) Assessment
Generates SHAP values, Normalized MDI, and Permutation Feature Importance
using the final regularized RF model.
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Loading dataset for XAI analysis...")
    df = pd.read_csv("catalyst_dataset.csv", sep=";")
    if 'Cu_Ratio' in df.columns:
        df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
        
    features = ['Cu_Loading', 'Reaction_Time']
    X = df[features]
    y_ct_c0 = df['Ct_C0']

    # Final Model Refit
    rf = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_leaf=1, random_state=42)
    rf.fit(X, y_ct_c0)

    # 1. SHAP Analysis
    print("Generating SHAP Summary Plot...")
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    plt.figure(figsize=(8, 5))
    shap.summary_plot(shap_values, X, feature_names=features, show=False)
    plt.title("SHAP Summary Plot for Ct/C0")
    plt.tight_layout()
    plt.savefig("SHAP_Summary.png", dpi=300)
    plt.close()

    # 2. Normalized MDI Analysis
    print("\nExtracting Normalized MDI...")
    mdi_importances = rf.feature_importances_
    for name, imp in zip(features, mdi_importances):
        print(f"Feature: {name} -> Normalized MDI: {imp*100:.2f}%")

    # 3. Permutation Feature Importance
    print("\nCalculating Permutation Feature Importance (PFI)...")
    perm_importance = permutation_importance(rf, X, y_ct_c0, n_repeats=50, random_state=42, n_jobs=-1)
    
    for i, name in enumerate(features):
        print(f"Feature: {name} -> Mean Decrease in R2: {perm_importance.importances_mean[i]:.4f} +/- {perm_importance.importances_std[i]:.4f}")

    print("\nXAI Analysis complete. Plots saved to local directory.")

if __name__ == '__main__':
    main()