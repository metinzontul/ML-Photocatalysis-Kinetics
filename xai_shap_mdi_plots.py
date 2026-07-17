# -*- coding: utf-8 -*-
"""
Explainable AI (XAI) Plot Generator
Generates publication-quality (600 DPI) SHAP summary plots and Impurity-based Feature Importance (MDI)
stacked bar charts.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
import warnings

warnings.filterwarnings('ignore')[cite: 8]

# 1. Plot settings for publication quality (Times New Roman, high resolution)
sns.set_theme(style="ticks", context="paper", font_scale=1.2)[cite: 8]
plt.rcParams["font.family"] = "serif"[cite: 8]
plt.rcParams["font.serif"] = ["Times New Roman"][cite: 8]

print("Preparing XAI figures (Figure 3 and Figure 4). Please wait...")[cite: 8]

# 2. Load Dataset and Preprocess
try:
    df = pd.read_excel('ML_Dataset_Catalist.xlsx')[cite: 8]
except FileNotFoundError:
    print("ERROR: 'ML_Dataset_SabitHoca_Catalist.xlsx' not found.")[cite: 8]
    exit()[cite: 8]

X = df[['Cu_Ratio', 'Reaction_Time']][cite: 8]
feature_names = ['Cu Doping Ratio (%)', 'Reaction Time (min)'][cite: 8]

# Standardize inputs for model stability (Outputs remain in physical scale)
scaler = StandardScaler()[cite: 8]
X_scaled = scaler.fit_transform(X)[cite: 8]
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)[cite: 8]

# Target definitions for looping
targets = [
    {'col': 'Efficiency', 'label': 'Degradation Efficiency (%)', 'suffix': 'Efficiency'},
    {'col': 'Ct_C0', 'label': 'Ct/C0', 'suffix': 'Ct_C0'},
    {'col': 'ln_C0_Ct', 'label': 'ln(C0/Ct)', 'suffix': 'ln_C0_Ct'}
][cite: 8]

mdi_results = []

# 3. Independent Model Training, MDI, and SHAP Extraction per Target
for target in targets:
    y = df[target['col']][cite: 8]
    
    # Train target-specific model (critical for isolating MDI percentages)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)[cite: 8]
    rf_model.fit(X_scaled_df, y)[cite: 8]
    
    # --- A. Extract Impurity-based Feature Importance (MDI) ---
    importances = rf_model.feature_importances_ * 100 # Convert to percentage format[cite: 8]
    mdi_results.append({
        'Target': target['label'],
        'Cu Doping Ratio': importances[0],
        'Reaction Time': importances[1]
    })[cite: 8]
    
    # --- B. SHAP Analysis (Figure 3) ---
    explainer = shap.TreeExplainer(rf_model)[cite: 8]
    shap_values = explainer.shap_values(X_scaled_df)[cite: 8]
    
    plt.figure(figsize=(7, 5))[cite: 8]
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, show=False)[cite: 8]
    
    # Publication-standard titles and labels
    plt.title(f"SHAP Summary: Impact on {target['label']}", fontweight='bold', pad=15)[cite: 8]
    plt.xlabel(f"SHAP Value (Impact on Model Output for {target['label']})", fontweight='bold')[cite: 8]
    
    plt.tight_layout()[cite: 8]
    output_shap = f"Figure_3_SHAP_{target['suffix']}_600DPI.png"[cite: 8]
    plt.savefig(output_shap, dpi=600, bbox_inches='tight')[cite: 8]
    plt.close()[cite: 8]
    
    print(f" - SHAP summary plot saved for {target['label']}: {output_shap}")[cite: 8]

# 4. MDI Feature Importance Bar Chart Generation (Figure 4)
print("\nGenerating MDI Percentage Bar Chart (Figure 4)...")[cite: 8]
mdi_df = pd.DataFrame(mdi_results).set_index('Target')[cite: 8]

fig, ax = plt.subplots(figsize=(8, 6))[cite: 8]

# Create stacked bar chart conforming to reviewer terminology
mdi_df[['Reaction Time', 'Cu Doping Ratio']].plot(
    kind='bar', 
    stacked=True, 
    ax=ax, 
    color=['#1f77b4', '#ff7f0e'], 
    edgecolor='black',
    width=0.5
)[cite: 8]

# Chart text and axis configurations
ax.set_title("Impurity-based Feature Importance (MDI)", fontweight='bold', pad=15)[cite: 8]
ax.set_ylabel("Relative Importance (%)", fontweight='bold')[cite: 8]
ax.set_xlabel("Predicted Model Outputs", fontweight='bold')[cite: 8]
ax.set_ylim(0, 100)[cite: 8]
plt.xticks(rotation=0, fontweight='bold')[cite: 8]

# Position the legend horizontally below the plot
ax.legend(['Reaction Time', 'Cu Doping Ratio'], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)[cite: 8]

# Print percentage values centrally within the bars
for container in ax.containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container][cite: 8]
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=12)[cite: 8]

plt.tight_layout()[cite: 8]
output_mdi = "Figure_4_MDI_Importance_600DPI.png"[cite: 8]
plt.savefig(output_mdi, dpi=600, bbox_inches='tight')[cite: 8]
plt.close()[cite: 8]

print(f" - MDI Bar chart successfully saved: {output_mdi}")[cite: 8]
print("\nAll XAI figures are ready in high resolution!")[cite: 8]