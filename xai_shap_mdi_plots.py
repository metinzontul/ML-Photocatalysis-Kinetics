# -*- coding: utf-8 -*-
"""
Explainable AI (XAI) Plot Generator
Generates publication-quality (600 DPI) SHAP summary plots and Impurity-based Feature Importance (MDI)
stacked bar charts.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

"""
Manuscript XAI Plotting Code: SHAP and MDI (Figure 3 & Figure 4)
Updated according to reviewer revisions (Regularization and Cu-loading terminology).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import shap
import warnings

warnings.filterwarnings('ignore')

# 1. Plot settings for manuscript format (Times New Roman, high resolution)
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

print("Preparing manuscript XAI figures (Figure 3 and Figure 4). Please wait...")

# 2. Load Dataset and Preprocess (Compatible with the new CSV pipeline)
try:
    df = pd.read_csv("catalyst_dataset.csv", sep=";")
    if 'Cu_Ratio' in df.columns:
        df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
except FileNotFoundError:
    print("ERROR: 'catalyst_dataset.csv' not found. Please ensure the file is in the same directory.")
    exit()

X = df[['Cu_Loading', 'Reaction_Time']]

# Reviewer 1 correction: "Cu-loading" terminology instead of "Cu Doping"
feature_names = ['Cu-loading Ratio (%)', 'Reaction Time (min)']

# Scale inputs (Standardize X for model stability, keep Y physical)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Normalize the efficiency column to (n/100) format as presented in the manuscript
df['Efficiency_Norm'] = df['Efficiency'] / 100

# Targets to Plot
targets = [
    {'col': 'Efficiency_Norm', 'label': 'Degradation Efficiency', 'suffix': 'Efficiency'},
    {'col': 'Ct_C0', 'label': 'Ct/C0', 'suffix': 'Ct_C0'},
    {'col': 'ln_C0_Ct', 'label': 'ln(C0/Ct)', 'suffix': 'ln_C0_Ct'}
]

mdi_results = []

# 3. Independent Model Training, MDI, and SHAP Extraction per Target
for target in targets:
    y = df[target['col']]
    
    # REVIEWER 2 UPDATE: Regularized model to prevent overfitting
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        min_samples_leaf=1, 
        random_state=42,
        n_jobs=None
    )
    rf_model.fit(X_scaled_df, y)
    
    # --- A. MDI (Impurity-based Feature Importance) Extraction ---
    importances = rf_model.feature_importances_ * 100 # Convert to percentage format
    mdi_results.append({
        'Target': target['label'],
        'Cu-loading Ratio': importances[0],
        'Reaction Time': importances[1]
    })
    
    # --- B. SHAP Analysis (Figure 3) ---
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, show=False)
    
    # Manuscript-standard title and label settings
    plt.title(f"SHAP Summary: Impact on {target['label']}", fontweight='bold', pad=15)
    plt.xlabel(f"SHAP Value (Impact on Model Output for {target['label']})", fontweight='bold')
    
    plt.tight_layout()
    output_shap = f"Figure_3_SHAP_{target['suffix']}_600DPI.png"
    plt.savefig(output_shap, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f" - SHAP summary plot saved for {target['label']}: {output_shap}")

# 4. MDI Feature Importance Bar Chart Generation (Figure 4)
print("\nPlotting MDI Percentage Ratios (Figure 4)...")
mdi_df = pd.DataFrame(mdi_results).set_index('Target')

fig, ax = plt.subplots(figsize=(8, 6))

# Create 'stacked bar' reflecting reviewer corrections ("Cu-loading" terminology)
mdi_df[['Reaction Time', 'Cu-loading Ratio']].plot(
    kind='bar', 
    stacked=True, 
    ax=ax, 
    color=['#1f77b4', '#ff7f0e'], 
    edgecolor='black',
    width=0.5
)

# Chart text and axis settings
ax.set_title("Impurity-based Feature Importance (MDI)", fontweight='bold', pad=15)
ax.set_ylabel("Relative Importance (%)", fontweight='bold')
ax.set_xlabel("Predicted Model Outputs", fontweight='bold')
ax.set_ylim(0, 100)
plt.xticks(rotation=0, fontweight='bold')

# Place legend horizontally below the chart (with updated names)
ax.legend(['Reaction Time', 'Cu-loading Ratio'], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Print percentage values inside the bars
for container in ax.containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
output_mdi = "Figure_4_MDI_Importance_600DPI.png"
plt.savefig(output_mdi, dpi=600, bbox_inches='tight')
plt.close()

print(f" - MDI Bar chart successfully saved: {output_mdi}")
print("\nAll XAI figures are ready in high resolution!")