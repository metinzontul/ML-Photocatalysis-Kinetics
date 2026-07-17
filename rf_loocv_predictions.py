# -*- coding: utf-8 -*-
"""
LOOCV Data Generator for Scatter Plots
Executes a Leave-One-Out Cross-Validation routine to generate unscaled true vs. predicted 
coordinates for accurate visual representation of the physical dimensions.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings('ignore')[cite: 5]

# 1. Load the dataset
df = pd.read_excel('ML_Dataset_Catalist.xlsx')[cite: 5]
X = df[['Cu_Ratio', 'Reaction_Time']].values[cite: 5]
Y = df[['Efficiency', 'Ct_C0', 'ln_C0_Ct']].values[cite: 5]

# 2. Create empty arrays to store outputs
Y_pred_all = np.zeros_like(Y, dtype=float)[cite: 5]
Y_true_all = np.zeros_like(Y, dtype=float)[cite: 5]

# 3. LOOCV (Leave-One-Out) Loop Setup
loo = LeaveOneOut()[cite: 5]
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)[cite: 5]

print("Generating LOOCV predictions (y_pred). Please wait...")[cite: 5]

for train_idx, test_idx in loo.split(X):
    X_train, X_test = X[train_idx], X[test_idx][cite: 5]
    Y_train, Y_test = Y[train_idx], Y[test_idx][cite: 5]
    
    # Scale Inputs (X) only
    scaler_X = StandardScaler()[cite: 5]
    X_train_scaled = scaler_X.fit_transform(X_train)[cite: 5]
    X_test_scaled = scaler_X.transform(X_test)[cite: 5]
    
    # REVIEWER CORRECTION: Y outputs are strictly left unscaled.
    rf_model.fit(X_train_scaled, Y_train)[cite: 5]
    
    # Predict and record the test instance
    Y_pred_all[test_idx] = rf_model.predict(X_test_scaled)[cite: 5]
    Y_true_all[test_idx] = Y_test[cite: 5]

# 4. Convert results into a DataFrame with standardized column names
df_results = pd.DataFrame({
    'Actual_Efficiency': Y_true_all[:, 0],
    'Predicted_Efficiency': Y_pred_all[:, 0],
    'Actual_Ct_C0': Y_true_all[:, 1],
    'Predicted_Ct_C0': Y_pred_all[:, 1],
    'Actual_ln_C0_Ct': Y_true_all[:, 2],
    'Predicted_ln_C0_Ct': Y_pred_all[:, 2]
})[cite: 5]

# 5. Export to Excel
output_filename = 'RF_Actual_vs_Predicted_LOOCV.xlsx'[cite: 5]
df_results.to_excel(output_filename, index=False)[cite: 5]

print(f"\nProcess complete! True and predicted values saved to '{output_filename}'.")[cite: 5]
print("You may now execute the scatter plot generation script.")[cite: 5]