# -*- coding: utf-8 -*-
"""
Random Forest Out-of-Time Validation
Splits the dataset strictly chronologically (t <= 100 vs t > 100) to prevent temporal data leakage 
and strictly test the extrapolative capabilities of the model.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd
import numpy as np
import warnings
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Suppress warnings
warnings.filterwarnings('ignore')[cite: 7]

# 1. Fix seed for absolute reproducibility
SEED = 42[cite: 7]
os.environ['PYTHONHASHSEED'] = str(SEED)[cite: 7]
random.seed(SEED)[cite: 7]
np.random.seed(SEED)[cite: 7]

# 2. Load the dataset
df = pd.read_excel('ML_Dataset_Catalist.xlsx')[cite: 7]

# =====================================================================
# REVIEWER RESPONSE: STRICT CHRONOLOGICAL MASKED SPLITTING
# =====================================================================
# Data is split logically by a strict time threshold, not proportionally.
df_train = df[df['Reaction_Time'] <= 100].reset_index(drop=True)[cite: 7]
df_test = df[df['Reaction_Time'] > 100].reset_index(drop=True)[cite: 7]

print(f"Total Dataset Size: {len(df)} rows")[cite: 7]
print(f"Training Set (t <= 100 min): {len(df_train)} rows")[cite: 7]
print(f"Testing Set (t > 100 min): {len(df_test)} rows")[cite: 7]

# Create X (Inputs) and Y (Outputs) matrices
X_train = df_train[['Cu_Ratio', 'Reaction_Time']].values[cite: 7]
Y_train = df_train[['Efficiency', 'Ct_C0', 'ln_C0_Ct']].values[cite: 7]

X_test = df_test[['Cu_Ratio', 'Reaction_Time']].values[cite: 7]
Y_test = df_test[['Efficiency', 'Ct_C0', 'ln_C0_Ct']].values[cite: 7]

# 3. Input Scaling
scaler_X = StandardScaler()[cite: 7]
X_train_scaled = scaler_X.fit_transform(X_train)[cite: 7]
X_test_scaled = scaler_X.transform(X_test)[cite: 7]

# Y outputs are NOT scaled; physical dimensions are preserved.

# 4. Model Training (Random Forest)
rf_model = RandomForestRegressor(n_estimators=100, random_state=SEED)[cite: 7]
rf_model.fit(X_train_scaled, Y_train)[cite: 7]

# 5. Forecasting Future Boundary Conditions
Y_pred = rf_model.predict(X_test_scaled)[cite: 7]

# 6. Calculate Error Metrics for each output
r2_raw = r2_score(Y_test, Y_pred, multioutput='raw_values')[cite: 7]
mae_raw = mean_absolute_error(Y_test, Y_pred, multioutput='raw_values')[cite: 7]
mse_raw = mean_squared_error(Y_test, Y_pred, multioutput='raw_values')[cite: 7]

# Tabulate the results
results = []
for i, target in enumerate(['Efficiency (%)', 'Ct_C0', 'ln(C0/Ct)']):
    results.append({
        'Validation Strategy': 'Chronological (t <= 100 min -> t > 100 min)',
        'Output Variable': target,
        'R² Score': round(r2_raw[i], 4),
        'MAE': round(mae_raw[i], 4),
        'MSE': round(mse_raw[i], 4)
    })[cite: 7]

df_res = pd.DataFrame(results)[cite: 7]
print("\n--- Out-of-Time (Chronological) Validation Results ---")[cite: 7]
print(df_res.to_string(index=False))[cite: 7]

# Save to Excel
output_filename = 'RF_Out_Of_Time_Validation_Fixed.xlsx'[cite: 7]
df_res.to_excel(output_filename, index=False)[cite: 7]
print(f"\nResults successfully saved to '{output_filename}'.")[cite: 7]