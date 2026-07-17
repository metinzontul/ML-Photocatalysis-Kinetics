# -*- coding: utf-8 -*-
"""
Data Preprocessing Script
Merges raw wide-format experimental data into a unified long-format ML dataset.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd

# 1. Read the Ct_C0.xlsx file (skipping the first two rows)
df_ct = pd.read_excel('Ct_C0.xlsx', skiprows=2, names=['Reaction_Time', '0', '3', '5', '7.5', '10'])[cite: 4]
df_ct_long = df_ct.melt(id_vars=['Reaction_Time'], var_name='Cu_Ratio', value_name='Ct_C0')[cite: 4]

# 2. Read the ln_C0_Ct.xlsx file
df_ln = pd.read_excel('ln C0_Ct.xlsx', skiprows=1, names=['Reaction_Time', '0', '3', '5', '7.5', '10'])[cite: 4]
df_ln_long = df_ln.melt(id_vars=['Reaction_Time'], var_name='Cu_Ratio', value_name='ln_C0_Ct')[cite: 4]

# 3. Read the Photocatalytic degradation.xlsx file
df_deg = pd.read_excel('Photocatalytic degradation.xlsx', skiprows=1, names=['Cu_Ratio', 'Reaction_Time', 'Efficiency'])[cite: 4]

# Ensure consistent numeric data types for merging
df_ct_long['Cu_Ratio'] = df_ct_long['Cu_Ratio'].astype(float)[cite: 4]
df_ln_long['Cu_Ratio'] = df_ln_long['Cu_Ratio'].astype(float)[cite: 4]
df_deg['Cu_Ratio'] = df_deg['Cu_Ratio'].astype(float)[cite: 4]

# 4. Merge all data into a single DataFrame using outer join
df_merged = pd.merge(df_deg, df_ct_long, on=['Cu_Ratio', 'Reaction_Time'], how='outer')[cite: 4]
df_merged = pd.merge(df_merged, df_ln_long, on=['Cu_Ratio', 'Reaction_Time'], how='outer')[cite: 4]

# Reorder columns logically (Inputs -> Outputs)
df_merged = df_merged[['Cu_Ratio', 'Reaction_Time', 'Efficiency', 'Ct_C0', 'ln_C0_Ct']][cite: 4]

# 5. Export the merged dataset to a new Excel file
output_filename = 'ML_Dataset_Catalist.xlsx'[cite: 4]
df_merged.to_excel(output_filename, index=False)[cite: 4]

print(f"Dataset successfully saved as '{output_filename}' in the working directory.")[cite: 4]
print(f"Total Rows: {df_merged.shape[0]}, Total Columns: {df_merged.shape[1]}")[cite: 4]