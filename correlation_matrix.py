# -*- coding: utf-8 -*-
"""
Correlation Matrix Generator
Calculates and visualizes the Pearson correlation matrix for input and output variables.

Author: Prof. Dr. Metin Zontul
Date: 17 July 2026
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_excel('ML_Dataset_Catalist.xlsx')[cite: 3]

# 2. Calculate the correlation matrix (Pearson is default)
correlation_matrix = df.corr()[cite: 3]

# 3. Visualization settings
plt.figure(figsize=(8, 6))[cite: 3]

# Generate the heatmap
sns.heatmap(
    correlation_matrix, 
    annot=True,         # Display numerical values inside the boxes
    cmap='coolwarm',    # Color palette (Red for positive, Blue for negative)
    fmt=".2f",          # Show up to 2 decimal places
    vmin=-1,            # Minimum scale value
    vmax=1              # Maximum scale value
)[cite: 3]

# Set title and layout
plt.title('Correlation Matrix: Relationships Between Input and Output Variables', pad=15)[cite: 3]
plt.tight_layout()[cite: 3]

# Save and display the figure
plt.savefig('Correlation_Matrix.png', dpi=600)[cite: 3]
plt.show()[cite: 3]

# Print the numerical matrix to the console
print("--- Numerical Correlation Matrix ---")[cite: 3]
print(correlation_matrix)[cite: 3]