# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 16:08:32 2026

@author: ASUS-WORKSTATION
"""

# -*- coding: utf-8 -*-
"""
Makale XAI Görselleri Çizim Kodu: SHAP ve MDI (Figure 3 & Figure 4)
Hakem revizyonlarına (Regularization ve Cu-loading terminolojisi) uygun olarak güncellenmiştir.
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

# 1. Makale formatı için grafik ayarları (Times New Roman, yüksek çözünürlük)
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

print("Makale XAI grafikleri (Figure 3 ve Figure 4) hazırlanıyor. Lütfen bekleyin...")

# 2. Veri Setini Yükle ve Ön İşleme (Yeni pipeline ile uyumlu CSV yüklemesi)
try:
    df = pd.read_csv("catalyst_dataset.csv", sep=";")
    if 'Cu_Ratio' in df.columns:
        df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
except FileNotFoundError:
    print("HATA: 'catalyst_dataset.csv' bulunamadı. Lütfen dosyanın aynı dizinde olduğundan emin olun.")
    exit()

X = df[['Cu_Loading', 'Reaction_Time']]

# Hakem 1'in düzeltmesi: "Cu Doping" yerine "Cu-loading"
feature_names = ['Cu-loading Ratio (%)', 'Reaction Time (min)']

# Girdileri Ölçekle (Modelin stabilite kazanması için X standartlaştırılır, Y fiziksel bırakılır)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Verim sütununu makaledeki gibi (n/100) formuna normalize ediyoruz
df['Efficiency_Norm'] = df['Efficiency'] / 100

# Çizilecek Hedefler
targets = [
    {'col': 'Efficiency_Norm', 'label': 'Degradation Efficiency', 'suffix': 'Efficiency'},
    {'col': 'Ct_C0', 'label': 'Ct/C0', 'suffix': 'Ct_C0'},
    {'col': 'ln_C0_Ct', 'label': 'ln(C0/Ct)', 'suffix': 'ln_C0_Ct'}
]

mdi_results = []

# 3. Her Bir Hedef İçin Bağımsız Model Eğitimi, MDI ve SHAP Çıkarımı
for target in targets:
    y = df[target['col']]
    
    # HAKEM 2 GÜNCELLEMESİ: Aşırı öğrenmeyi engelleyen budanmış (regularized) model
    rf_model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=5, 
        min_samples_leaf=1, 
        random_state=42,
        n_jobs=None
    )
    rf_model.fit(X_scaled_df, y)
    
    # --- A. MDI (Impurity-based Feature Importance) Çıkarımı ---
    importances = rf_model.feature_importances_ * 100 # Yüzde formatına çevir
    mdi_results.append({
        'Target': target['label'],
        'Cu-loading Ratio': importances[0],
        'Reaction Time': importances[1]
    })
    
    # --- B. SHAP Analizi (Figure 3) ---
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    plt.figure(figsize=(7, 5))
    shap.summary_plot(shap_values, X_scaled_df, feature_names=feature_names, show=False)
    
    # Makale standartlarında başlık ve etiket ayarları
    plt.title(f"SHAP Summary: Impact on {target['label']}", fontweight='bold', pad=15)
    plt.xlabel(f"SHAP Value (Impact on Model Output for {target['label']})", fontweight='bold')
    
    plt.tight_layout()
    output_shap = f"Figure_3_SHAP_{target['suffix']}_600DPI.png"
    plt.savefig(output_shap, dpi=600, bbox_inches='tight')
    plt.close()
    
    print(f" - {target['label']} için SHAP grafiği kaydedildi: {output_shap}")

# 4. MDI Feature Importance Bar Chart Çizimi (Figure 4)
print("\nMDI Yüzdelik Oranları (Figure 4) çiziliyor...")
mdi_df = pd.DataFrame(mdi_results).set_index('Target')

fig, ax = plt.subplots(figsize=(8, 6))

# Hakem düzeltmelerini yansıtacak şekilde 'stacked bar' oluştur ("Cu Doping" yerine "Cu-loading")
mdi_df[['Reaction Time', 'Cu-loading Ratio']].plot(
    kind='bar', 
    stacked=True, 
    ax=ax, 
    color=['#1f77b4', '#ff7f0e'], 
    edgecolor='black',
    width=0.5
)

# Grafik metin ve eksen ayarları
ax.set_title("Impurity-based Feature Importance (MDI)", fontweight='bold', pad=15)
ax.set_ylabel("Relative Importance (%)", fontweight='bold')
ax.set_xlabel("Predicted Model Outputs", fontweight='bold')
ax.set_ylim(0, 100)
plt.xticks(rotation=0, fontweight='bold')

# Legend'ı grafiğin altına yatay olarak yerleştir (Güncel isimlerle)
ax.legend(['Reaction Time', 'Cu-loading Ratio'], loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

# Çubukların içine yüzde değerlerini yazdır
for container in ax.containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() > 0 else '' for v in container]
    ax.bar_label(container, labels=labels, label_type='center', color='white', fontweight='bold', fontsize=12)

plt.tight_layout()
output_mdi = "Figure_4_MDI_Importance_600DPI.png"
plt.savefig(output_mdi, dpi=600, bbox_inches='tight')
plt.close()

print(f" - MDI Bar grafiği başarıyla kaydedildi: {output_mdi}")
print("\nTüm XAI grafikleri yüksek çözünürlükte hazır!")