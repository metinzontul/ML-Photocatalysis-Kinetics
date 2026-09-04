# -*- coding: utf-8 -*-
"""
Phase I-A: Candidate-Model Comparison
Evaluates RF, KNN, MLP, and SVR for the primary screening target Ct/C0
across 10 controlled random seeds using 5-Fold CV and LOOCV.

Author: Metin Zontul
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred)
    }

def main():
    print("Loading dataset...")
    try:
        df = pd.read_csv("catalyst_dataset.csv", sep=";")
        if 'Cu_Ratio' in df.columns:
            df.rename(columns={'Cu_Ratio': 'Cu_Loading'}, inplace=True)
    except FileNotFoundError:
        print("Dataset not found. Please provide 'catalyst_dataset.csv'.")
        return

    X = df[['Cu_Loading', 'Reaction_Time']].values
    y = df['Ct_C0'].values

    models = {
        'RF': RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42),
        'KNN': Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsRegressor(n_neighbors=5))]),
        'MLP': Pipeline([('scaler', StandardScaler()), ('mlp', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))]),
        'SVR': Pipeline([('scaler', StandardScaler()), ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))])
    }

    seeds = list(range(10))
    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name}...")
        r2_5f_list, r2_loo_list = [], []
        
        for seed in seeds:
            # Set seed for models that have random_state
            if name in ['RF', 'MLP']:
                if name == 'RF':
                    model.set_params(random_state=seed)
                else:
                    model.named_steps['mlp'].set_params(random_state=seed)
            
            # 5-Fold CV
            cv_5fold = KFold(n_splits=5, shuffle=True, random_state=seed)
            preds_5f = cross_val_predict(model, X, y, cv=cv_5fold, n_jobs=-1)
            r2_5f_list.append(r2_score(y, preds_5f))
            
            # LOOCV
            loo = LeaveOneOut()
            preds_loo = cross_val_predict(model, X, y, cv=loo, n_jobs=-1)
            r2_loo_list.append(r2_score(y, preds_loo))

        print(f"5-Fold CV R2: {np.mean(r2_5f_list):.4f} +/- {np.std(r2_5f_list, ddof=1):.4f}")
        print(f"LOOCV R2: {np.mean(r2_loo_list):.4f} +/- {np.std(r2_loo_list, ddof=1):.4f}")

if __name__ == '__main__':
    main()