# Explainable Machine Learning Framework for Photocatalytic Degradation Kinetics

This repository contains the dataset and Python source code for modeling the non-linear, time-dependent photocatalytic degradation kinetics of Cu-doped AC@NiO catalysts using a tree-based machine learning framework (Random Forest). 

The methodology strictly enforces physical scaling algebraically and utilizes a rigorous Out-of-Time (chronological) validation strategy to prevent temporal data leakage. It also includes Explainable AI (XAI) tools to decode the micro-kinetic drivers of the reaction.

## 🚀 Key Features

*   **Algebraic Consistency:** Target variables (Degradation Efficiency, $C_t/C_0$, and $\ln(C_0/C_t)$) are modeled on their native physical scales without arbitrary MinMax scaling.
*   **Out-of-Time Validation:** To prove genuine extrapolative learning rather than temporal interpolation, models are trained exclusively on historical data ($t \le 100$ min) and tested on entirely unseen future states ($t > 100$ min).
*   **Explainable AI (XAI):** Disentangles the dominant macroscopic impact of reaction time from the structural influence of the Cu doping ratio using both SHAP (SHapley Additive exPlanations) values and Impurity-based Feature Importance (MDI).

## 📂 Repository Structure

*   `data_preprocessing.py`: Merges raw wide-format experimental Excel files into a unified long-format ML dataset.
*   `correlation_matrix.py`: Generates the Pearson correlation matrix for input and output variables.
*   `rf_multiseed_robustness.py`: Evaluates the Random Forest model across 10 random seeds using K-Fold, LOOCV, and GroupKFold, reporting physical MAE and MSE.
*   `rf_out_of_time_validation.py`: Performs strict chronological splitting to test future boundary forecasting capabilities.
*   `rf_loocv_predictions.py`: Runs a Leave-One-Out Cross-Validation to generate unscaled `y_true` and `y_pred` coordinates for scatter plotting.
*   `xai_shap_mdi_plots.py`: Generates publication-quality (600 DPI) SHAP summary plots and stacked MDI bar charts.

## ⚙️ Requirements

To run the scripts, you will need Python 3.8+ and the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap openpyxl