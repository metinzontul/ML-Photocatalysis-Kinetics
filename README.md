# Explainable Machine Learning for Methylene Blue Removal Under Irradiation: Assessing Nominal Cu-Loading in AC@NiO
This repository contains the dataset and Python source code for modeling the time-dependent photocatalytic degradation kinetics of AC@NiO catalysts prepared with nominal Cu precursor loadings using a tree-based machine learning framework (Random Forest)[cite: 15]. 

The methodology strictly enforces physical scaling algebraically and utilizes a rigorous Out-of-Time (chronological) validation strategy to prevent temporal data leakage[cite: 15]. It also includes Explainable AI (XAI) tools to decode the macroscopic kinetic drivers of the reaction[cite: 15].

## 🚀 Key Features

*   **Algebraic Consistency:** Target variables (Degradation Efficiency, $C_t/C_0$, and $\ln(C_0/C_t)$) are modeled on their native physical scales without arbitrary MinMax scaling[cite: 15].
*   **Out-of-Time Validation:** To prove genuine extrapolative learning rather than temporal interpolation, models are trained exclusively on historical data ($t \le 100$ min) and tested on entirely unseen future states ($t > 100$ min)[cite: 15].
*   **Strict Regularization:** The Random Forest algorithm is regularized (e.g., `max_depth=5`, `min_samples_leaf=1`) to prevent overfitting and memorization of the limited experimental sample space.
*   **Unbiased Explainable AI (XAI):** Disentangles the dominant macroscopic impact of reaction time from the influence of the nominal Cu-loading using SHAP (SHapley Additive exPlanations) values, Impurity-based Feature Importance (MDI), and Out-of-Fold Permutation Feature Importance to eliminate cardinality bias[cite: 15].
*   **Advanced Diagnostics:** Includes Partial Dependence Plots (PDP), Accumulated Local Effects (ALE), discrete 2D response heatmaps, and a moving-block bootstrap stability analysis to quantify model-seed resampling variability.

## 📂 Repository Structure
*   `data_preprocessing.py`: Merges raw wide-format experimental Excel files into a unified long-format ML dataset.
*   `correlation_matrix.py`: Generates the Pearson correlation matrix for input and output variables.
*   `rf_multiseed_robustness.py`: Evaluates the Random Forest model across 10 random seeds using K-Fold, LOOCV, and GroupKFold, reporting physical MAE and MSE.
*   `rf_out_of_time_validation.py`: Performs strict chronological splitting to test future boundary forecasting capabilities.
*   `rf_loocv_predictions.py`: Runs a Leave-One-Out Cross-Validation to generate unscaled `y_true` and `y_pred` coordinates for scatter plotting.
*   `ml_photocatalysis_pipeline.py`: The core ML pipeline executing Random Forest regression, target-specific robustness evaluations (5-Fold CV, LOOCV), chronological diagnostics, SHAP analysis, MDI, Permutation Feature        Importance, and PDP generation.
*   `discrete_ale_analysis.py`: Computes and visualizes discrete first-order Accumulated Local Effects (ALE) curves for nominal Cu-loading and reaction time.
*   `marginal_permutation_importance.py`: Calculates cross-validated marginal permutation feature importance using shuffled five-fold validation to evaluate the decrease in $R^2$ and increase in MAE.
*   `marginal_vs_conditional_permutation_importance.py`: Compares marginal permutation importance with grouped-conditional permutation importance.
*   `model_based_bootstrap_stability.py`: Performs a synchronized moving-block bootstrap stability analysis over 1000 refits to quantify the resampling stability of MDI and PFI estimates.
*   `rf_response_heatmap.py`: Generates a discrete, unsmoothed two-dimensional RF response heatmap for predicted $C_t/C_0$ across the complete 5 x 13 experimental grid.
*   `xai_shap_mdi_plots.py`: Generates publication-quality (600 DPI) SHAP summary plots and stacked MDI bar charts[cite: 15].
*   `xai_pfi_bootstrap_metrics.py`: Computes the advanced feature-attribution metrics presented in Section 4.6. This script executes 10-seed cross-validated marginal and grouped-conditional Permutation Feature Importance       (PFI) and performs a 1000-repetition moving-block bootstrap stability analysis.
*   `rf_sensitivity_and_oot_evaluation.py`: The core monolithic script. It performs the rigorous 10-seed multi-metric robustness analysis (using K-Fold and LOOCV), evaluates 6 distinct tree-complexity configurations,  *  *    executes the Chronological OOT diagnostic, and automatically exports comprehensive performance metrics and tree complexities to CSV and JSON formats.
*   `phase3_xai_analysis.py`: Generates the Explainable AI (XAI) assessments, including the publication-quality SHAP summary plots, Normalized MDI percentages, and Permutation Feature Importance metrics based on the final *    regularized model.
*   `requirements.txt`: Contains the list of Python dependencies required to run the scripts.

## ⚙️ Requirements

To run the scripts, you will need Python 3.8+ and the following libraries[cite: 15]:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap joblib openpyxl
