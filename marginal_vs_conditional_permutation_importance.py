#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Marginal versus grouped-conditional permutation importance.

This script compares two permutation schemes for the Random Forest model used
to predict Ct/C0 from nominal Cu loading and reaction time:

1. Marginal permutation
   The selected feature is shuffled across all 65 observations.

2. Grouped-conditional permutation
   * Cu loading is shuffled within each fixed reaction-time stratum.
   * Reaction time is shuffled within each fixed Cu-loading stratum.

The dataset is a complete 5 x 13 factorial grid (five Cu loadings and thirteen
reaction times, one observation per cell). To avoid the very small conditional
strata that would arise inside individual validation folds, permutation is
performed on the complete feature grid and evaluated through cross-fitted
out-of-fold (OOF) predictions. Each observation is always predicted by the RF
model for which that observation belonged to the validation fold.

Analysis design
---------------
* Primary response: Ct/C0.
* Random Forest: 100 trees, max_depth=5, min_samples_leaf=1.
* Ten controlled seeds: 0, ..., 9.
* Shuffled five-fold cross-fitting for every seed.
* Fifty permutations for every feature and permutation scheme.
* Importance metrics:
    - decrease in OOF R^2;
    - increase in OOF MAE.
* Final uncertainty display: mean +/- sample SD across the ten seed-level
  estimates. This describes model-seed variability, not experimental or
  bootstrap uncertainty.

The grouped procedure is a discrete, design-based approximation to
conditional permutation importance. It should not be described as causal
importance or as a full conditional-randomization test.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


FEATURE_COLUMNS = ["Cu_Loading", "Reaction_Time"]
FEATURE_LABELS = {
    "Cu_Loading": "Nominal Cu loading",
    "Reaction_Time": "Reaction time",
}
CONDITIONING_FEATURE = {
    "Cu_Loading": "Reaction_Time",
    "Reaction_Time": "Cu_Loading",
}
SCHEME_LABELS = {
    "Marginal": "Marginal permutation",
    "Conditional": "Conditional permutation",
}

RF_SEEDS = tuple(range(10))
N_SPLITS = 5
N_PERMUTATIONS = 50
FIGURE_DPI = 600


def default_dataset_path() -> Path:
    """Find a uniquely named catalyst dataset next to the script."""
    preferred_names = [
        "catalyst_dataset.csv",
        "catalyst_dataset(1).csv",
    ]
    for filename in preferred_names:
        candidate = SCRIPT_DIR / filename
        if candidate.exists():
            return candidate

    candidates = sorted(SCRIPT_DIR.glob("catalyst_dataset*.csv"))
    if len(candidates) == 1:
        return candidates[0]

    # A helpful unresolved default is returned so load_and_validate_data can
    # produce one clear error message when the dataset is absent.
    return SCRIPT_DIR / "catalyst_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare marginal and grouped-conditional permutation importance "
            "using 10 RF seeds and 50 permutations."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=default_dataset_path(),
        help="Semicolon-separated catalyst dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "conditional_pfi_results",
        help="Directory for CSV summaries and publication figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively (useful in Spyder).",
    )
    return parser.parse_args()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Load the CSV and verify the complete factorial feature grid."""
    if not data_path.exists():
        candidates = sorted(SCRIPT_DIR.glob("*.csv"))
        available = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"CSV files next to the script: {available}\n"
            "Place catalyst_dataset.csv next to this script or run with "
            "--data followed by the complete CSV path."
        )

    df = pd.read_csv(data_path, sep=";", encoding="utf-8-sig")
    if "Cu_Ratio" in df.columns:
        df = df.rename(columns={"Cu_Ratio": "Cu_Loading"})

    required = {"Cu_Loading", "Reaction_Time", "Ct_C0"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    analysis_df = df.loc[:, ["Cu_Loading", "Reaction_Time", "Ct_C0"]].copy()
    if analysis_df.isna().any().any():
        raise ValueError("Missing values were found in the analysis columns.")

    duplicate_count = analysis_df.duplicated(FEATURE_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicated Cu-loading/time cells."
        )

    n_cu = analysis_df["Cu_Loading"].nunique()
    n_time = analysis_df["Reaction_Time"].nunique()
    expected_cells = n_cu * n_time
    if len(analysis_df) != expected_cells:
        raise ValueError(
            "Grouped-conditional permutation requires a complete Cu-by-time "
            f"grid. Found {len(analysis_df)} of {expected_cells} cells."
        )

    counts_by_time = analysis_df.groupby("Reaction_Time").size()
    counts_by_cu = analysis_df.groupby("Cu_Loading").size()
    if not (counts_by_time.eq(n_cu).all() and counts_by_cu.eq(n_time).all()):
        raise ValueError("The Cu-by-time grid is not balanced and complete.")

    return analysis_df.sort_values(FEATURE_COLUMNS).reset_index(drop=True)


def make_rf(seed: int) -> RandomForestRegressor:
    """Create the RF configuration used in the manuscript."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=seed,
        # Single-process prediction avoids repeated joblib warnings and is
        # faster for these very small validation subsets.
        n_jobs=None,
    )


def fit_cross_fitted_models(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[list[tuple[RandomForestRegressor, np.ndarray]], np.ndarray, pd.DataFrame]:
    """Fit five RF models and return their cross-fitted OOF predictions."""
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    fitted_folds: list[tuple[RandomForestRegressor, np.ndarray]] = []
    oof_prediction = np.empty(len(X), dtype=float)
    fold_records: list[dict[str, float | int]] = []

    for fold, (train_index, validation_index) in enumerate(cv.split(X), start=1):
        model = make_rf(seed)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        fold_prediction = model.predict(X.iloc[validation_index])
        oof_prediction[validation_index] = fold_prediction
        fitted_folds.append((model, validation_index))
        fold_records.append(
            {
                "seed": seed,
                "fold": fold,
                "validation_n": len(validation_index),
                "validation_r2": r2_score(
                    y.iloc[validation_index], fold_prediction
                ),
                "validation_mae": mean_absolute_error(
                    y.iloc[validation_index], fold_prediction
                ),
            }
        )

    return fitted_folds, oof_prediction, pd.DataFrame(fold_records)


def predict_cross_fitted(
    X_permuted: pd.DataFrame,
    fitted_folds: list[tuple[RandomForestRegressor, np.ndarray]],
) -> np.ndarray:
    """Predict every row with the model that did not train on that row."""
    prediction = np.empty(len(X_permuted), dtype=float)
    for model, validation_index in fitted_folds:
        prediction[validation_index] = model.predict(
            X_permuted.iloc[validation_index]
        )
    return prediction


def permute_marginally(
    X: pd.DataFrame,
    feature: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle one feature across the complete experimental grid."""
    X_permuted = X.copy()
    X_permuted[feature] = rng.permutation(X[feature].to_numpy())
    return X_permuted


def permute_conditionally(
    X: pd.DataFrame,
    feature: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle a feature only within exact strata of the other predictor."""
    condition_on = CONDITIONING_FEATURE[feature]
    X_permuted = X.copy()
    permuted_values = X[feature].to_numpy(copy=True)

    for group_indices in X.groupby(condition_on, sort=True).groups.values():
        indices = np.asarray(list(group_indices), dtype=int)
        permuted_values[indices] = rng.permutation(
            X.loc[indices, feature].to_numpy()
        )

    X_permuted[feature] = permuted_values
    return X_permuted


def calculate_importance(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Calculate repetition-, seed-, fold-, and final-level summaries."""
    X = df.loc[:, FEATURE_COLUMNS]
    y = df["Ct_C0"]

    repetition_records: list[dict[str, float | int | str]] = []
    seed_records: list[dict[str, float | int | str]] = []
    fold_tables: list[pd.DataFrame] = []
    baseline_records: list[dict[str, float | int]] = []

    permutation_functions = {
        "Marginal": permute_marginally,
        "Conditional": permute_conditionally,
    }

    for seed in RF_SEEDS:
        fitted_folds, baseline_oof, fold_table = fit_cross_fitted_models(
            X, y, seed
        )
        fold_tables.append(fold_table)

        baseline_r2 = r2_score(y, baseline_oof)
        baseline_mae = mean_absolute_error(y, baseline_oof)
        baseline_records.append(
            {
                "seed": seed,
                "oof_r2": baseline_r2,
                "oof_mae": baseline_mae,
            }
        )

        per_seed_values: dict[tuple[str, str], dict[str, list[float]]] = {}

        for scheme, permutation_function in permutation_functions.items():
            for feature_index, feature in enumerate(FEATURE_COLUMNS):
                key = (scheme, feature)
                per_seed_values[key] = {
                    "R2 decrease": [],
                    "MAE increase": [],
                }

                for repeat in range(N_PERMUTATIONS):
                    permutation_seed = (
                        1_000_000
                        + 100_000 * seed
                        + 1_000 * feature_index
                        + repeat
                    )
                    rng = np.random.default_rng(permutation_seed)
                    X_permuted = permutation_function(X, feature, rng)
                    permuted_oof = predict_cross_fitted(
                        X_permuted, fitted_folds
                    )

                    permuted_r2 = r2_score(y, permuted_oof)
                    permuted_mae = mean_absolute_error(y, permuted_oof)
                    r2_decrease = baseline_r2 - permuted_r2
                    mae_increase = permuted_mae - baseline_mae

                    per_seed_values[key]["R2 decrease"].append(r2_decrease)
                    per_seed_values[key]["MAE increase"].append(mae_increase)
                    repetition_records.append(
                        {
                            "seed": seed,
                            "scheme": scheme,
                            "feature": FEATURE_LABELS[feature],
                            "repeat": repeat + 1,
                            "baseline_oof_r2": baseline_r2,
                            "permuted_oof_r2": permuted_r2,
                            "r2_decrease": r2_decrease,
                            "baseline_oof_mae": baseline_mae,
                            "permuted_oof_mae": permuted_mae,
                            "mae_increase": mae_increase,
                        }
                    )

        for (scheme, feature), metric_values in per_seed_values.items():
            for metric, values in metric_values.items():
                seed_records.append(
                    {
                        "seed": seed,
                        "scheme": scheme,
                        "feature": FEATURE_LABELS[feature],
                        "metric": metric,
                        "seed_level_importance": float(np.mean(values)),
                        "sd_across_50_permutations": float(
                            np.std(values, ddof=1)
                        ),
                    }
                )

    repetition_details = pd.DataFrame(repetition_records)
    seed_summary = pd.DataFrame(seed_records)
    fold_details = pd.concat(fold_tables, ignore_index=True)
    baseline_summary = pd.DataFrame(baseline_records)

    final_summary = (
        seed_summary.groupby(
            ["metric", "feature", "scheme"], as_index=False
        )["seed_level_importance"]
        .agg(mean="mean", sd="std", minimum="min", maximum="max")
        .sort_values(["metric", "feature", "scheme"])
        .reset_index(drop=True)
    )
    final_summary["n_rf_seeds"] = len(RF_SEEDS)
    final_summary["permutations_per_seed"] = N_PERMUTATIONS
    final_summary["cv_folds"] = N_SPLITS

    return repetition_details, seed_summary, fold_details, final_summary, baseline_summary


def plot_comparison(
    final_summary: pd.DataFrame,
    output_path: Path,
    show_figure: bool,
) -> None:
    """Create a compact two-panel marginal-versus-conditional comparison."""
    feature_order = [
        FEATURE_LABELS["Cu_Loading"],
        FEATURE_LABELS["Reaction_Time"],
    ]
    schemes = ["Marginal", "Conditional"]
    metrics = ["R2 decrease", "MAE increase"]
    x_labels = [
        r"Mean decrease in OOF $R^2$",
        "Mean increase in OOF MAE",
    ]
    colors = {
        "Marginal": "#2F5D8A",
        "Conditional": "#C06C3B",
    }

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05))
        y_positions = np.arange(len(feature_order), dtype=float)
        bar_height = 0.27
        offsets = {"Marginal": -bar_height / 2, "Conditional": bar_height / 2}

        legend_handles = []
        legend_labels = []

        for panel_index, (axis, metric, x_label) in enumerate(
            zip(axes, metrics, x_labels)
        ):
            for scheme in schemes:
                table = (
                    final_summary.loc[
                        (final_summary["metric"] == metric)
                        & (final_summary["scheme"] == scheme)
                    ]
                    .set_index("feature")
                    .reindex(feature_order)
                )
                means = table["mean"].to_numpy(dtype=float)
                standard_deviations = table["sd"].to_numpy(dtype=float)
                bars = axis.barh(
                    y_positions + offsets[scheme],
                    means,
                    xerr=standard_deviations,
                    height=bar_height,
                    color=colors[scheme],
                    alpha=0.92,
                    capsize=2.5,
                    error_kw={"elinewidth": 0.9, "capthick": 0.9},
                    label=SCHEME_LABELS[scheme],
                )
                if panel_index == 0:
                    legend_handles.append(bars)
                    legend_labels.append(SCHEME_LABELS[scheme])

            axis.set_yticks(y_positions)
            axis.set_yticklabels(feature_order)
            axis.invert_yaxis()
            axis.set_xlabel(x_label)
            axis.axvline(0.0, color="#666666", linewidth=0.8)
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.set_axisbelow(True)
            axis.text(
                0.0,
                1.055,
                f"({chr(97 + panel_index)})",
                transform=axis.transAxes,
                ha="left",
                va="bottom",
                fontweight="bold",
                fontsize=9.5,
                clip_on=False,
            )

        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=2,
            frameon=False,
            fontsize=8.5,
            handlelength=2.3,
            columnspacing=1.8,
        )
        fig.subplots_adjust(
            left=0.185,
            right=0.985,
            bottom=0.215,
            top=0.77,
            wspace=0.72,
        )
        fig.savefig(output_path, dpi=FIGURE_DPI, facecolor="white")
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")

        if show_figure:
            plt.show()
        else:
            plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_data(args.data.resolve())
    (
        repetition_details,
        seed_summary,
        fold_details,
        final_summary,
        baseline_summary,
    ) = calculate_importance(df)

    repetition_details.to_csv(
        output_dir / "permutation_repetition_details.csv", index=False
    )
    seed_summary.to_csv(
        output_dir / "permutation_seed_summary.csv", index=False
    )
    fold_details.to_csv(
        output_dir / "cross_fitted_fold_performance.csv", index=False
    )
    baseline_summary.to_csv(
        output_dir / "cross_fitted_seed_performance.csv", index=False
    )
    final_summary.to_csv(
        output_dir / "marginal_vs_conditional_pfi_summary.csv", index=False
    )

    figure_path = output_dir / "Figure_Marginal_vs_Conditional_PFI_600dpi.png"
    plot_comparison(final_summary, figure_path, args.show)

    print("\nMarginal versus grouped-conditional permutation importance")
    print("Primary response: Ct/C0")
    print("Evaluation: cross-fitted OOF predictions from shuffled five-fold CV")
    print("RF seeds: 0-9; permutations per scheme/feature/seed: 50\n")
    print(final_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nConditional scheme:")
    print("  Cu loading shuffled within fixed reaction time")
    print("  Reaction time shuffled within fixed Cu loading")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
