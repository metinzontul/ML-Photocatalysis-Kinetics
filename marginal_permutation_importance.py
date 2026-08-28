#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cross-validated marginal permutation importance for the AC@NiO dataset.

The analysis uses Ct/C0 as the primary response because degradation efficiency
and ln(C0/Ct) are deterministic transformations of the same measurement.

Design
------
* Random Forest configuration: 100 trees, max_depth=5, min_samples_leaf=1.
* Ten controlled RF seeds: 0, ..., 9.
* Shuffled five-fold cross-validation, matching the manuscript's within-domain
  validation protocol. The same seed controls fold construction and RF fitting.
* Fifty independent feature permutations within each validation fold.
* Importance metrics:
    - decrease in validation R^2;
    - increase in validation MAE (implemented with neg_mean_absolute_error).

The script first averages the fold/repetition results within each RF seed and
then reports the mean and sample standard deviation across the ten seed-level
estimates. These intervals quantify computational/model variability, not
experimental uncertainty.
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
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold


FEATURE_COLUMNS = ["Cu_Loading", "Reaction_Time"]
FEATURE_LABELS = {
    "Cu_Loading": "Nominal Cu loading",
    "Reaction_Time": "Reaction time",
}
RF_SEEDS = tuple(range(10))
N_SPLITS = 5
N_PERMUTATIONS = 50
FIGURE_DPI = 600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate cross-validated marginal permutation importance for "
            "Ct/C0 using 10 RF seeds and 50 permutations per fold."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=SCRIPT_DIR / "catalyst_dataset(1).csv",
        help="Semicolon-separated catalyst dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Directory for CSV summaries and figures.",
    )
    return parser.parse_args()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Load the source CSV without modifying the original measurements."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

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

    duplicate_cells = analysis_df.duplicated(
        subset=["Cu_Loading", "Reaction_Time"]
    ).sum()
    if duplicate_cells:
        raise ValueError(
            f"Found {duplicate_cells} duplicated Cu-loading/time cells."
        )

    if analysis_df["Reaction_Time"].nunique() < N_SPLITS:
        raise ValueError(
            f"At least {N_SPLITS} distinct reaction times are required."
        )

    return analysis_df.sort_values(
        ["Reaction_Time", "Cu_Loading"]
    ).reset_index(drop=True)


def make_rf(seed: int) -> RandomForestRegressor:
    """Create the RF configuration used in the manuscript."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=None,
    )


def calculate_marginal_pfi(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return fold-level, seed-level, and final PFI summaries."""
    X = df.loc[:, FEATURE_COLUMNS]
    y = df["Ct_C0"]
    fold_records: list[dict[str, float | int | str]] = []
    seed_records: list[dict[str, float | int | str]] = []

    for seed in RF_SEEDS:
        per_seed = {
            "R2 decrease": {feature: [] for feature in FEATURE_COLUMNS},
            "MAE increase": {feature: [] for feature in FEATURE_COLUMNS},
        }

        # This intentionally matches the manuscript's shuffled within-domain
        # five-fold CV. It is not a chronological or out-of-time evaluation.
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        for fold_index, (train_index, validation_index) in enumerate(
            cv.split(X), start=1
        ):
            X_train = X.iloc[train_index]
            X_validation = X.iloc[validation_index]
            y_train = y.iloc[train_index]
            y_validation = y.iloc[validation_index]

            model = make_rf(seed)
            model.fit(X_train, y_train)
            validation_prediction = model.predict(X_validation)

            baseline_r2 = r2_score(y_validation, validation_prediction)
            baseline_mae = mean_absolute_error(
                y_validation, validation_prediction
            )

            permutation_seed = 100_000 + 1_000 * seed + fold_index
            pfi_r2 = permutation_importance(
                model,
                X_validation,
                y_validation,
                scoring="r2",
                n_repeats=N_PERMUTATIONS,
                random_state=permutation_seed,
                n_jobs=None,
            )
            # For a negative-MAE scorer, sklearn's importance equals
            # permuted MAE minus baseline MAE, i.e. the increase in MAE.
            pfi_mae = permutation_importance(
                model,
                X_validation,
                y_validation,
                scoring="neg_mean_absolute_error",
                n_repeats=N_PERMUTATIONS,
                random_state=permutation_seed,
                n_jobs=None,
            )

            for feature_index, feature_name in enumerate(FEATURE_COLUMNS):
                metric_arrays = {
                    "R2 decrease": pfi_r2.importances[feature_index],
                    "MAE increase": pfi_mae.importances[feature_index],
                }
                for metric_name, importance_values in metric_arrays.items():
                    per_seed[metric_name][feature_name].extend(
                        importance_values.tolist()
                    )
                    fold_records.append(
                        {
                            "seed": seed,
                            "fold": fold_index,
                            "feature": FEATURE_LABELS[feature_name],
                            "metric": metric_name,
                            "baseline_r2": baseline_r2,
                            "baseline_mae": baseline_mae,
                            "importance_mean": float(
                                np.mean(importance_values)
                            ),
                            "importance_sd_across_50_permutations": float(
                                np.std(importance_values, ddof=1)
                            ),
                        }
                    )

        for metric_name, feature_values in per_seed.items():
            for feature_name, importance_values in feature_values.items():
                seed_records.append(
                    {
                        "seed": seed,
                        "feature": FEATURE_LABELS[feature_name],
                        "metric": metric_name,
                        "seed_level_importance": float(
                            np.mean(importance_values)
                        ),
                    }
                )

    fold_details = pd.DataFrame(fold_records)
    seed_summary = pd.DataFrame(seed_records)
    final_summary = (
        seed_summary.groupby(["metric", "feature"], as_index=False)[
            "seed_level_importance"
        ]
        .agg(mean="mean", sd="std", minimum="min", maximum="max")
        .sort_values(["metric", "feature"])
        .reset_index(drop=True)
    )
    final_summary["n_rf_seeds"] = len(RF_SEEDS)
    final_summary["permutations_per_fold"] = N_PERMUTATIONS
    final_summary["cv_folds"] = N_SPLITS

    return fold_details, seed_summary, final_summary


def plot_summary(final_summary: pd.DataFrame, output_path: Path) -> None:
    """Create a compact, journal-ready two-panel PFI figure.

    Panel identifiers are deliberately placed above the axes rather than
    inside the plotting area so that they cannot be confused with bars,
    error bars, or tick labels. The figure title and methodological note are
    left to the manuscript caption, as recommended for a compact two-column
    journal layout.
    """
    feature_order = [
        FEATURE_LABELS["Cu_Loading"],
        FEATURE_LABELS["Reaction_Time"],
    ]
    metrics = ["R2 decrease", "MAE increase"]
    colors = ["#1F4E79", "#2F6F5E"]
    x_labels = [
        r"Mean decrease in validation $R^2$",
        r"Mean increase in validation MAE",
    ]

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
        # Approximately 180 mm wide: suitable for a two-column MDPI figure.
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.75))
        y_positions = np.arange(len(feature_order))

        for panel_index, (axis, metric, color, x_label) in enumerate(
            zip(axes, metrics, colors, x_labels)
        ):
            metric_table = (
                final_summary.loc[final_summary["metric"] == metric]
                .set_index("feature")
                .reindex(feature_order)
            )
            means = metric_table["mean"].to_numpy(dtype=float)
            standard_deviations = metric_table["sd"].to_numpy(dtype=float)

            axis.barh(
                y_positions,
                means,
                xerr=standard_deviations,
                color=color,
                alpha=0.9,
                height=0.55,
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
            )
            axis.set_yticks(y_positions)
            axis.set_yticklabels(feature_order)
            axis.invert_yaxis()
            axis.set_xlabel(x_label)
            axis.axvline(0.0, color="#666666", linewidth=0.8)
            axis.grid(
                axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8
            )
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

        # Keep all explanatory prose in the manuscript caption, not in the
        # plotting area. Extra horizontal space prevents long feature labels
        # from colliding with the neighbouring panel.
        fig.subplots_adjust(
            left=0.185,
            right=0.985,
            bottom=0.235,
            top=0.84,
            wspace=0.72,
        )
        fig.savefig(
            output_path,
            dpi=FIGURE_DPI,
            facecolor="white",
        )
        fig.savefig(output_path.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_and_validate_data(args.data.resolve())
    fold_details, seed_summary, final_summary = calculate_marginal_pfi(df)

    fold_details.to_csv(
        output_dir / "marginal_pfi_fold_details.csv", index=False
    )
    seed_summary.to_csv(
        output_dir / "marginal_pfi_seed_summary.csv", index=False
    )
    final_summary.to_csv(
        output_dir / "marginal_pfi_final_summary.csv", index=False
    )

    figure_path = output_dir / "Figure_Marginal_PFI_600dpi.png"
    plot_summary(final_summary, figure_path)

    print("\nCross-validated marginal permutation importance")
    print("Primary response: Ct/C0")
    print("Validation: shuffled five-fold within-domain CV")
    print("RF seeds: 0-9; permutations per fold: 50\n")
    print(final_summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
