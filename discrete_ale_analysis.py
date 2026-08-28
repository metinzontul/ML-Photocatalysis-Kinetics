#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discrete first-order ALE analysis for the AC@NiO catalyst dataset.

The Random Forest predicts Ct/C0 from nominal Cu-loading and reaction time.
Both predictors are observed on discrete experimental grids:

* nominal Cu-loading: 0, 3, 5, 7.5, and 10%;
* reaction time: 0, 10, ..., 120 min.

No intermediate Cu-loading values are generated. For each ordered predictor,
the local prediction difference between two consecutive observed levels is
evaluated using observations at the upper interval endpoint. The local effects
are averaged, accumulated across intervals, and centered to have a weighted
mean of zero over the empirical predictor distribution. Because the dataset is
a complete and balanced Cu-loading-by-time grid, the distribution of the other
predictor is identical at every observed level.

The calculation is repeated for 10 controlled Random Forest seeds. Curves show
the mean discrete ALE effect across seeds; bands denote +/- 1 sample standard
deviation across seeds. These bands quantify model-seed variability, not
experimental or bootstrap uncertainty.
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


FEATURE_COLUMNS = ["Cu_Loading", "Reaction_Time"]
FEATURE_LABELS = {
    "Cu_Loading": "Nominal Cu-loading",
    "Reaction_Time": "Reaction time",
}
EXPECTED_CU_LEVELS = np.array([0.0, 3.0, 5.0, 7.5, 10.0])
RF_SEEDS = tuple(range(10))
FIGURE_DPI = 600


def default_dataset_path() -> Path:
    """Find a catalyst dataset placed next to this script."""
    preferred_names = ["catalyst_dataset.csv", "catalyst_dataset(1).csv"]
    for filename in preferred_names:
        candidate = SCRIPT_DIR / filename
        if candidate.exists():
            return candidate

    candidates = sorted(SCRIPT_DIR.glob("catalyst_dataset*.csv"))
    if len(candidates) == 1:
        return candidates[0]

    return SCRIPT_DIR / "catalyst_dataset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate discrete first-order ALE curves for nominal "
            "Cu-loading and reaction time."
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
        default=SCRIPT_DIR / "ale_results",
        help="Directory for ALE tables and publication figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively (useful in Spyder).",
    )
    return parser.parse_args()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Load the source measurements and verify the factorial grid."""
    if not data_path.exists():
        candidates = sorted(SCRIPT_DIR.glob("*.csv"))
        available = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"CSV files next to the script: {available}\n"
            "Place catalyst_dataset.csv next to this script or specify the "
            "complete path with --data."
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

    observed_cu_levels = np.sort(
        analysis_df["Cu_Loading"].unique().astype(float)
    )
    if not np.array_equal(observed_cu_levels, EXPECTED_CU_LEVELS):
        raise ValueError(
            "The observed nominal Cu-loading levels must be exactly "
            f"{EXPECTED_CU_LEVELS.tolist()}, but found "
            f"{observed_cu_levels.tolist()}."
        )

    n_cu = analysis_df["Cu_Loading"].nunique()
    n_time = analysis_df["Reaction_Time"].nunique()
    expected_cells = n_cu * n_time
    if len(analysis_df) != expected_cells:
        raise ValueError(
            "Discrete ALE requires the complete Cu-loading-by-time grid. "
            f"Found {len(analysis_df)} of {expected_cells} cells."
        )

    counts_by_time = analysis_df.groupby("Reaction_Time").size()
    counts_by_cu = analysis_df.groupby("Cu_Loading").size()
    if not (counts_by_time.eq(n_cu).all() and counts_by_cu.eq(n_time).all()):
        raise ValueError("The Cu-loading-by-time grid is not balanced.")

    return analysis_df.sort_values(FEATURE_COLUMNS).reset_index(drop=True)


def make_rf(seed: int) -> RandomForestRegressor:
    """Create the Random Forest configuration used in the manuscript."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=None,
    )


def calculate_discrete_ale(
    model: RandomForestRegressor,
    X: pd.DataFrame,
    feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate centered first-order ALE at observed discrete levels.

    For interval (z[k-1], z[k]], observations at z[k] define the empirical
    distribution of the other predictor. Predictions are evaluated after
    replacing the selected feature by the lower and upper interval endpoints.
    The averaged local differences are accumulated and centered.
    """
    levels = np.sort(X[feature].unique().astype(float))
    local_effects = np.zeros(len(levels), dtype=float)
    interval_records: list[dict[str, float | int | str]] = []

    for level_index in range(1, len(levels)):
        lower_level = levels[level_index - 1]
        upper_level = levels[level_index]

        interval_rows = X.loc[X[feature] == upper_level].copy()
        X_lower = interval_rows.copy()
        X_upper = interval_rows.copy()
        X_lower[feature] = lower_level
        X_upper[feature] = upper_level

        prediction_difference = (
            model.predict(X_upper) - model.predict(X_lower)
        )
        mean_local_effect = float(np.mean(prediction_difference))
        local_effects[level_index] = mean_local_effect

        interval_records.append(
            {
                "feature": FEATURE_LABELS[feature],
                "lower_level": lower_level,
                "upper_level": upper_level,
                "interval_n": len(interval_rows),
                "mean_local_effect": mean_local_effect,
                "sd_local_prediction_difference": float(
                    np.std(prediction_difference, ddof=1)
                ),
            }
        )

    accumulated_effect = np.cumsum(local_effects)
    level_counts = (
        X[feature]
        .value_counts()
        .reindex(levels, fill_value=0)
        .to_numpy(dtype=float)
    )
    centering_constant = float(
        np.average(accumulated_effect, weights=level_counts)
    )
    centered_ale = accumulated_effect - centering_constant

    ale_table = pd.DataFrame(
        {
            "feature": FEATURE_LABELS[feature],
            "level": levels,
            "level_n": level_counts.astype(int),
            "accumulated_effect_before_centering": accumulated_effect,
            "centering_constant": centering_constant,
            "ale_effect": centered_ale,
        }
    )
    interval_table = pd.DataFrame(interval_records)
    return ale_table, interval_table


def run_multi_seed_ale(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit ten RF models and summarize discrete ALE across seeds."""
    X = df.loc[:, FEATURE_COLUMNS]
    y = df["Ct_C0"]
    seed_tables: list[pd.DataFrame] = []
    interval_tables: list[pd.DataFrame] = []

    for seed in RF_SEEDS:
        model = make_rf(seed)
        model.fit(X, y)

        for feature in FEATURE_COLUMNS:
            ale_table, interval_table = calculate_discrete_ale(
                model, X, feature
            )
            ale_table.insert(0, "seed", seed)
            interval_table.insert(0, "seed", seed)
            seed_tables.append(ale_table)
            interval_tables.append(interval_table)

    seed_values = pd.concat(seed_tables, ignore_index=True)
    interval_details = pd.concat(interval_tables, ignore_index=True)
    summary = (
        seed_values.groupby(["feature", "level"], as_index=False)["ale_effect"]
        .agg(mean="mean", sd="std", minimum="min", maximum="max")
        .sort_values(["feature", "level"])
        .reset_index(drop=True)
    )
    summary["n_rf_seeds"] = len(RF_SEEDS)
    return seed_values, interval_details, summary


def plot_ale(
    summary: pd.DataFrame,
    output_path: Path,
    show_figure: bool,
) -> None:
    """Create the two-panel Catalysts-ready ALE figure."""
    panels = [
        (
            "Nominal Cu-loading",
            "Nominal Cu-loading (%)",
            "#2F5D8A",
        ),
        (
            "Reaction time",
            "Reaction time (min)",
            "#2F6F5E",
        ),
    ]

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True)

        for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
            feature_label, x_label, color = panel
            table = (
                summary.loc[summary["feature"] == feature_label]
                .sort_values("level")
                .reset_index(drop=True)
            )
            x_values = table["level"].to_numpy(dtype=float)
            means = table["mean"].to_numpy(dtype=float)
            standard_deviations = table["sd"].to_numpy(dtype=float)

            axis.errorbar(
                x_values,
                means,
                yerr=standard_deviations,
                color=color,
                ecolor=color,
                marker="o",
                markersize=4.2,
                linewidth=1.5,
                elinewidth=0.8,
                capsize=2.2,
                capthick=0.8,
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=3,
            )
            axis.fill_between(
                x_values,
                means - standard_deviations,
                means + standard_deviations,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )
            axis.axhline(0.0, color="#666666", linewidth=0.8, linestyle="--")
            axis.set_xlabel(x_label)
            axis.grid(
                axis="both",
                color="#D9D9D9",
                linewidth=0.6,
                alpha=0.75,
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

            if feature_label == "Nominal Cu-loading":
                axis.set_xticks(EXPECTED_CU_LEVELS)
                axis.set_xticklabels(["0", "3", "5", "7.5", "10"])
            else:
                axis.set_xticks(np.arange(0, 121, 20))

        axes[0].set_ylabel(r"ALE effect on predicted $C_t/C_0$")
        fig.subplots_adjust(
            left=0.13,
            right=0.985,
            bottom=0.22,
            top=0.82,
            wspace=0.24,
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
    seed_values, interval_details, summary = run_multi_seed_ale(df)

    seed_values.to_csv(output_dir / "ale_seed_values.csv", index=False)
    interval_details.to_csv(
        output_dir / "ale_local_interval_effects.csv", index=False
    )
    summary.to_csv(output_dir / "ale_summary.csv", index=False)

    figure_path = output_dir / "Figure_Discrete_ALE_Ct_C0_600dpi.png"
    plot_ale(summary, figure_path, args.show)

    print("\nDiscrete first-order ALE analysis")
    print("Primary response: Ct/C0")
    print("RF seeds: 0-9")
    print("Nominal Cu-loading levels: 0, 3, 5, 7.5, and 10%\n")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
