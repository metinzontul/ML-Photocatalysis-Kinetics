#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Discrete two-dimensional RF response map for predicted Ct/C0.

The heat map contains only the experimentally observed 5 x 13 grid:

* nominal Cu-loading: 0, 3, 5, 7.5, and 10%;
* reaction time: 0, 10, ..., 120 min.

For each of 10 controlled Random Forest seeds, the model is fitted to the
complete dataset and predictions are generated only at the 65 observed grid
cells. The plotted color is the cell-wise mean predicted Ct/C0 across seeds.
No interpolation, smoothing, contouring, extrapolated Cu-loading value, or
three-dimensional surface is used. The result is a fitted-model response map,
not an out-of-time or cross-validated performance evaluation.
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


FEATURE_COLUMNS = ["Cu-loading", "Reaction_Time"]
EXPECTED_CU_LEVELS = np.array([0.0, 3.0, 5.0, 7.5, 10.0])
EXPECTED_TIME_LEVELS = np.arange(0, 121, 10, dtype=float)
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
            "Create a discrete 5 x 13 RF response heat map for predicted "
            "Ct/C0 without smoothing or interpolation."
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
        default=SCRIPT_DIR / "rf_response_heatmap_results",
        help="Directory for prediction tables and publication figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively (useful in Spyder).",
    )
    return parser.parse_args()


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Load measurements and require the exact balanced 5 x 13 grid."""
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
    df = df.rename(
        columns={"Cu_Ratio": "Cu-loading", "Cu_Loading": "Cu-loading"}
    )

    required = {"Cu-loading", "Reaction_Time", "Ct_C0"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    analysis_df = df.loc[:, ["Cu-loading", "Reaction_Time", "Ct_C0"]].copy()
    if analysis_df.isna().any().any():
        raise ValueError("Missing values were found in the analysis columns.")

    duplicate_count = analysis_df.duplicated(FEATURE_COLUMNS).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicated Cu-loading/time cells."
        )

    observed_cu_levels = np.sort(
        analysis_df["Cu-loading"].unique().astype(float)
    )
    observed_time_levels = np.sort(
        analysis_df["Reaction_Time"].unique().astype(float)
    )
    if not np.array_equal(observed_cu_levels, EXPECTED_CU_LEVELS):
        raise ValueError(
            "Nominal Cu-loading levels must be exactly "
            f"{EXPECTED_CU_LEVELS.tolist()}, but found "
            f"{observed_cu_levels.tolist()}."
        )
    if not np.array_equal(observed_time_levels, EXPECTED_TIME_LEVELS):
        raise ValueError(
            "Reaction-time levels must be exactly "
            f"{EXPECTED_TIME_LEVELS.tolist()}, but found "
            f"{observed_time_levels.tolist()}."
        )

    expected_cells = len(EXPECTED_CU_LEVELS) * len(EXPECTED_TIME_LEVELS)
    if len(analysis_df) != expected_cells:
        raise ValueError(
            "The response map requires the complete 5 x 13 experimental "
            f"grid. Found {len(analysis_df)} of {expected_cells} cells."
        )

    counts_by_time = analysis_df.groupby("Reaction_Time").size()
    counts_by_cu = analysis_df.groupby("Cu-loading").size()
    if not (
        counts_by_time.eq(len(EXPECTED_CU_LEVELS)).all()
        and counts_by_cu.eq(len(EXPECTED_TIME_LEVELS)).all()
    ):
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


def make_observed_grid() -> pd.DataFrame:
    """Construct only the 65 experimentally observed predictor combinations."""
    return pd.DataFrame(
        [
            {
                "Cu-loading": cu_level,
                "Reaction_Time": time_level,
            }
            for cu_level in EXPECTED_CU_LEVELS
            for time_level in EXPECTED_TIME_LEVELS
        ]
    )


def calculate_multi_seed_grid_predictions(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return seed-level and cell-level prediction tables."""
    X = df.loc[:, FEATURE_COLUMNS]
    y = df["Ct_C0"]
    grid = make_observed_grid()
    seed_records: list[pd.DataFrame] = []

    for seed in RF_SEEDS:
        model = make_rf(seed)
        model.fit(X, y)
        seed_table = grid.copy()
        seed_table.insert(0, "seed", seed)
        seed_table["predicted_Ct_C0"] = model.predict(grid)
        seed_records.append(seed_table)

    predictions_by_seed = pd.concat(seed_records, ignore_index=True)
    prediction_summary = (
        predictions_by_seed.groupby(FEATURE_COLUMNS, as_index=False)[
            "predicted_Ct_C0"
        ]
        .agg(
            mean_predicted_Ct_C0="mean",
            sd_across_RF_seeds="std",
            minimum_across_RF_seeds="min",
            maximum_across_RF_seeds="max",
        )
        .sort_values(FEATURE_COLUMNS)
        .reset_index(drop=True)
    )
    prediction_summary["n_rf_seeds"] = len(RF_SEEDS)
    return predictions_by_seed, prediction_summary


def make_prediction_matrix(
    prediction_summary: pd.DataFrame,
    value_column: str,
) -> pd.DataFrame:
    """Pivot one summary column into the exact 5 x 13 grid."""
    matrix = prediction_summary.pivot(
        index="Cu-loading",
        columns="Reaction_Time",
        values=value_column,
    )
    return matrix.reindex(
        index=EXPECTED_CU_LEVELS,
        columns=EXPECTED_TIME_LEVELS,
    )


def plot_response_heatmap(
    mean_matrix: pd.DataFrame,
    output_path: Path,
    show_figure: bool,
) -> None:
    """Create a journal-ready unsmoothed heat map of the observed grid."""
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(style):
        fig, axis = plt.subplots(figsize=(7.1, 3.25))
        image = axis.imshow(
            mean_matrix.to_numpy(dtype=float),
            origin="lower",
            aspect="auto",
            interpolation="none",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
        )

        x_positions = np.arange(len(EXPECTED_TIME_LEVELS))
        y_positions = np.arange(len(EXPECTED_CU_LEVELS))
        axis.set_xticks(x_positions)
        axis.set_xticklabels(
            [f"{int(value)}" for value in EXPECTED_TIME_LEVELS]
        )
        axis.set_yticks(y_positions)
        axis.set_yticklabels(["0", "3", "5", "7.5", "10"])
        axis.set_xlabel("Reaction time (min)")
        axis.set_ylabel("Nominal Cu-loading (%)")

        # Cell boundaries make the discrete 5 x 13 design explicit.
        axis.set_xticks(np.arange(-0.5, len(EXPECTED_TIME_LEVELS), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(EXPECTED_CU_LEVELS), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=0.7, alpha=0.9)
        axis.tick_params(which="minor", bottom=False, left=False)

        colorbar = fig.colorbar(image, ax=axis, pad=0.025, fraction=0.045)
        colorbar.set_label(r"Mean predicted $C_t/C_0$", fontsize=9)
        colorbar.set_ticks(np.linspace(0.0, 1.0, 6))
        colorbar.ax.tick_params(labelsize=8)

        fig.subplots_adjust(left=0.12, right=0.91, bottom=0.20, top=0.97)
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
    predictions_by_seed, prediction_summary = (
        calculate_multi_seed_grid_predictions(df)
    )
    mean_matrix = make_prediction_matrix(
        prediction_summary, "mean_predicted_Ct_C0"
    )
    sd_matrix = make_prediction_matrix(
        prediction_summary, "sd_across_RF_seeds"
    )

    predictions_by_seed.to_csv(
        output_dir / "rf_grid_predictions_by_seed.csv", index=False
    )
    prediction_summary.to_csv(
        output_dir / "rf_grid_prediction_summary.csv", index=False
    )
    mean_matrix.to_csv(output_dir / "rf_grid_mean_matrix.csv")
    sd_matrix.to_csv(output_dir / "rf_grid_sd_matrix.csv")

    figure_path = output_dir / "Figure_RF_Response_Heatmap_Ct_C0_600dpi.png"
    plot_response_heatmap(mean_matrix, figure_path, args.show)

    print("\nDiscrete two-dimensional RF response map")
    print("Color: mean predicted Ct/C0 across 10 RF seeds")
    print("Grid: 5 nominal Cu-loading levels x 13 reaction times")
    print("Interpolation/smoothing: none\n")
    print(mean_matrix.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
