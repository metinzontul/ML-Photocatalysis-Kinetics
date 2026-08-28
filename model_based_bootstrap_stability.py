#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Model-based block-bootstrap stability analysis for the AC@NiO RF model.

The analysis quantifies the stability of three Random Forest importance
measures for predicting Ct/C0 from nominal Cu-loading and reaction time:

1. normalized mean decrease in impurity (MDI);
2. marginal permutation feature importance (PFI);
3. grouped-conditional PFI.

Bootstrap design
----------------
The dataset contains one observation for every cell of a balanced 5 x 13
Cu-loading-by-time grid, but it contains no independent experimental
replicates. Ordinary row bootstrap would incorrectly treat the 65 cells as
exchangeable. Instead, this script uses a synchronized moving-block bootstrap
over the 13 ordered reaction-time levels:

* blocks contain three adjacent sampling occasions by default;
* the same sampled time sequence is applied to all five Cu-loading groups;
* every bootstrap training sample therefore retains all five Cu-loading
  groups at each selected time occurrence;
* a new RF model is fitted in every bootstrap repetition.

PFI is evaluated on the unchanged 65-cell reference grid so estimates from
different bootstrap refits remain comparable. Marginal PFI shuffles one
feature across the complete reference grid. Grouped-conditional PFI shuffles
Cu-loading within fixed reaction time and reaction time within fixed
Cu-loading. Both R2 decrease and MAE increase are saved; the publication
figure displays R2 decrease.

The default analysis uses 1000 bootstrap refits and 50 inner permutations per
PFI estimate. Reported 2.5th--97.5th percentiles describe model-based
resampling stability. They are not experimental confidence intervals and do
not quantify catalyst-batch reproducibility.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


FEATURE_COLUMNS = ["Cu-loading", "Reaction_Time"]
FEATURE_LABELS = {
    "Cu-loading": "Nominal Cu-loading",
    "Reaction_Time": "Reaction time",
}
CONDITIONING_FEATURE = {
    "Cu-loading": "Reaction_Time",
    "Reaction_Time": "Cu-loading",
}
EXPECTED_CU_LEVELS = np.array([0.0, 3.0, 5.0, 7.5, 10.0])
EXPECTED_TIME_LEVELS = np.arange(0.0, 121.0, 10.0)

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_BLOCK_LENGTH = 3
DEFAULT_N_PERMUTATIONS = 50
DEFAULT_MASTER_SEED = 20260824
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
            "Run synchronized moving-block bootstrap stability analysis for "
            "MDI, marginal PFI, and grouped-conditional PFI."
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
        default=SCRIPT_DIR / "bootstrap_stability_results",
        help="Directory for CSV results and publication figures.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Number of block-bootstrap RF refits (default: 1000).",
    )
    parser.add_argument(
        "--block-length",
        type=int,
        default=DEFAULT_BLOCK_LENGTH,
        help="Number of adjacent sampling occasions per time block.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
        help="Inner permutations per PFI estimate (default: 50).",
    )
    parser.add_argument(
        "--master-seed",
        type=int,
        default=DEFAULT_MASTER_SEED,
        help="Master seed controlling resampling, RF fitting, and permutation.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many bootstrap repetitions.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display the figure interactively (useful in Spyder).",
    )
    args = parser.parse_args()

    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be at least 1.")
    if args.n_permutations < 1:
        parser.error("--n-permutations must be at least 1.")
    if args.block_length < 1 or args.block_length > len(EXPECTED_TIME_LEVELS):
        parser.error(
            "--block-length must be between 1 and the number of time levels."
        )
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1.")
    return args


def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """Load the data without changing measurements and verify the 5 x 13 grid."""
    if not data_path.exists():
        candidates = sorted(SCRIPT_DIR.glob("*.csv"))
        available = ", ".join(path.name for path in candidates) or "none"
        raise FileNotFoundError(
            f"Dataset not found: {data_path}\n"
            f"CSV files next to the script: {available}\n"
            "Place catalyst_dataset.csv next to this script or provide the "
            "complete path using --data."
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

    observed_cu = np.sort(analysis_df["Cu-loading"].unique().astype(float))
    observed_time = np.sort(
        analysis_df["Reaction_Time"].unique().astype(float)
    )
    if not np.array_equal(observed_cu, EXPECTED_CU_LEVELS):
        raise ValueError(
            "Nominal Cu-loading levels must be exactly "
            f"{EXPECTED_CU_LEVELS.tolist()}, but found {observed_cu.tolist()}."
        )
    if not np.array_equal(observed_time, EXPECTED_TIME_LEVELS):
        raise ValueError(
            "Reaction times must be exactly "
            f"{EXPECTED_TIME_LEVELS.tolist()}, but found {observed_time.tolist()}."
        )

    expected_cells = len(EXPECTED_CU_LEVELS) * len(EXPECTED_TIME_LEVELS)
    if len(analysis_df) != expected_cells:
        raise ValueError(
            f"Expected a complete 5 x 13 grid ({expected_cells} cells), "
            f"but found {len(analysis_df)} rows."
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
    """Create the RF configuration used throughout the manuscript."""
    return RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=None,
    )


def sample_synchronized_time_blocks(
    df: pd.DataFrame,
    block_length: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Resample adjacent time blocks jointly across all Cu-loading groups."""
    n_time = len(EXPECTED_TIME_LEVELS)
    possible_starts = np.arange(0, n_time - block_length + 1)
    n_blocks = math.ceil(n_time / block_length)
    block_starts = rng.choice(possible_starts, size=n_blocks, replace=True)
    sampled_positions = np.concatenate(
        [
            np.arange(start, start + block_length, dtype=int)
            for start in block_starts
        ]
    )[:n_time]
    sampled_times = EXPECTED_TIME_LEVELS[sampled_positions]

    pieces = [
        df.loc[df["Reaction_Time"].eq(time_value)]
        for time_value in sampled_times
    ]
    bootstrap_df = pd.concat(pieces, ignore_index=True)

    expected_rows = n_time * len(EXPECTED_CU_LEVELS)
    if len(bootstrap_df) != expected_rows:
        raise RuntimeError("The synchronized block bootstrap changed sample size.")
    if not bootstrap_df.groupby("Reaction_Time", sort=False)["Cu-loading"].apply(
        lambda values: set(values) == set(EXPECTED_CU_LEVELS)
    ).all():
        raise RuntimeError("At least one selected time occurrence lost a Cu group.")

    return bootstrap_df, sampled_times, block_starts


def permute_feature(
    X: pd.DataFrame,
    feature: str,
    scheme: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply marginal or grouped-conditional permutation on the fixed grid."""
    X_permuted = X.copy()

    if scheme == "Marginal PFI":
        X_permuted[feature] = rng.permutation(X[feature].to_numpy())
        return X_permuted

    if scheme != "Conditional PFI":
        raise ValueError(f"Unsupported permutation scheme: {scheme}")

    condition_on = CONDITIONING_FEATURE[feature]
    permuted_values = X[feature].to_numpy(copy=True)
    for group_indices in X.groupby(condition_on, sort=True).groups.values():
        positions = np.asarray(list(group_indices), dtype=int)
        permuted_values[positions] = rng.permutation(
            X.loc[positions, feature].to_numpy()
        )
    X_permuted[feature] = permuted_values
    return X_permuted


def calculate_pfi(
    model: RandomForestRegressor,
    X_reference: pd.DataFrame,
    y_reference: np.ndarray,
    baseline_r2: float,
    baseline_mae: float,
    feature: str,
    scheme: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Calculate one bootstrap replicate's mean PFI using batched prediction."""
    permuted_frames = [
        permute_feature(X_reference, feature, scheme, rng)
        for _ in range(n_permutations)
    ]
    stacked_features = pd.concat(permuted_frames, ignore_index=True)
    stacked_predictions = model.predict(stacked_features).reshape(
        n_permutations, len(X_reference)
    )

    residuals = y_reference[np.newaxis, :] - stacked_predictions
    total_sum_of_squares = np.sum(
        (y_reference - np.mean(y_reference)) ** 2
    )
    permuted_r2 = 1.0 - np.sum(residuals**2, axis=1) / total_sum_of_squares
    permuted_mae = np.mean(np.abs(residuals), axis=1)

    r2_decrease = baseline_r2 - permuted_r2
    mae_increase = permuted_mae - baseline_mae
    return {
        "r2_decrease_mean": float(np.mean(r2_decrease)),
        "r2_decrease_inner_sd": float(np.std(r2_decrease, ddof=1)),
        "mae_increase_mean": float(np.mean(mae_increase)),
        "mae_increase_inner_sd": float(np.std(mae_increase, ddof=1)),
    }


def run_bootstrap_analysis(
    df: pd.DataFrame,
    n_bootstrap: int,
    block_length: int,
    n_permutations: int,
    master_seed: int,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Refit the RF and record all requested importance estimates."""
    X_reference = df.loc[:, FEATURE_COLUMNS].reset_index(drop=True)
    y_reference = df["Ct_C0"].to_numpy(dtype=float)

    importance_records: list[dict[str, float | int | str]] = []
    baseline_records: list[dict[str, float | int]] = []
    resampling_records: list[dict[str, int | str]] = []

    for bootstrap_id in range(1, n_bootstrap + 1):
        seed_sequence = np.random.SeedSequence([master_seed, bootstrap_id])
        bootstrap_seed, pfi_seed = seed_sequence.spawn(2)
        bootstrap_rng = np.random.default_rng(bootstrap_seed)
        pfi_rng = np.random.default_rng(pfi_seed)
        rf_seed = int(bootstrap_rng.integers(0, 2**31 - 1))

        bootstrap_df, sampled_times, block_starts = (
            sample_synchronized_time_blocks(df, block_length, bootstrap_rng)
        )
        model = make_rf(rf_seed)
        model.fit(
            bootstrap_df.loc[:, FEATURE_COLUMNS],
            bootstrap_df["Ct_C0"],
        )

        baseline_prediction = model.predict(X_reference)
        baseline_r2 = float(r2_score(y_reference, baseline_prediction))
        baseline_mae = float(
            mean_absolute_error(y_reference, baseline_prediction)
        )
        baseline_records.append(
            {
                "bootstrap_id": bootstrap_id,
                "rf_seed": rf_seed,
                "fixed_grid_r2": baseline_r2,
                "fixed_grid_mae": baseline_mae,
            }
        )
        resampling_records.append(
            {
                "bootstrap_id": bootstrap_id,
                "rf_seed": rf_seed,
                "block_length": block_length,
                "block_start_positions": "|".join(
                    str(int(value)) for value in block_starts
                ),
                "sampled_time_sequence": "|".join(
                    f"{value:g}" for value in sampled_times
                ),
            }
        )

        mdi_percentages = 100.0 * model.feature_importances_
        for feature, mdi_value in zip(FEATURE_COLUMNS, mdi_percentages):
            importance_records.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "rf_seed": rf_seed,
                    "method": "MDI",
                    "feature": FEATURE_LABELS[feature],
                    "metric": "Normalized importance (%)",
                    "importance": float(mdi_value),
                    "inner_permutation_sd": np.nan,
                    "n_inner_permutations": 0,
                    "block_length": block_length,
                }
            )

        for scheme in ["Marginal PFI", "Conditional PFI"]:
            for feature in FEATURE_COLUMNS:
                pfi_values = calculate_pfi(
                    model=model,
                    X_reference=X_reference,
                    y_reference=y_reference,
                    baseline_r2=baseline_r2,
                    baseline_mae=baseline_mae,
                    feature=feature,
                    scheme=scheme,
                    n_permutations=n_permutations,
                    rng=pfi_rng,
                )
                importance_records.extend(
                    [
                        {
                            "bootstrap_id": bootstrap_id,
                            "rf_seed": rf_seed,
                            "method": scheme,
                            "feature": FEATURE_LABELS[feature],
                            "metric": "R2 decrease",
                            "importance": pfi_values["r2_decrease_mean"],
                            "inner_permutation_sd": pfi_values[
                                "r2_decrease_inner_sd"
                            ],
                            "n_inner_permutations": n_permutations,
                            "block_length": block_length,
                        },
                        {
                            "bootstrap_id": bootstrap_id,
                            "rf_seed": rf_seed,
                            "method": scheme,
                            "feature": FEATURE_LABELS[feature],
                            "metric": "MAE increase",
                            "importance": pfi_values["mae_increase_mean"],
                            "inner_permutation_sd": pfi_values[
                                "mae_increase_inner_sd"
                            ],
                            "n_inner_permutations": n_permutations,
                            "block_length": block_length,
                        },
                    ]
                )

        if bootstrap_id % progress_every == 0 or bootstrap_id == n_bootstrap:
            print(
                f"Completed {bootstrap_id}/{n_bootstrap} bootstrap refits",
                flush=True,
            )

    return (
        pd.DataFrame(importance_records),
        pd.DataFrame(baseline_records),
        pd.DataFrame(resampling_records),
    )


def summarize_bootstrap_estimates(estimates: pd.DataFrame) -> pd.DataFrame:
    """Report median and 2.5th--97.5th bootstrap percentiles."""
    summary_records: list[dict[str, float | int | str]] = []
    group_columns = ["method", "feature", "metric"]
    for keys, group in estimates.groupby(group_columns, sort=False):
        values = group["importance"].to_numpy(dtype=float)
        method, feature, metric = keys
        summary_records.append(
            {
                "method": method,
                "feature": feature,
                "metric": metric,
                "median": float(np.median(values)),
                "percentile_2_5": float(np.quantile(values, 0.025)),
                "percentile_97_5": float(np.quantile(values, 0.975)),
                "mean": float(np.mean(values)),
                "standard_deviation": float(np.std(values, ddof=1)),
                "proportion_positive": float(np.mean(values > 0.0)),
                "n_bootstrap": len(values),
                "block_length": int(group["block_length"].iloc[0]),
                "inner_permutations": int(
                    group["n_inner_permutations"].iloc[0]
                ),
            }
        )
    return pd.DataFrame(summary_records)


def plot_bootstrap_stability(
    summary: pd.DataFrame,
    output_path: Path,
    show_figure: bool,
) -> None:
    """Plot medians and percentile intervals for MDI and R2-based PFI."""
    panels = [
        ("MDI", "Normalized importance (%)", "Normalized MDI importance (%)"),
        ("Marginal PFI", "R2 decrease", r"Decrease in fixed-grid $R^2$"),
        ("Conditional PFI", "R2 decrease", r"Decrease in fixed-grid $R^2$"),
    ]
    feature_order = ["Nominal Cu-loading", "Reaction time"]
    colors = {
        "Nominal Cu-loading": "#2F5D8A",
        "Reaction time": "#C06C3B",
    }

    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8.3,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
    }

    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75), sharey=True)
        y_positions = np.arange(len(feature_order), dtype=float)

        for panel_index, (axis, panel) in enumerate(zip(axes, panels)):
            method, metric, x_label = panel
            table = (
                summary.loc[
                    (summary["method"] == method)
                    & (summary["metric"] == metric)
                ]
                .set_index("feature")
                .reindex(feature_order)
            )
            if table[["median", "percentile_2_5", "percentile_97_5"]].isna().any().any():
                raise ValueError(f"Missing bootstrap summary values for {method}.")

            for y_position, feature in zip(y_positions, feature_order):
                median = float(table.loc[feature, "median"])
                lower = float(table.loc[feature, "percentile_2_5"])
                upper = float(table.loc[feature, "percentile_97_5"])
                axis.errorbar(
                    median,
                    y_position,
                    xerr=np.array([[median - lower], [upper - median]]),
                    fmt="o",
                    markersize=5,
                    color=colors[feature],
                    ecolor=colors[feature],
                    elinewidth=1.2,
                    capsize=3,
                    capthick=1.0,
                    zorder=3,
                )

            axis.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
            axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.8)
            axis.set_axisbelow(True)
            axis.set_xlabel(x_label)
            axis.set_title(
                f"({chr(97 + panel_index)}) {method}",
                loc="left",
                fontweight="bold",
            )

        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels(feature_order)
        axes[0].invert_yaxis()

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                color=colors[feature],
                linestyle="none",
                markersize=5,
                label=feature,
            )
            for feature in feature_order
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.56, 0.995),
            ncol=2,
            frameon=False,
            handletextpad=0.45,
            columnspacing=1.5,
            fontsize=8,
        )
        fig.subplots_adjust(
            left=0.19,
            right=0.985,
            bottom=0.23,
            top=0.78,
            wspace=0.48,
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
    estimates, baseline_performance, resampling_log = run_bootstrap_analysis(
        df=df,
        n_bootstrap=args.n_bootstrap,
        block_length=args.block_length,
        n_permutations=args.n_permutations,
        master_seed=args.master_seed,
        progress_every=args.progress_every,
    )
    summary = summarize_bootstrap_estimates(estimates)

    estimates.to_csv(
        output_dir / "bootstrap_importance_estimates.csv", index=False
    )
    summary.to_csv(
        output_dir / "bootstrap_importance_summary.csv", index=False
    )
    baseline_performance.to_csv(
        output_dir / "bootstrap_fixed_grid_performance.csv", index=False
    )
    resampling_log.to_csv(
        output_dir / "bootstrap_resampling_log.csv", index=False
    )

    figure_path = (
        output_dir / "Figure_Model_Based_Bootstrap_Stability_600dpi.png"
    )
    plot_bootstrap_stability(summary, figure_path, args.show)

    print("\nModel-based bootstrap stability analysis")
    print(f"Bootstrap RF refits: {args.n_bootstrap}")
    print(f"Synchronized moving-block length: {args.block_length}")
    print(f"Inner permutations per PFI estimate: {args.n_permutations}")
    print("PFI evaluation grid: unchanged 5 x 13 experimental grid\n")
    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
