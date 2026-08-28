# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 10:34:45 2026

@author: sedak
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RF complexity sensitivity analysis for the small AC@NiO dataset.


Default: 6 configurations, 10 paired seeds, shuffled 5-fold CV, LOOCV,
and a fixed t<=100 -> t=110,120 min chronological diagnostic (55/10 rows).
The holdout is not used for automatic model selection. Reusing it to compare
configurations means it must not be described as an untouched final test.
"""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut


FEATURES = ["Cu_Loading", "Reaction_Time"]
CU_LEVELS = (0.0, 3.0, 5.0, 7.5, 10.0)
TIME_LEVELS = tuple(float(t) for t in range(0, 121, 10))

CONFIGURATIONS = {
    "unrestricted": {"max_depth": None, "min_samples_leaf": 1},
    "depth_3": {"max_depth": 3, "min_samples_leaf": 1},
    "depth_5": {"max_depth": 5, "min_samples_leaf": 1},
    "leaf_2": {"max_depth": None, "min_samples_leaf": 2},
    "leaf_3": {"max_depth": None, "min_samples_leaf": 3},
    "depth_5_leaf_2": {"max_depth": 5, "min_samples_leaf": 2},
}

RF_COMMON = {
    "n_estimators": 100,
    "criterion": "squared_error",
    "min_samples_split": 2,
    "max_features": 1.0,
    "bootstrap": True,
    "max_samples": None,
    "ccp_alpha": 0.0,
    "n_jobs": 1,
}

PROTOCOLS = ("fivefold", "loocv", "chronological")


def calculate_metrics(y_true, y_pred):
    return {
        "R2": float(r2_score(y_true, y_pred)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
    }


def make_rf(config_name, seed):
    return RandomForestRegressor(
        **RF_COMMON,
        **CONFIGURATIONS[config_name],
        random_state=int(seed),
    )


def load_dataset(path, include_log=False):
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig")

    if "Cu_Ratio" in df.columns and "Cu_Loading" not in df.columns:
        df = df.rename(columns={"Cu_Ratio": "Cu_Loading"})

    targets = ["Ct_C0"] + (["ln_C0_Ct"] if include_log else [])
    required = FEATURES + targets

    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df[required] = df[required].apply(pd.to_numeric, errors="raise")

    if len(df) != 65:
        raise ValueError(
            f"Expected 65 rows, found {len(df)}; no rows were removed."
        )

    if not np.isfinite(df[required].to_numpy()).all():
        raise ValueError(
            "Nonfinite values found; no automatic imputation performed."
        )

    if df.duplicated(FEATURES).any():
        raise ValueError(
            "Repeated Cu/time pairs found; review the experimental design."
        )

    grid = set(map(tuple, df[FEATURES].to_numpy(dtype=float)))
    expected_grid = {(c, t) for c in CU_LEVELS for t in TIME_LEVELS}

    if grid != expected_grid:
        raise ValueError(
            "The data do not match the documented five-by-thirteen grid."
        )

    if not df.Ct_C0.between(0, 1, inclusive="right").all():
        raise ValueError("Ct/C0 must be positive and <=1 for this dataset.")

    if not np.allclose(df.loc[df.Reaction_Time == 0, "Ct_C0"], 1):
        raise ValueError(
            "The post-dark baseline does not satisfy Ct/C0=1 at t=0."
        )

    audit = {}

    if "Efficiency" in df:
        audit["max_abs_efficiency_difference_normalized"] = float(
            np.max(
                np.abs(
                    pd.to_numeric(df.Efficiency) / 100 - (1 - df.Ct_C0)
                )
            )
        )

    if "ln_C0_Ct" in df:
        audit["max_abs_log_difference"] = float(
            np.max(
                np.abs(
                    pd.to_numeric(df.ln_C0_Ct) + np.log(df.Ct_C0)
                )
            )
        )

    # This derived column does not overwrite any source measurements.
    df["Efficiency_Norm_Derived"] = 1 - df.Ct_C0

    return df, targets, audit


def check_applicability(query, training_features, diagnostic_only=False):
    """Reject unsupported study-range queries and flag training-range violations.

    This is a numerical-range check, not proof of scientific applicability.
    Intermediate loadings/times are model-based interpolations, not validated
    new experiments. A chronological diagnostic may explicitly allow values
    beyond the fitted training time range but never outside the study range.
    """
    q = np.asarray(query, dtype=float)
    train = np.asarray(training_features, dtype=float)

    if q.ndim != 2 or q.shape[1] != 2 or not np.isfinite(q).all():
        raise ValueError(
            "Query must have finite [Cu_Loading, Reaction_Time] rows."
        )

    if (
        train.ndim != 2
        or train.shape[1] != 2
        or len(train) == 0
        or not np.isfinite(train).all()
    ):
        raise ValueError("Training feature matrix is invalid.")

    study_lower = np.array([0.0, 0.0])
    study_upper = np.array([10.0, 120.0])

    if np.any((q < study_lower) | (q > study_upper)):
        raise ValueError(
            "Unsupported query: nominal Cu outside [0,10]% "
            "or time outside [0,120] min."
        )

    outside_training = np.any(
        (q < train.min(axis=0)) | (q > train.max(axis=0)),
        axis=1,
    )

    if outside_training.any() and not diagnostic_only:
        raise ValueError(
            "Query is outside the fitted model's training range; "
            "diagnostic override required."
        )

    return pd.DataFrame(
        {
            "Cu_Loading": q[:, 0],
            "Reaction_Time": q[:, 1],
            "outside_training_range": outside_training,
            "unobserved_Cu_level": ~np.isin(q[:, 0], CU_LEVELS),
            "unobserved_time_level": ~np.isin(q[:, 1], TIME_LEVELS),
            "diagnostic_only": outside_training & diagnostic_only,
        }
    )


def predict_with_domain_guard(
    model,
    query,
    training_features,
    diagnostic_only=False,
):
    flags = check_applicability(
        query,
        training_features,
        diagnostic_only,
    )
    predictions = model.predict(np.asarray(query, dtype=float))

    return predictions, flags


def make_splits(x, seed, include_fivefold):
    splits = {}

    if include_fivefold:
        splits["fivefold"] = list(
            KFold(
                n_splits=5,
                shuffle=True,
                random_state=seed,
            ).split(x)
        )

    splits["loocv"] = list(LeaveOneOut().split(x))

    splits["chronological"] = [
        (
            np.flatnonzero(x[:, 1] <= 100),
            np.flatnonzero(x[:, 1] > 100),
        )
    ]

    for protocol, folds in splits.items():
        expected_ids = (
            np.arange(len(x))
            if protocol != "chronological"
            else np.flatnonzero(x[:, 1] > 100)
        )

        held_ids = np.concatenate([test for _, test in folds])

        if not np.array_equal(np.sort(held_ids), expected_ids):
            raise AssertionError("Holdout coverage or ordering is wrong.")

        for train, test in folds:
            if np.intersect1d(train, test).size:
                raise AssertionError("Training/validation row overlap.")

    train, test = splits["chronological"][0]

    if len(train) != 55 or len(test) != 10:
        raise AssertionError(
            "Chronological split must contain 55/10 rows."
        )

    return splits


def evaluate_one(config, seed, target, x, y, splits):
    """Evaluate one paired seed/config/target; no information crosses folds."""
    result = {
        "metrics": [],
        "predictions": [],
        "boundary": [],
        "complexity": [],
    }

    for protocol, folds in splits.items():
        prediction = np.full(len(y), np.nan)
        fold_ids = np.full(len(y), -1, dtype=int)

        for fold, (train, test) in enumerate(folds):
            model = make_rf(config, seed)
            model.fit(x[train], y[train])

            prediction[test] = model.predict(x[test])
            fold_ids[test] = fold

            if protocol == "chronological":
                grid = np.array(
                    [
                        (c, t)
                        for c in CU_LEVELS
                        for t in (100.0, 110.0, 120.0)
                    ]
                )

                boundary_prediction, flags = predict_with_domain_guard(
                    model,
                    grid,
                    x[train],
                    diagnostic_only=True,
                )

                for index, (cu, reaction_time) in enumerate(grid):
                    result["boundary"].append(
                        {
                            "config": config,
                            "seed": seed,
                            "target": target,
                            "Cu_Loading": cu,
                            "Reaction_Time": reaction_time,
                            "predicted": float(boundary_prediction[index]),
                            "outside_training_range": bool(
                                flags.iloc[index].outside_training_range
                            ),
                        }
                    )

                boundary_spread = np.ptp(
                    boundary_prediction.reshape(5, 3),
                    axis=1,
                )

                if np.max(boundary_spread) > 1e-12:
                    raise AssertionError(
                        "Unexpected nonconstant RF boundary prediction; "
                        "investigate."
                    )

                depths = [
                    tree.tree_.max_depth
                    for tree in model.estimators_
                ]
                leaves = [
                    tree.tree_.n_leaves
                    for tree in model.estimators_
                ]

                training_scores = calculate_metrics(
                    y[train],
                    model.predict(x[train]),
                )

                result["complexity"].append(
                    {
                        "config": config,
                        "seed": seed,
                        "target": target,
                        "training_rows": len(train),
                        "mean_realized_depth": float(np.mean(depths)),
                        "max_realized_depth": int(max(depths)),
                        "mean_leaf_count": float(np.mean(leaves)),
                        **{
                            f"training_{key}": value
                            for key, value in training_scores.items()
                        },
                    }
                )

        scored = np.flatnonzero(np.isfinite(prediction))
        score = calculate_metrics(y[scored], prediction[scored])

        result["metrics"].append(
            {
                "config": config,
                "seed": seed,
                "target": target,
                "protocol": protocol,
                "n_evaluated": len(scored),
                **score,
            }
        )

        if target == "Ct_C0":
            derived_score = calculate_metrics(
                1 - y[scored],
                1 - prediction[scored],
            )

            for metric in score:
                if not np.isclose(
                    score[metric],
                    derived_score[metric],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise AssertionError("Affine metric identity failed.")

        result["predictions"].extend(
            {
                "config": config,
                "seed": seed,
                "target": target,
                "protocol": protocol,
                "fold_zero_based": int(fold_ids[index]),
                "row_zero_based": int(index),
                "Cu_Loading": float(x[index, 0]),
                "Reaction_Time": float(x[index, 1]),
                "actual": float(y[index]),
                "predicted": float(prediction[index]),
            }
            for index in scored
        )

    return result


def last_value_reference(df, targets):
    """A no-fit diagnostic: carry each composition's t=100 observation forward."""
    records = []
    predictions = []

    test = df.loc[df.Reaction_Time > 100]

    for target in targets:
        last = (
            df.loc[df.Reaction_Time == 100]
            .set_index("Cu_Loading")[target]
        )

        pred = test.Cu_Loading.map(last).to_numpy()

        records.append(
            {
                "target": target,
                "method": "last_observed_100min",
                "protocol": "chronological",
                "n_evaluated": len(test),
                **calculate_metrics(test[target], pred),
            }
        )

        predictions.extend(
            {
                "row_zero_based": int(index),
                "target": target,
                "Cu_Loading": float(df.loc[index, "Cu_Loading"]),
                "Reaction_Time": float(df.loc[index, "Reaction_Time"]),
                "actual": float(df.loc[index, target]),
                "predicted": float(value),
            }
            for index, value in zip(test.index, pred)
        )

    return pd.DataFrame(records), pd.DataFrame(predictions)


def summarize(metrics):
    summary = (
        metrics.groupby(
            ["config", "target", "protocol"],
            sort=False,
        )[["R2", "MAE", "MSE"]]
        .agg(["mean", "std"])
    )

    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    baseline = metrics.loc[
        metrics.config == "unrestricted",
        ["seed", "target", "protocol", "R2", "MAE", "MSE"],
    ]

    paired = metrics.merge(
        baseline,
        on=["seed", "target", "protocol"],
        suffixes=("", "_baseline"),
        validate="many_to_one",
    )

    for metric in ("R2", "MAE", "MSE"):
        paired[f"delta_{metric}"] = (
            paired[metric] - paired[f"{metric}_baseline"]
        )

    deltas = (
        paired.groupby(
            ["config", "target", "protocol"],
            sort=False,
        )[["delta_R2", "delta_MAE", "delta_MSE"]]
        .agg(["mean", "std"])
    )

    deltas.columns = [
        "_".join(pair)
        for pair in deltas.columns
    ]

    return summary, paired, deltas.reset_index()


def make_markdown_table(summary, target):
    data = (
        summary.loc[summary.target == target]
        .set_index(["config", "protocol"])
    )

    protocols = [
        p for p in PROTOCOLS
        if p in set(summary.protocol)
    ]

    headings = {
        "fivefold": "5-fold R2",
        "loocv": "LOOCV R2",
        "chronological": "Chronological R2",
    }

    lines = [
        "| Configuration | "
        + " | ".join(headings[p] for p in protocols)
        + " |",
        "|---|" + "---:|" * len(protocols),
    ]

    for config in CONFIGURATIONS:
        values = [
            (
                f"{data.loc[(config, p), 'R2_mean']:.4f} +/- "
                f"{data.loc[(config, p), 'R2_std']:.4f}"
            )
            for p in protocols
        ]
        lines.append(
            "| " + config + " | " + " | ".join(values) + " |"
        )

    return "\n".join(lines)


def write_report(out, summary, persistence, seeds):
    sections = [
        "# RF complexity sensitivity results",
        "",
        (
            "Computed from the supplied dataset. Values are means +/- "
            "sample SD across controlled RF seeds (ddof=1)."
        ),
        (
            "This SD is not experimental uncertainty; the seeds are not "
            "independent experimental replicates."
        ),
        "",
    ]

    for target in summary.target.unique():
        sections += [
            f"## {target}",
            "",
            make_markdown_table(summary, target),
            "",
        ]

    p = persistence.loc[persistence.target == "Ct_C0"].iloc[0]

    sections += [
        "## Chronological boundary diagnostic",
        "",
        (
            "Training uses 55 rows at t<=100 min; scoring uses "
            "10 rows at 110/120 min."
        ),
        (
            "For each fixed composition, the RF predictions at 100, 110, "
            "and 120 min were checked for equality."
        ),
        (
            "Thus this is outside-training-range evaluation in the time "
            "input, but not learned extrapolation of a temporal trend."
        ),
        (
            f"Carrying the observed value at 100 min forward gives "
            f"R2={p.R2:.6f}, MAE={p.MAE:.6f}, MSE={p.MSE:.8f}."
        ),
        (
            "This reference was computed without fitting or using "
            "later-time outcomes. It is provided only to contextualize "
            "the holdout difficulty."
        ),
        "",
        "## Interpretation limits",
        "",
        (
            "- No automatic hyperparameter selection is performed; "
            "no other manuscript analyses are replaced."
        ),
        (
            "- Comparisons on the reused chronological holdout are "
            "descriptive diagnostics, not an untouched test after selection."
        ),
        (
            "- Row-wise LOOCV can share other times from the same "
            "composition between training and validation."
        ),
        (
            "- Good scores do not establish freedom from overfitting, "
            "external generalizability, causality, or a removal mechanism."
        ),
        (
            "- The investigated numerical envelope is 0<=t<=120 min "
            "and 0<=nominal Cu<=10%. Only five Cu levels are observed."
        ),
        (
            "- Intermediate Cu levels/times are unvalidated model-based "
            "interpolation. Being inside the envelope is not sufficient "
            "evidence of reliability."
        ),
        (
            "- New experimental conditions, independently synthesized "
            "batches, or values outside the observed envelope require "
            "independent validation."
        ),
        "",
        "## Source and reproducibility",
        "",
        (
            "See protocol.json, split_membership.json, metrics_by_seed.csv, "
            "predictions.csv, and boundary_diagnostics.csv."
        ),
        f"Seeds used: {', '.join(map(str, seeds))}.",
        (
            "The default sensitivity analysis covers Ct/C0 only. "
            "Use --include-log for a separate logarithmic-target analysis."
        ),
        (
            "The omitted efficiency response is defined deterministically "
            "by e=1-r, including its predictions; it is not refitted."
        ),
        "",
        (
            "Technical reference "
            "(not a substitute for the supplied experimental data):"
        ),
        (
            "https://scikit-learn.org/stable/auto_examples/ensemble/"
            "plot_random_forest_regression_multioutput.html"
        ),
        "",
    ]

    (out / "RF_sensitivity_results.md").write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("rf_sensitivity_results"),
    )
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(10)),
    )
    parser.add_argument("--include-log", action="store_true")
    parser.add_argument("--skip-fivefold", action="store_true")

    args = parser.parse_args()

    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error(
            "Use at least two distinct seeds to report sample SD."
        )

    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be integers from 0 to 2**32-1.")

    if args.jobs < 1:
        parser.error(
            "--jobs must be a positive integer; avoid nested parallelism."
        )

    data_path = args.data

    if data_path is None:
        local = Path("catalyst_dataset.csv")
        data_path = (
            local
            if local.is_file()
            else Path(__file__).resolve().with_name("catalyst_dataset.csv")
        )

    if not data_path.is_file():
        parser.error(
            f"Dataset not found: {data_path}. "
            "Supply --data with the actual CSV path."
        )

    # Fail on populated output folders instead of overwriting previous runs.
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Output directory is not empty; use a new --out directory "
            "to preserve prior results."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    df, targets, audit = load_dataset(
        data_path,
        include_log=args.include_log,
    )
    x = df[FEATURES].to_numpy(dtype=float)

    splits = {
        seed: make_splits(
            x,
            seed,
            include_fivefold=not args.skip_fivefold,
        )
        for seed in args.seeds
    }

    manifest = {
        str(seed): {
            protocol: [
                {
                    "train_rows": train.tolist(),
                    "validation_rows": test.tolist(),
                }
                for train, test in folds
            ]
            for protocol, folds in split.items()
        }
        for seed, split in splits.items()
    }

    (args.out / "split_membership.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    protocol = {
        "input_filename": data_path.name,
        "input_sha256": hashlib.sha256(
            data_path.read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "N": len(df),
        "features": FEATURES,
        "targets": targets,
        "common_RF_parameters": RF_COMMON,
        "configurations": CONFIGURATIONS,
        "seeds": args.seeds,
        "summary_sd_ddof": 1,
        "data_checks": audit,
        "scoring": (
            "Pooled held-out predictions per seed, "
            "not mean of fold R2 values."
        ),
        "chronological_split": (
            "t<=100 training (55); "
            "t=110/120 diagnostic evaluation (10)."
        ),
        "paired_design": (
            "Identical folds, seed and bootstrap random stream "
            "for every compared configuration."
        ),
        "hyperparameter_selection": (
            "None; descriptive sensitivity only. "
            "No best-OOT-model selection."
        ),
        "feature_or_target_scaling": "None",
        "XAI_scope": (
            "No SHAP/PFI/PDP analyses run or overwritten by this script."
        ),
        "versions": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "joblib": joblib.__version__,
        },
    }

    (args.out / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    persistence, persistence_predictions = last_value_reference(
        df,
        targets,
    )

    persistence.to_csv(
        args.out / "last_value_reference_metrics.csv",
        index=False,
    )

    persistence_predictions.to_csv(
        args.out / "last_value_reference_predictions.csv",
        index=False,
    )

    domain = {
        "investigated_envelope": {
            "Cu_Loading_percent": [0, 10],
            "Reaction_Time_min": [0, 120],
        },
        "observed_Cu_levels_percent": list(CU_LEVELS),
        "observed_time_levels_min": list(TIME_LEVELS),
        "chronological_training_envelope": {
            "Cu_Loading_percent": [0, 10],
            "Reaction_Time_min": [0, 100],
        },
        "intermediate_inputs": (
            "Model-based interpolation; not independently validated."
        ),
        "outside_envelope": (
            "Unsupported; blocked by predict_with_domain_guard."
        ),
        "scope": (
            "Same investigated experimental conditions only; "
            "numeric bounds alone do not establish scientific applicability."
        ),
    }

    (args.out / "applicability_domain.json").write_text(
        json.dumps(domain, indent=2),
        encoding="utf-8",
    )

    tasks = [
        (config, seed, target)
        for config in CONFIGURATIONS
        for seed in args.seeds
        for target in targets
    ]

    combined = {
        key: []
        for key in ("metrics", "predictions", "boundary", "complexity")
    }

    start = time.perf_counter()

    with parallel_config(
        backend="loky",
        inner_max_num_threads=1,
    ):
        results = Parallel(
            n_jobs=args.jobs,
            return_as="generator_unordered",
        )(
            delayed(evaluate_one)(
                config,
                seed,
                target,
                x,
                df[target].to_numpy(dtype=float),
                splits[seed],
            )
            for config, seed, target in tasks
        )

        for completed, result in enumerate(results, 1):
            for key in combined:
                combined[key].extend(result[key])

            # Per-task checkpoints are audit records, not cached reruns.
            record = result["metrics"][0]

            checkpoint = args.out / "checkpoints"
            checkpoint.mkdir(exist_ok=True)

            filename = (
                f"{record['config']}_seed{record['seed']}_"
                f"{record['target']}.json"
            )

            (checkpoint / filename).write_text(
                json.dumps(result),
                encoding="utf-8",
            )

            loo = next(
                row["R2"]
                for row in result["metrics"]
                if row["protocol"] == "loocv"
            )

            oot = next(
                row["R2"]
                for row in result["metrics"]
                if row["protocol"] == "chronological"
            )

            elapsed = time.perf_counter() - start

            print(
                f"[{completed}/{len(tasks)}] "
                f"{record['config']} "
                f"seed={record['seed']} "
                f"{record['target']}: "
                f"LOOCV R2={loo:.6f}; "
                f"chronological R2={oot:.6f}; "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    metrics = (
        pd.DataFrame(combined["metrics"])
        .sort_values(["config", "target", "protocol", "seed"])
    )

    predictions = (
        pd.DataFrame(combined["predictions"])
        .sort_values(
            [
                "config",
                "target",
                "protocol",
                "seed",
                "row_zero_based",
            ]
        )
    )

    summary, paired, delta_summary = summarize(metrics)

    metrics.to_csv(
        args.out / "metrics_by_seed.csv",
        index=False,
    )

    predictions.to_csv(
        args.out / "predictions.csv",
        index=False,
    )

    summary.to_csv(
        args.out / "sensitivity_summary.csv",
        index=False,
    )

    paired.to_csv(
        args.out / "paired_differences_by_seed.csv",
        index=False,
    )

    delta_summary.to_csv(
        args.out / "paired_differences_summary.csv",
        index=False,
    )

    (
        pd.DataFrame(combined["boundary"])
        .sort_values(
            [
                "config",
                "target",
                "seed",
                "Cu_Loading",
                "Reaction_Time",
            ]
        )
        .to_csv(
            args.out / "boundary_diagnostics.csv",
            index=False,
        )
    )

    (
        pd.DataFrame(combined["complexity"])
        .sort_values(["config", "target", "seed"])
        .to_csv(
            args.out / "realized_tree_complexity.csv",
            index=False,
        )
    )

    write_report(
        args.out,
        summary,
        persistence,
        args.seeds,
    )

    print(
        "\n" + make_markdown_table(summary, "Ct_C0"),
        flush=True,
    )

    print(
        "\nLast-observation reference:\n"
        + persistence.to_string(index=False),
        flush=True,
    )

    print(
        f"\nSaved actual results to: {args.out.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()