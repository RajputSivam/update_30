"""Tune the two development-CV winners and evaluate them on the final hold-out."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.employee_attrition import (
    SEED,
    ExperimentConfig,
    _fit_final,
    _metric_row,
    load_dataset,
)

REPORT_METRICS = [
    "accuracy", "balanced_accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc", "mcc",
]
EXPECTED_DEVELOPMENT_ROWS = 1176
EXPECTED_HOLDOUT_ROWS = 294
BOOTSTRAP_RESAMPLES = 2000


def _bootstrap_intervals(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    seed: int,
) -> dict[str, tuple[float, float]]:
    labels = y_true.to_numpy()
    class_indices = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("The hold-out set must contain both attrition classes.")

    random = np.random.default_rng(seed)
    samples = {metric: np.empty(BOOTSTRAP_RESAMPLES, dtype=float) for metric in REPORT_METRICS}
    for resample_index in range(BOOTSTRAP_RESAMPLES):
        sampled_indices = np.concatenate([
            random.choice(indices, size=len(indices), replace=True) for indices in class_indices
        ])
        row = _metric_row(
            pd.Series(labels[sampled_indices]), probabilities[sampled_indices], threshold
        )
        for metric in REPORT_METRICS:
            samples[metric][resample_index] = float(row[metric])

    return {
        metric: tuple(np.percentile(values, [2.5, 97.5]).tolist())
        for metric, values in samples.items()
    }


def _select_configurations(summary_path: Path) -> list[tuple[str, pd.Series]]:
    summary = pd.read_csv(summary_path)
    required = {"model", "strategy", "pr_auc_mean", "mcc_mean"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"CV summary is missing required columns: {', '.join(sorted(missing))}")

    summary = summary.dropna(subset=["model", "strategy", "pr_auc_mean", "mcc_mean"])
    if summary.empty:
        raise ValueError("CV summary has no complete learner/strategy rows.")

    pr_auc_winner = summary.sort_values("pr_auc_mean", ascending=False, kind="stable").iloc[0]
    mcc_winner = summary.sort_values("mcc_mean", ascending=False, kind="stable").iloc[0]
    return [("highest_mean_pr_auc", pr_auc_winner), ("highest_mean_mcc", mcc_winner)]


def run_final_holdout(
    data_path: Path,
    summary_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selections = _select_configurations(summary_path)
    X, y, categorical, numeric = load_dataset(data_path)
    X_development, X_holdout, y_development, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.seed
    )
    if len(X_development) != EXPECTED_DEVELOPMENT_ROWS or len(X_holdout) != EXPECTED_HOLDOUT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_DEVELOPMENT_ROWS} development and {EXPECTED_HOLDOUT_ROWS} hold-out "
            f"rows; got {len(X_development)} and {len(X_holdout)}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []
    cost_rows: list[dict[str, Any]] = []
    baseline_counts = {
        "flag_nobody": (int(y_holdout.sum()), 0),
        "flag_everybody": (0, int((y_holdout == 0).sum())),
    }

    for selection_index, (selection, chosen) in enumerate(selections):
        model = str(chosen["model"])
        strategy = str(chosen["strategy"])
        fitted, threshold, best_params = _fit_final(
            X_development, y_development, categorical, numeric,
            model, strategy, config,
        )
        probabilities = fitted.predict_proba(X_holdout)[:, 1]
        metric_values = _metric_row(y_holdout, probabilities, threshold)
        intervals = _bootstrap_intervals(
            y_holdout, probabilities, threshold, config.seed + 1000 + selection_index
        )

        row: dict[str, Any] = {
            "selection": selection,
            "model": model,
            "strategy": strategy,
            "cv_selection_score": float(
                chosen["pr_auc_mean"] if selection == "highest_mean_pr_auc" else chosen["mcc_mean"]
            ),
            "threshold": threshold,
            "best_params": json.dumps(best_params, sort_keys=True, default=str),
            "holdout_rows": len(y_holdout),
            "TN": int(metric_values["tn"]),
            "FP": int(metric_values["fp"]),
            "FN": int(metric_values["fn"]),
            "TP": int(metric_values["tp"]),
        }
        for metric in REPORT_METRICS:
            row[metric] = float(metric_values[metric])
            row[f"{metric}_ci95_low"] = intervals[metric][0]
            row[f"{metric}_ci95_high"] = intervals[metric][1]
        result_rows.append(row)

        print(f"{selection}: {model} / {strategy}")
        print(f"  Best hyperparameters: {json.dumps(best_params, sort_keys=True, default=str)}")
        print(f"  Decision threshold: {threshold:.6g}")

        model_fn, model_fp = int(metric_values["fn"]), int(metric_values["fp"])
        for cost_ratio in (1, 3, 5, 10):
            cost_rows.append({
                "selection": selection,
                "model": model,
                "strategy": strategy,
                "comparison": "model",
                "fn_cost_to_fp_cost": cost_ratio,
                "FN": model_fn,
                "FP": model_fp,
                "total_cost": cost_ratio * model_fn + model_fp,
            })
            for baseline, (false_negatives, false_positives) in baseline_counts.items():
                cost_rows.append({
                    "selection": selection,
                    "model": model,
                    "strategy": strategy,
                    "comparison": baseline,
                    "fn_cost_to_fp_cost": cost_ratio,
                    "FN": false_negatives,
                    "FP": false_positives,
                    "total_cost": cost_ratio * false_negatives + false_positives,
                })

        print("  Hold-out metrics:")
        for metric in REPORT_METRICS:
            low, high = intervals[metric]
            print(f"    {metric}: {metric_values[metric]:.4f} (95% CI {low:.4f}, {high:.4f})")

    holdout_results = pd.DataFrame(result_rows)
    cost_table = pd.DataFrame(cost_rows)
    holdout_results.to_csv(output_dir / "holdout_results.csv", index=False)
    cost_table.to_csv(output_dir / "cost_table.csv", index=False)
    print(f"Saved {output_dir / 'holdout_results.csv'}")
    print(f"Saved {output_dir / 'cost_table.csv'}")
    return holdout_results, cost_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/cv_summary.csv"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--trials", type=int, default=30, help="Randomized-search trials for inner tuning")
    parser.add_argument("--jobs", type=int, default=-1, help="Parallel search jobs; use 1 to disable parallelism")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if not args.summary.exists():
        parser.error(f"CV summary not found: {args.summary}")

    config = ExperimentConfig(search_trials=args.trials, n_jobs=args.jobs)
    run_final_holdout(args.data, args.summary, args.output, config)


if __name__ == "__main__":
    main()