"""Compare deliberate preprocessing leakage with a training-only pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tqdm.auto import tqdm
from xgboost import XGBClassifier


DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
LEARNERS = ["Logistic Regression", "Random Forest", "XGBoost", "Stacking", "MLP"]
PIPELINES = ["A: full-data oversampling", "B: full-data PCA", "C: training-only"]
METRICS = ["accuracy", "recall", "precision", "f1", "roc_auc", "pr_auc"]
SEEDS = range(42, 52)


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    data = pd.read_csv(path).drop(columns=DROP_COLUMNS, errors="ignore")
    if "Attrition" not in data:
        raise ValueError("Expected an 'Attrition' target column.")
    y = data.pop("Attrition").map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Attrition must contain only 'No' and 'Yes'.")
    categorical = data.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in data.columns if column not in categorical]
    return data, y.astype(int), categorical, numeric


def make_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
            ("numeric", SimpleImputer(strategy="median"), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_learner(name: str, seed: int):
    if name == "Logistic Regression":
        return LogisticRegression(C=1.0, max_iter=2000, solver="liblinear", random_state=seed)
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, max_features="sqrt",
            random_state=seed, n_jobs=1,
        )
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.85,
            colsample_bytree=0.85, eval_metric="logloss", random_state=seed,
            n_jobs=1, verbosity=0,
        )
    if name == "Stacking":
        estimators = [
            ("rf", make_learner("Random Forest", seed)),
            ("xgb", make_learner("XGBoost", seed + 1)),
            ("lgbm", LGBMClassifier(
                n_estimators=200, num_leaves=15, max_depth=5, learning_rate=0.05,
                random_state=seed + 2, n_jobs=1, verbosity=-1,
            )),
        ]
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=2000, solver="liblinear",
                                               random_state=seed + 3),
            cv=3,
            stack_method="predict_proba",
            n_jobs=1,
        )
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            alpha=0.0001, max_iter=400, random_state=seed,
        )
    raise ValueError(f"Unknown learner: {name}")


def make_feature_pipeline(categorical: list[str], numeric: list[str], seed: int,
                          include_sampler: bool):
    steps = [
        ("preprocessor", make_preprocessor(categorical, numeric)),
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
    ]
    if include_sampler:
        steps.append(("sampler", RandomOverSampler(random_state=seed)))
    return steps


def metric_values(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
    }


def run_demo(data_path: Path, output_dir: Path) -> pd.DataFrame:
    X, y, categorical, numeric = load_dataset(data_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_rows = []

    for seed in tqdm(SEEDS, desc="Leakage demo", unit="seed"):
        # Oversample all rows first; reserve the original hold-out rows for scoring,
        # while their duplicated copies are allowed into A's training data.
        whole_sampler = RandomOverSampler(random_state=seed)
        X_resampled, y_resampled = whole_sampler.fit_resample(X, y)
        source_indices = whole_sampler.sample_indices_
        if not np.array_equal(source_indices[:len(y)], np.arange(len(y))):
            raise RuntimeError("RandomOverSampler did not preserve original rows first")

        # Fit preprocessing and PCA on every original row before using the split.
        global_preprocessor = make_preprocessor(categorical, numeric)
        global_encoded = global_preprocessor.fit_transform(X)
        global_scaled = StandardScaler().fit_transform(global_encoded)
        global_pca = PCA(n_components=0.95, svd_solver="full")
        global_features = global_pca.fit_transform(global_scaled)

        train_indices, test_indices = train_test_split(
            np.arange(len(y)), test_size=0.20, stratify=y, random_state=seed,
        )
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        X_test = X.iloc[test_indices]
        test_original_mask = np.zeros(len(source_indices), dtype=bool)
        test_original_mask[:len(y)] = np.isin(np.arange(len(y)), test_indices)
        X_a_train = X_resampled.iloc[~test_original_mask]
        y_a_train = y_resampled[~test_original_mask]
        leaked_test_indices = np.intersect1d(test_indices, source_indices[~test_original_mask])
        if not len(leaked_test_indices):
            raise RuntimeError("Expected full-data oversampling to leak hold-out rows into A training")
        X_b_train, X_b_test = global_features[train_indices], global_features[test_indices]

        for learner_index, learner in enumerate(LEARNERS):
            learner_seed = seed + learner_index * 100

            pipeline_a = ImbPipeline(
                make_feature_pipeline(categorical, numeric, seed, include_sampler=False)
                + [("classifier", make_learner(learner, learner_seed))]
            )
            pipeline_a.fit(X_a_train, y_a_train)
            probability_a = pipeline_a.predict_proba(X_test)[:, 1]
            raw_rows.append({
                "seed": seed, "learner": learner, "pipeline": PIPELINES[0],
                **metric_values(y_test, probability_a),
            })

            pipeline_b = ImbPipeline([
                ("sampler", RandomOverSampler(random_state=seed)),
                ("classifier", make_learner(learner, learner_seed)),
            ])
            pipeline_b.fit(X_b_train, y_train)
            probability_b = pipeline_b.predict_proba(X_b_test)[:, 1]
            raw_rows.append({
                "seed": seed, "learner": learner, "pipeline": PIPELINES[1],
                **metric_values(y_test, probability_b),
            })

            pipeline_c = ImbPipeline(
                make_feature_pipeline(categorical, numeric, seed, include_sampler=True)
                + [("classifier", make_learner(learner, learner_seed))]
            )
            pipeline_c.fit(X.iloc[train_indices], y_train)
            probability_c = pipeline_c.predict_proba(X_test)[:, 1]
            raw_rows.append({
                "seed": seed, "learner": learner, "pipeline": PIPELINES[2],
                **metric_values(y_test, probability_c),
            })

    raw = pd.DataFrame(raw_rows)
    summary_rows = []
    for (learner, pipeline), group in raw.groupby(["learner", "pipeline"], sort=False):
        row = {"learner": learner, "pipeline": pipeline, "n_seeds": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "leakage_demo.csv", index=False)
    plot_summary(summary, output_dir / "leakage_demo.png")
    return summary


def plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
    axes = axes.ravel()
    colors = ["#C45B32", "#397A68", "#3D6C8E"]
    learner_positions = np.arange(len(LEARNERS))
    bar_width = 0.24

    for axis, metric in zip(axes, METRICS):
        for pipeline_index, pipeline in enumerate(PIPELINES):
            rows = summary[summary.pipeline == pipeline].set_index("learner").loc[LEARNERS]
            positions = learner_positions + (pipeline_index - 1) * bar_width
            axis.bar(
                positions,
                rows[f"{metric}_mean"],
                width=bar_width,
                yerr=rows[f"{metric}_std"],
                capsize=2,
                color=colors[pipeline_index],
                label=pipeline,
            )
        axis.set_title(metric.replace("_", " ").upper())
        axis.set_xticks(learner_positions, LEARNERS, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8)
    figure.suptitle("Hold-out performance: leakage pipelines vs training-only fit (10 seeds)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        default=project_dir / "WA_Fn-UseC_-HR-Employee-Attrition.csv")
    parser.add_argument("--output", type=Path, default=project_dir / "results")
    args = parser.parse_args()
    summary = run_demo(args.data, args.output)
    print(f"Saved {len(summary)} learner/pipeline summaries to {args.output.resolve()}")


if __name__ == "__main__":
    main()