"""Leakage-aware IBM HR attrition experiment and paper-output generator."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
from itertools import combinations
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from scipy.stats import binomtest, kendalltau, loguniform, rankdata, spearmanr, t
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

SEED = 42
DROP_COLUMNS = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
MODEL_NAMES = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM", "Stacking"]
STRATEGIES = [
    "No handling",
    "Class weight",
    "SMOTE-NC",
    "Random undersampling",
    "Threshold tuning",
    "Calibrated sigmoid + threshold",
    "Calibrated isotonic + threshold",
]
METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc"]
RESUME_KEYS = ("seed", "folds", "repeats", "trials")
PACKAGE_DISTRIBUTIONS = (
    "numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "lightgbm",
    "scipy", "matplotlib", "shap", "lime", "joblib",
)


@dataclass
class ExperimentConfig:
    seed: int = SEED
    outer_folds: int = 5
    outer_repeats: int = 10
    inner_folds: int = 3
    search_trials: int = 12
    n_jobs: int = -1
    shap_samples: int = 150
    shap_stability_resamples: int = 5
    lime_explanations: int = 3
    quick_run: bool = False

    @classmethod
    def quick(cls) -> "ExperimentConfig":
        return cls(outer_folds=2, outer_repeats=1, inner_folds=2, search_trials=2,
                   shap_samples=40, shap_stability_resamples=2, lime_explanations=2,
                   n_jobs=1, quick_run=True)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in PACKAGE_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "N/A"
    return versions


def _resume_signature(config: ExperimentConfig) -> dict[str, int]:
    return {
        "seed": config.seed,
        "folds": config.outer_folds,
        "repeats": config.outer_repeats,
        "trials": config.search_trials,
    }


def _prepare_run_config(output: Path, config: ExperimentConfig) -> tuple[Path, dict[str, Any]]:
    config_path = output / "run_config.json"
    signature = _resume_signature(config)
    checkpoint_dir = output / "checkpoints"
    if not config_path.exists() and checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
        raise RuntimeError("Cannot resume: checkpoints exist without run_config.json")
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as stream:
            saved = json.load(stream)
        saved_signature = {key: saved.get(key) for key in RESUME_KEYS}
        mismatches = [
            f"{key}: saved={saved_signature[key]!r}, current={signature[key]!r}"
            for key in RESUME_KEYS if saved_signature[key] != signature[key]
        ]
        if mismatches:
            raise RuntimeError("Cannot resume: run configuration differs (" + "; ".join(mismatches) + ")")
        start_time = saved.get("start_time", _utc_timestamp())
    else:
        start_time = _utc_timestamp()
    run_config = {
        "quick_run": config.quick_run,
        **signature,
        "package_versions": _package_versions(),
        "start_time": start_time,
        "end_time": None,
    }
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)
    return config_path, run_config


def _log_line(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{_utc_timestamp()} | {message}\n")


def _checkpoint_path(checkpoints: Path, model_index: int, strategy_index: int,
                     fold_index: int) -> Path:
    return checkpoints / f"unit_{model_index:02d}_{strategy_index:02d}_{fold_index:03d}.joblib"


def _progress(completed: int, total: int, started: float) -> None:
    elapsed = time.perf_counter() - started
    remaining = elapsed / completed * (total - completed) if completed else 0.0
    print(f"Completed {completed}/{total}; estimated time remaining: {remaining:.1f} seconds")


class CategoryPreservingSMOTENC(BaseEstimator):
    """Apply SMOTENC to temporary category codes, then restore nominal values."""

    _sampling_type = "over-sampling"

    def __init__(self, categorical_columns: list[str], random_state: int = SEED, k_neighbors: int = 5):
        self.categorical_columns = categorical_columns
        self.random_state = random_state
        self.k_neighbors = k_neighbors

    def _fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        frame = X.copy()
        categories: dict[str, np.ndarray] = {}
        for column in self.categorical_columns:
            values = frame[column].astype("string").fillna("__MISSING__")
            categories[column], codes = np.unique(values, return_inverse=True)
            frame[column] = codes.astype(float)
        categorical_indices = [frame.columns.get_loc(column) for column in self.categorical_columns]
        sampler = SMOTENC(categorical_features=categorical_indices, random_state=self.random_state,
                          k_neighbors=self.k_neighbors)
        sampled, labels = sampler.fit_resample(frame, y)
        sampled = pd.DataFrame(sampled, columns=X.columns)
        for column, labels_for_column in categories.items():
            codes = np.rint(sampled[column].to_numpy()).astype(int)
            sampled[column] = labels_for_column[codes]
        for column in X.columns:
            if column not in self.categorical_columns:
                sampled[column] = pd.to_numeric(sampled[column])
        return sampled, np.asarray(labels)

    def fit_resample(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
        return self._fit_resample(X, y)


class ThresholdedEstimator(BaseEstimator):
    """Persist the selected probability threshold alongside the fitted estimator."""

    def __init__(self, estimator: BaseEstimator, threshold: float = 0.5):
        self.estimator = estimator
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ThresholdedEstimator":
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


class FoldBalancedXGBClassifier(ClassifierMixin, BaseEstimator):
    """Compute scale_pos_weight from the labels seen by each fit call."""

    def __init__(self, base_estimator: XGBClassifier):
        self.base_estimator = base_estimator

    def fit(self, X: Any, y: Any) -> "FoldBalancedXGBClassifier":
        labels = np.asarray(y)
        positive_weight = float(np.sum(labels == 0) / max(np.sum(labels == 1), 1))
        self.estimator_ = clone(self.base_estimator).set_params(scale_pos_weight=positive_weight)
        self.estimator_.fit(X, labels)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.estimator_.predict(X)

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Download WA_Fn-UseC_-HR-Employee-Attrition.csv "
            "and set DATA_PATH in the notebook or pass --data."
        )
    data = pd.read_csv(data_path).reset_index(drop=True)
    if "Attrition" not in data:
        raise ValueError("Expected target column 'Attrition'.")
    data = data.drop(columns=[column for column in DROP_COLUMNS if column in data.columns])
    y = data.pop("Attrition").map({"No": 0, "Yes": 1})
    if y.isna().any():
        raise ValueError("Attrition must contain only 'No' and 'Yes'.")
    X = data
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in X.columns if column not in categorical]
    return X, y.astype(int), categorical, numeric


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", ImbPipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
            ("numeric", "passthrough", numeric),
        ], remainder="drop", verbose_feature_names_out=False,
    )


def build_numeric_scaler(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("numeric", ImbPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric)],
        remainder="passthrough", verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _classifier(name: str, weighted: bool, positive_weight: float, seed: int) -> BaseEstimator:
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2500, class_weight="balanced" if weighted else None,
                                  random_state=seed, solver="liblinear")
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=300, class_weight="balanced" if weighted else None,
                                      random_state=seed, n_jobs=1)
    if name == "XGBoost":
        classifier = XGBClassifier(n_estimators=300, eval_metric="logloss", scale_pos_weight=1.0,
                                    random_state=seed, n_jobs=1, verbosity=0)
        return FoldBalancedXGBClassifier(classifier) if weighted else classifier
    if name == "LightGBM":
        return LGBMClassifier(n_estimators=300, class_weight="balanced" if weighted else None,
                              random_state=seed, n_jobs=1, verbosity=-1)
    raise ValueError(f"Unknown base model: {name}")


def _model_pipeline(name: str, strategy: str, categorical: list[str], numeric: list[str],
                    positive_weight: float, seed: int) -> ImbPipeline:
    sampler: Any = "passthrough"
    if strategy == "SMOTE-NC":
        sampler = CategoryPreservingSMOTENC(categorical, random_state=seed, k_neighbors=3)
    elif strategy == "Random undersampling":
        sampler = RandomUnderSampler(random_state=seed)
    return ImbPipeline([
        ("scale_numeric", build_numeric_scaler(categorical, numeric)),
        ("sampler", sampler),
        ("preprocessor", build_preprocessor(categorical, numeric)),
        ("classifier", _classifier(name, strategy == "Class weight", positive_weight, seed)),
    ])


def _build_model(name: str, strategy: str, categorical: list[str], numeric: list[str],
                 positive_weight: float, seed: int, inner_folds: int) -> BaseEstimator:
    if name != "Stacking":
        estimator = _model_pipeline(name, strategy, categorical, numeric, positive_weight, seed)
    else:
        base_models = []
        for key, base_name in [("rf", "Random Forest"), ("xgb", "XGBoost"), ("lgbm", "LightGBM")]:
            base_models.append((key, _model_pipeline(base_name, strategy, categorical, numeric,
                                                     positive_weight, seed)))
        estimator = StackingClassifier(
            estimators=base_models,
            final_estimator=LogisticRegression(max_iter=2500, class_weight="balanced" if strategy == "Class weight" else None,
                                               random_state=seed),
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed),
            stack_method="predict_proba", n_jobs=1,
        )
    if strategy.startswith("Calibrated"):
        method = "sigmoid" if "sigmoid" in strategy else "isotonic"
        estimator = CalibratedClassifierCV(
            estimator=estimator, method=method,
            cv=StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed),
        )
    return estimator


def _parameter_space(name: str, strategy: str, inner_folds: int) -> dict[str, Any]:
    if name == "Stacking":
        common = {
            "rf__classifier__max_depth": [None, 10, 20],
            "rf__classifier__min_samples_leaf": [1, 2, 4],
            "xgb__classifier__max_depth": [2, 4, 6],
            "xgb__classifier__learning_rate": loguniform(0.02, 0.2),
            "lgbm__classifier__num_leaves": [7, 15, 31],
            "lgbm__classifier__learning_rate": loguniform(0.02, 0.2),
        }
    elif name == "Logistic Regression":
        common = {"classifier__C": loguniform(0.01, 100)}
    elif name == "Random Forest":
        common = {"classifier__max_depth": [None, 8, 16, 24],
                  "classifier__min_samples_leaf": [1, 2, 4],
                  "classifier__max_features": ["sqrt", 0.7, 1.0]}
    elif name == "XGBoost":
        common = {"classifier__max_depth": [2, 3, 5, 7],
                  "classifier__learning_rate": loguniform(0.015, 0.2),
                  "classifier__subsample": [0.7, 0.85, 1.0],
                  "classifier__colsample_bytree": [0.7, 0.85, 1.0],
                  "classifier__min_child_weight": [1, 3, 6]}
    else:
        common = {"classifier__num_leaves": [7, 15, 31, 63],
                  "classifier__learning_rate": loguniform(0.015, 0.2),
                  "classifier__max_depth": [-1, 5, 10, 15],
                  "classifier__min_child_samples": [10, 20, 40]}
    if strategy == "Class weight" and name == "XGBoost":
        common = {key.replace("classifier__", "classifier__base_estimator__"): value
                  for key, value in common.items()}
    if strategy == "Class weight" and name == "Stacking":
        common = {key.replace("xgb__classifier__", "xgb__classifier__base_estimator__"): value
                  for key, value in common.items()}
    if strategy.startswith("Calibrated"):
        prefix = "estimator__"
        common = {prefix + key: value for key, value in common.items()}
    return common


def _threshold_for_f1(y_true: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f1_values))])


def _metric_row(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "balanced_accuracy": balanced_accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, zero_division=0),
        "recall": recall_score(y_true, predicted, zero_division=0),
        "f1": f1_score(y_true, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "mcc": matthews_corrcoef(y_true, predicted),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def _search(estimator: BaseEstimator, model: str, strategy: str, X: pd.DataFrame, y: pd.Series,
            config: ExperimentConfig, cv: StratifiedKFold, n_jobs: int) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        estimator, _parameter_space(model, strategy, config.inner_folds),
        n_iter=config.search_trials, scoring="average_precision", cv=cv,
        random_state=config.seed, n_jobs=n_jobs, refit=True, error_score="raise",
    ).fit(X, y)


def _crossfit_tuned_probabilities(model: str, strategy: str, X: pd.DataFrame, y: pd.Series,
                                 categorical: list[str], numeric: list[str],
                                 config: ExperimentConfig, seed: int) -> np.ndarray:
    """Generate threshold-selection probabilities without tuning on each validation fold."""
    threshold_cv = StratifiedKFold(n_splits=config.inner_folds, shuffle=True, random_state=seed)
    probabilities = np.empty(len(y), dtype=float)
    for split_index, (fit_indices, valid_indices) in enumerate(threshold_cv.split(X, y)):
        X_fit, X_valid = X.iloc[fit_indices], X.iloc[valid_indices]
        y_fit, y_valid = y.iloc[fit_indices], y.iloc[valid_indices]
        fold_weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        tuning_cv = StratifiedKFold(n_splits=config.inner_folds, shuffle=True,
                                    random_state=seed + split_index + 1)
        estimator = _build_model(model, strategy, categorical, numeric, fold_weight,
                                 seed + split_index + 1, config.inner_folds)
        search = _search(estimator, model, strategy, X_fit, y_fit, config, tuning_cv,
                         config.n_jobs)
        probabilities[valid_indices] = search.best_estimator_.predict_proba(X_valid)[:, 1]
    return probabilities


def _nested_cv(X: pd.DataFrame, y: pd.Series, categorical: list[str], numeric: list[str],
               config: ExperimentConfig, output: Path, log_path: Path
               ) -> tuple[pd.DataFrame, dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]]:
    outer = RepeatedStratifiedKFold(n_splits=config.outer_folds, n_repeats=config.outer_repeats,
                                    random_state=config.seed)
    splits = list(outer.split(X, y))
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    total_units = len(MODEL_NAMES) * len(STRATEGIES) * len(splits)
    completed_units = 0
    progress_started = time.perf_counter()
    folds: list[dict[str, Any]] = []
    curves: dict[tuple[str, str], tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for model_index, model in enumerate(MODEL_NAMES):
        for strategy_index, strategy in enumerate(STRATEGIES):
            key = (model, strategy)
            curves[key] = ([], [])
            for fold_index, (train_indices, valid_indices) in enumerate(splits):
                checkpoint_path = _checkpoint_path(checkpoints, model_index, strategy_index, fold_index)
                unit = f"model={model}; strategy={strategy}; outer_fold={fold_index}"
                if checkpoint_path.exists():
                    checkpoint = joblib.load(checkpoint_path)
                    if checkpoint.get("signature") != _resume_signature(config):
                        raise RuntimeError(f"Cannot resume: checkpoint configuration differs for {unit}")
                    row = checkpoint["row"]
                    y_valid_values = np.asarray(checkpoint["y_valid"])
                    probabilities = np.asarray(checkpoint["probabilities"])
                    folds.append(row)
                    curves[key][0].append(y_valid_values)
                    curves[key][1].append(probabilities)
                    completed_units += 1
                    _log_line(log_path, f"SKIP {unit}; checkpoint={checkpoint_path.name}")
                    _progress(completed_units, total_units, progress_started)
                    continue
                unit_started = time.perf_counter()
                _log_line(log_path, f"START {unit}")
                X_train, X_valid = X.iloc[train_indices], X.iloc[valid_indices]
                y_train, y_valid = y.iloc[train_indices], y.iloc[valid_indices]
                inner = StratifiedKFold(n_splits=config.inner_folds, shuffle=True,
                                        random_state=config.seed + fold_index)
                positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
                estimator = _build_model(model, strategy, categorical, numeric, positive_weight,
                                         config.seed + fold_index, config.inner_folds)
                search = _search(estimator, model, strategy, X_train, y_train, config, inner,
                                 config.n_jobs)
                threshold = 0.5
                if "threshold" in strategy.lower():
                    oof = _crossfit_tuned_probabilities(model, strategy, X_train, y_train,
                                                        categorical, numeric, config,
                                                        config.seed + fold_index)
                    threshold = _threshold_for_f1(y_train, oof)
                probabilities = search.best_estimator_.predict_proba(X_valid)[:, 1]
                row = {"model": model, "strategy": strategy, "fold": fold_index,
                       "repeat": fold_index // config.outer_folds, "threshold": threshold}
                row.update(_metric_row(y_valid, probabilities, threshold))
                joblib.dump({
                    "signature": _resume_signature(config),
                    "row": row,
                    "y_valid": y_valid.to_numpy(),
                    "probabilities": probabilities,
                }, checkpoint_path)
                folds.append(row)
                curves[key][0].append(y_valid.to_numpy())
                curves[key][1].append(probabilities)
                completed_units += 1
                elapsed = time.perf_counter() - unit_started
                _log_line(log_path, f"FINISH {unit}; elapsed_seconds={elapsed:.3f}")
                _progress(completed_units, total_units, progress_started)
    majority = int(y.value_counts().idxmax())
    for fold_index, (_, valid_indices) in enumerate(splits):
        y_valid = y.iloc[valid_indices]
        constant_probabilities = np.full(len(y_valid), float(majority))
        baseline_metrics = _metric_row(y_valid, constant_probabilities, threshold=0.5)
        row = {
            "model": "Majority baseline",
            "strategy": "Majority class",
            "fold": fold_index,
            "repeat": fold_index // config.outer_folds,
            "threshold": 1.0,
            **baseline_metrics,
        }
        folds.append(row)
    return pd.DataFrame(folds), {key: (np.concatenate(value[0]), np.concatenate(value[1]))
                                  for key, value in curves.items()}


def _summary_table(folds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, strategy), group in folds.groupby(["model", "strategy"], sort=False):
        row: dict[str, Any] = {"model": model, "strategy": strategy, "n_folds": len(group)}
        for metric in METRICS:
            values = group[metric].dropna().to_numpy()
            if not len(values):
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            margin = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
            row[f"{metric}_ci95_low"] = mean - margin
            row[f"{metric}_ci95_high"] = mean + margin
        rows.append(row)
    return pd.DataFrame(rows)


def _save_table(table: pd.DataFrame, path: Path) -> None:
    table.to_csv(path, index=False)
    path.with_suffix(".md").write_text(table.to_markdown(index=False), encoding="utf-8")
    path.with_suffix(".tex").write_text(table.to_latex(index=False, float_format="%.3f"), encoding="utf-8")


def _significance_tests(folds: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scipy.stats import wilcoxon

    tests = []
    for model in MODEL_NAMES:
        for left_strategy, right_strategy in combinations(STRATEGIES, 2):
            left = folds[(folds.model == model) & (folds.strategy == left_strategy)].sort_values("fold")
            right = folds[(folds.model == model) & (folds.strategy == right_strategy)].sort_values("fold")
            if len(left) != len(right):
                continue
            for metric in ["pr_auc", "f1"]:
                a, b = left[metric].to_numpy(), right[metric].to_numpy()
                try:
                    stat, p_value = wilcoxon(a, b, zero_method="zsplit")
                except ValueError:
                    stat, p_value = 0.0, 1.0
                tests.append({"model": model, "comparison": f"{left_strategy} vs {right_strategy}",
                              "metric": metric, "wilcoxon_statistic": stat, "p_value": p_value})
    tests_frame = pd.DataFrame(tests)
    if len(tests_frame):
        order = np.argsort(tests_frame["p_value"].to_numpy())
        ranks = np.arange(1, len(order) + 1)
        adjusted = tests_frame.loc[order, "p_value"].to_numpy() * len(order) / ranks
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1.0)
        tests_frame.loc[order, "bh_q_value"] = adjusted
    ranking = folds[folds.model.isin(MODEL_NAMES)].copy()
    ranking["fold_rank"] = ranking.groupby("fold")["pr_auc"].rank(ascending=False, method="average")
    ranks = ranking.groupby(["model", "strategy"], as_index=False)["fold_rank"].mean()
    return tests_frame, ranks.rename(columns={"fold_rank": "average_rank_across_folds"})


def _cost_analysis(folds: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    development = folds[folds.model.isin(MODEL_NAMES)].groupby(["model", "strategy"])[["fn", "fp"]].sum()
    rows = []
    for ratio in [1, 3, 5, 10]:
        for (model, strategy), counts in development.iterrows():
            total_cost = ratio * counts.fn + counts.fp
            evaluated_employees = folds.loc[folds.model == model, ["tn", "fp", "fn", "tp"]].to_numpy().sum()
            rows.append({"data_scope": "development CV aggregate", "fn_cost_to_fp_cost": f"{ratio}:1",
                         "model": model, "strategy": strategy, "total_cost": float(total_cost),
                         "cost_per_employee_fold": float(total_cost / evaluated_employees)})
    table = pd.DataFrame(rows)
    table["cheapest_development_configuration"] = False
    for _, indexes in table.groupby("fn_cost_to_fp_cost").groups.items():
        index = table.loc[indexes, "total_cost"].idxmin()
        table.loc[index, "cheapest_development_configuration"] = True
    return table


def _plot_cost_curves(costs: pd.DataFrame, figures: Path) -> None:
    ratios = ["1:1", "3:1", "5:1", "10:1"]
    development = costs[costs.data_scope == "development CV aggregate"]
    pivot = development.pivot(index=["model", "strategy"], columns="fn_cost_to_fp_cost",
                              values="cost_per_employee_fold")
    top = pivot.mean(axis=1).nsmallest(6).index
    fig, axis = plt.subplots(figsize=(8, 5))
    for model, strategy in top:
        axis.plot([1, 3, 5, 10], [pivot.loc[(model, strategy), ratio] for ratio in ratios],
                  marker="o", label=f"{model} / {strategy}")
    axis.set(xlabel="Missed-leaver cost / unnecessary-intervention cost",
             ylabel="Expected cost per employee-fold", title="Development CV cost vs. cost ratio")
    axis.legend(fontsize=7); axis.grid(alpha=0.25); fig.tight_layout()
    fig.savefig(figures / "cost_ratio_curves.png", dpi=300); plt.close(fig)


def _plot_average_ranks(ranks: pd.DataFrame, figures: Path) -> None:
    ordered = ranks.sort_values("average_rank_across_folds")
    fig, axis = plt.subplots(figsize=(9, max(4, 0.24 * len(ordered))))
    labels = [f"{row.model} / {row.strategy}" for row in ordered.itertuples(index=False)]
    axis.barh(labels, ordered["average_rank_across_folds"], color="#397a68")
    axis.invert_yaxis()
    axis.set(xlabel="Average PR-AUC rank across outer folds (lower is better)",
             title="Critical-difference-style ranking")
    axis.grid(axis="x", alpha=0.2); fig.tight_layout()
    fig.savefig(figures / "average_rank_by_fold.png", dpi=300); plt.close(fig)


def _mcnemar_comparison(summary: pd.DataFrame, X_train: pd.DataFrame, y_train: pd.Series,
                        X_compare: pd.DataFrame, y_compare: pd.Series, categorical: list[str],
                        numeric: list[str], config: ExperimentConfig, tables: Path) -> pd.DataFrame:
    candidates = summary[summary.model.isin(MODEL_NAMES)].nlargest(3, "pr_auc_mean")
    predictions: dict[str, np.ndarray] = {}
    for row in candidates.itertuples(index=False):
        estimator, threshold, _ = _fit_final(X_train, y_train, categorical, numeric,
                                              row.model, row.strategy, config)
        label = f"{row.model} / {row.strategy}"
        predictions[label] = (estimator.predict_proba(X_compare)[:, 1] >= threshold).astype(int)
    records = []
    labels = list(predictions)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1:]:
            left_correct = predictions[left] == y_compare.to_numpy()
            right_correct = predictions[right] == y_compare.to_numpy()
            n01 = int(np.sum(left_correct & ~right_correct))
            n10 = int(np.sum(~left_correct & right_correct))
            discordant = n01 + n10
            p_value = float(binomtest(min(n01, n10), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
            records.append({"comparison_split": "development-only; not final hold-out",
                            "model_a": left, "model_b": right, "a_correct_b_wrong": n01,
                            "a_wrong_b_correct": n10, "mcnemar_exact_p_value": p_value})
    result = pd.DataFrame(records)
    _save_table(result, tables / "mcnemar_comparison.csv")
    return result


def _plot_cv_curves(curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
                    summary: pd.DataFrame, figures: Path) -> None:
    top = summary[summary.model.isin(MODEL_NAMES)].nlargest(3, "pr_auc_mean")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for row in top.itertuples(index=False):
        labels, scores = curves[(row.model, row.strategy)]
        fpr, tpr, _ = roc_curve(labels, scores)
        precision, recall, _ = precision_recall_curve(labels, scores)
        label = f"{row.model} / {row.strategy} (AP={row.pr_auc_mean:.3f})"
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (development OOF)")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (development OOF)")
    for axis in axes:
        axis.legend(fontsize=7)
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures / "roc_pr_top_configurations.png", dpi=300)
    plt.close(fig)


def _fit_final(X: pd.DataFrame, y: pd.Series, categorical: list[str], numeric: list[str],
               model: str, strategy: str, config: ExperimentConfig) -> tuple[BaseEstimator, float, dict[str, Any]]:
    inner = StratifiedKFold(n_splits=config.inner_folds, shuffle=True, random_state=config.seed)
    positive_weight = float((y == 0).sum() / max((y == 1).sum(), 1))
    estimator = _build_model(model, strategy, categorical, numeric, positive_weight,
                             config.seed, config.inner_folds)
    search = _search(estimator, model, strategy, X, y, config, inner, config.n_jobs)
    threshold = 0.5
    if "threshold" in strategy.lower():
        oof = _crossfit_tuned_probabilities(model, strategy, X, y, categorical, numeric,
                                            config, config.seed + 1)
        threshold = _threshold_for_f1(y, oof)
    return search.best_estimator_, threshold, search.best_params_


def _holdout_outputs(final_model: BaseEstimator, threshold: float, X_test: pd.DataFrame,
                     y_test: pd.Series, output: Path, figures: Path, tables: Path) -> None:
    probabilities = final_model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    print(f"Final hold-out confusion matrix counts: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    pd.DataFrame([{"TN": tn, "FP": fp, "FN": fn, "TP": tp, "threshold": threshold,
                   **_metric_row(y_test, probabilities, threshold)}]).to_csv(tables / "final_holdout_metrics.csv", index=False)
    matrix = np.array([[tn, fp], [fn, tp]])
    fig, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, str(value), ha="center", va="center", color="black", fontsize=13)
    axis.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Predicted stay", "Predicted leave"],
             yticklabels=["Actual stay", "Actual leave"], title="Final hold-out confusion matrix")
    fig.colorbar(image, ax=axis, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figures / "final_confusion_matrix.png", dpi=300)
    plt.close(fig)

    fraction_positive, mean_predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    fig, axis = plt.subplots(figsize=(5, 4))
    axis.plot(mean_predicted, fraction_positive, "o-", label="Final model")
    axis.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axis.set(xlabel="Mean predicted risk", ylabel="Observed attrition rate", title="Hold-out calibration")
    axis.legend(); axis.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(figures / "calibration_curve.png", dpi=300); plt.close(fig)

    cost_rows = []
    for ratio in [1, 3, 5, 10]:
        cost_rows.append({"data_scope": "final hold-out; selected configuration only",
                          "fn_cost_to_fp_cost": f"{ratio}:1", "model": "selected CV configuration",
                          "strategy": "selected CV configuration", "FN": int(fn), "FP": int(fp),
                          "expected_total_cost": float(ratio * fn + fp)})
    pd.DataFrame(cost_rows).to_csv(tables / "final_holdout_cost.csv", index=False)

    ranked = np.argsort(-probabilities)
    gain_rows = []
    for percentage in [10, 20, 30]:
        count = max(1, int(np.ceil(len(y_test) * percentage / 100)))
        caught = int(y_test.iloc[ranked[:count]].sum())
        gain_rows.append({"top_percent": percentage, "employees_flagged": count, "leavers_caught": caught,
                          "recall_at_top_percent": caught / max(int(y_test.sum()), 1),
                          "leavers_per_100_flagged": 100 * caught / count})
    pd.DataFrame(gain_rows).to_csv(tables / "holdout_gain_lift.csv", index=False)
    x = np.arange(1, len(y_test) + 1)
    cumulative = np.cumsum(y_test.iloc[ranked].to_numpy()) / max(int(y_test.sum()), 1)
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot(100 * x / len(x), cumulative, label="Risk-ranked employees")
    axis.plot([0, 100], [0, 1], "k--", label="Random ranking")
    axis.set(xlabel="Employees flagged (%)", ylabel="Leavers caught (%)", title="Hold-out cumulative gain")
    axis.legend(); axis.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(figures / "gain_lift_chart.png", dpi=300); plt.close(fig)
    lift = [row["leavers_per_100_flagged"] / (100 * float(y_test.mean())) for row in gain_rows]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.plot([row["top_percent"] for row in gain_rows], lift, "o-", color="#c45b32")
    axis.axhline(1, color="black", linestyle="--", linewidth=0.8)
    axis.set(xlabel="Top-risk group flagged (%)", ylabel="Lift over prevalence",
             title="Hold-out lift at top risk cutoffs")
    axis.grid(alpha=0.2); fig.tight_layout()
    fig.savefig(figures / "lift_chart.png", dpi=300); plt.close(fig)


def _leakage_demo(X: pd.DataFrame, y: pd.Series, categorical: list[str], numeric: list[str],
                  output_tables: Path, config: ExperimentConfig) -> pd.DataFrame:
    """Demonstrate oversampling-before-split on development data only, never final hold-out."""
    X_fit, X_valid, y_fit, y_valid = train_test_split(X, y, test_size=0.25, stratify=y,
                                                       random_state=config.seed)
    correct_rows, leaky_rows = [], []
    # The leaky split intentionally permits related synthetic observations across partitions.
    full_sampler = CategoryPreservingSMOTENC(categorical, random_state=config.seed, k_neighbors=3)
    X_resampled, y_resampled = full_sampler.fit_resample(X, y)
    X_leaky_fit, X_leaky_valid, y_leaky_fit, y_leaky_valid = train_test_split(
        X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=config.seed)
    for model in MODEL_NAMES:
        weight = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))
        correct = _build_model(model, "No handling", categorical, numeric, weight, config.seed, config.inner_folds)
        leaky = _build_model(model, "No handling", categorical, numeric, weight, config.seed, config.inner_folds)
        # Fixed baseline settings keep this diagnostic inexpensive and paired by model/seed.
        correct.fit(X_fit, y_fit)
        leaky.fit(X_leaky_fit, y_leaky_fit)
        for label, estimator, X_eval, y_eval, destination in [
            ("Correct pipeline", correct, X_valid, y_valid, correct_rows),
            ("Leaky SMOTE before split", leaky, X_leaky_valid, y_leaky_valid, leaky_rows),
        ]:
            proba = estimator.predict_proba(X_eval)[:, 1]
            pred = proba >= 0.5
            destination.append({"model": model, "pipeline": label,
                                "accuracy": accuracy_score(y_eval, pred),
                                "recall": recall_score(y_eval, pred, zero_division=0),
                                "roc_auc": roc_auc_score(y_eval, proba)})
    table = pd.DataFrame(correct_rows + leaky_rows)
    pivots = []
    for model, group in table.groupby("model"):
        correct = group[group.pipeline == "Correct pipeline"].iloc[0]
        leaky = group[group.pipeline == "Leaky SMOTE before split"].iloc[0]
        pivots.append({"model": model,
                       **{f"correct_{metric}": correct[metric] for metric in ["accuracy", "recall", "roc_auc"]},
                       **{f"leaky_{metric}": leaky[metric] for metric in ["accuracy", "recall", "roc_auc"]},
                       **{f"inflation_{metric}": leaky[metric] - correct[metric] for metric in ["accuracy", "recall", "roc_auc"]}})
    result = pd.DataFrame(pivots)
    _save_table(result, output_tables / "leakage_comparison.csv")
    return result


def _shap_explain(final_estimator: BaseEstimator, model_name: str, strategy: str, X_train: pd.DataFrame,
                  X_test: pd.DataFrame, categorical: list[str], figures: Path, tables: Path,
                  config: ExperimentConfig) -> None:
    """Explain the final pipeline or a named RF component when the selected model is a stack."""
    fitted = final_estimator
    if isinstance(fitted, CalibratedClassifierCV):
        fitted = fitted.calibrated_classifiers_[0].estimator
    if isinstance(fitted, StackingClassifier):
        fitted = dict(fitted.estimators_)["rf"]
        component = "stack_rf_component"
    else:
        component = model_name.lower().replace(" ", "_")
    if not isinstance(fitted, ImbPipeline):
        warnings.warn("SHAP skipped: final estimator does not expose the expected pipeline.")
        return
    prep = fitted.named_steps["preprocessor"]
    classifier = fitted.named_steps["classifier"]
    sample = X_test.iloc[:min(config.shap_samples, len(X_test))]
    scaled_sample = fitted.named_steps["scale_numeric"].transform(sample)
    transformed = prep.transform(scaled_sample)
    feature_names = prep.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        elif getattr(values, "ndim", 0) == 3:
            values = values[:, :, -1]
    else:
        background_sample = X_train.sample(min(100, len(X_train)), random_state=config.seed)
        background = prep.transform(fitted.named_steps["scale_numeric"].transform(background_sample))
        explainer = shap.LinearExplainer(classifier, background)
        values = explainer.shap_values(transformed)
        if getattr(values, "ndim", 0) == 3:
            values = values[:, :, -1]
    values = np.asarray(values)
    mean_abs = np.mean(np.abs(values), axis=0)
    importance = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False)
    importance.to_csv(tables / "shap_importance_final.csv", index=False)
    suffix = component
    plt.figure(figsize=(8, 6))
    shap.summary_plot(values, transformed, feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout(); plt.savefig(figures / f"shap_summary_{suffix}.png", dpi=300, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(7, 5))
    shap.summary_plot(values, transformed, feature_names=feature_names, plot_type="bar", show=False, max_display=10)
    plt.tight_layout(); plt.savefig(figures / f"shap_top10_{suffix}.png", dpi=300, bbox_inches="tight"); plt.close()
    for feature in importance.head(3)["feature"]:
        plt.figure(figsize=(6, 4))
        shap.dependence_plot(feature, values, transformed, feature_names=feature_names, show=False)
        plt.tight_layout(); plt.savefig(figures / f"shap_dependence_{feature[:40]}.png", dpi=300, bbox_inches="tight"); plt.close()
    importance["rank"] = rankdata(-importance["mean_abs_shap"], method="average")
    stability = []
    rng = np.random.default_rng(config.seed)
    for resample in range(config.shap_stability_resamples):
        indices = rng.choice(len(values), size=len(values), replace=True)
        resample_importance = np.mean(np.abs(values[indices]), axis=0)
        ranks = rankdata(-resample_importance, method="average")
        stability.append(ranks)
    if stability:
        reference = importance["rank"].to_numpy()
        rows = []
        top_five_counts = np.zeros(len(reference), dtype=int)
        for ranks in stability:
            top_five_counts += ranks <= 5
            rows.append({"spearman": spearmanr(reference, ranks).statistic,
                         "kendall": kendalltau(reference, ranks).statistic})
        importance["top5_frequency"] = top_five_counts / len(stability)
        importance.to_csv(tables / "shap_stability.csv", index=False)
        pd.DataFrame(rows).to_csv(tables / "shap_rank_correlations.csv", index=False)
    _ = strategy, categorical


def _shap_strategy_comparison(model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                              categorical: list[str], numeric: list[str], tables: Path,
                              config: ExperimentConfig) -> pd.DataFrame:
    rankings = []
    reference = X_train.sample(min(config.shap_samples, len(X_train)), random_state=config.seed)
    background = X_train.sample(min(100, len(X_train)), random_state=config.seed + 1)
    for strategy in ["SMOTE-NC", "Class weight", "Threshold tuning"]:
        estimator, _, _ = _fit_final(X_train, y_train, categorical, numeric,
                                     model_name, strategy, config)
        component = estimator
        if isinstance(component, CalibratedClassifierCV):
            component = component.calibrated_classifiers_[0].estimator
        if isinstance(component, StackingClassifier):
            component = dict(component.estimators_)["rf"]
        preprocessor = component.named_steps["preprocessor"]
        classifier = component.named_steps["classifier"]
        scaler = component.named_steps["scale_numeric"]
        transformed = preprocessor.transform(scaler.transform(reference))
        names = preprocessor.get_feature_names_out()
        if hasattr(classifier, "feature_importances_"):
            values = shap.TreeExplainer(classifier).shap_values(transformed)
            if isinstance(values, list):
                values = values[-1]
            elif getattr(values, "ndim", 0) == 3:
                values = values[:, :, -1]
        else:
            explainer = shap.LinearExplainer(classifier, preprocessor.transform(scaler.transform(background)))
            values = explainer.shap_values(transformed)
            if getattr(values, "ndim", 0) == 3:
                values = values[:, :, -1]
        scores = np.mean(np.abs(np.asarray(values)), axis=0)
        ranks = rankdata(-scores, method="average")
        rankings.extend({"strategy": strategy, "feature": feature, "mean_abs_shap": float(score),
                         "rank": float(rank)} for feature, score, rank in zip(names, scores, ranks))
    result = pd.DataFrame(rankings)
    result.to_csv(tables / "shap_strategy_rankings.csv", index=False)
    wide = result.pivot(index="feature", columns="strategy", values="rank")
    comparisons = []
    for strategy in ["SMOTE-NC", "Class weight"]:
        aligned = wide[[strategy, "Threshold tuning"]].dropna()
        comparisons.append({"comparison": f"{strategy} vs Threshold tuning",
                            "spearman_rank_correlation": spearmanr(aligned[strategy], aligned["Threshold tuning"]).statistic,
                            "kendall_rank_correlation": kendalltau(aligned[strategy], aligned["Threshold tuning"]).statistic,
                            "features_in_both_top5": int(((aligned[strategy] <= 5) &
                                                            (aligned["Threshold tuning"] <= 5)).sum())})
    _save_table(pd.DataFrame(comparisons), tables / "shap_strategy_rank_correlations.csv")
    return result


def _lime_explain(final_estimator: BaseEstimator, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  categorical: list[str], tables: Path, config: ExperimentConfig) -> None:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        warnings.warn("LIME is unavailable; install the project requirements to generate local explanations.")
        return
    columns = list(X_train.columns)
    category_indices = [columns.index(column) for column in categorical]
    categories: dict[int, list[str]] = {}
    encoded_train = X_train.copy()
    encoded_test = X_test.copy()
    for column in categorical:
        values = sorted(X_train[column].astype(str).unique().tolist())
        unknown = "__UNKNOWN__"
        mapping = {value: index for index, value in enumerate(values)}
        categories[columns.index(column)] = values + [unknown]
        unknown_index = len(values)
        encoded_train[column] = X_train[column].astype(str).map(mapping).fillna(unknown_index)
        encoded_test[column] = X_test[column].astype(str).map(mapping).fillna(unknown_index)
    explainer = LimeTabularExplainer(encoded_train.to_numpy(dtype=float), feature_names=columns,
                                     categorical_features=category_indices, categorical_names=categories,
                                     class_names=["Stay", "Leave"], discretize_continuous=True,
                                     random_state=config.seed)
    def predict_fn(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=columns)
        for column in categorical:
            idx = columns.index(column)
            labels = categories[idx]
            codes = np.clip(np.rint(frame[column]).astype(int), 0, len(labels) - 1)
            frame[column] = [labels[value] for value in codes]
        return final_estimator.predict_proba(frame)
    output = []
    for index in range(min(config.lime_explanations, len(encoded_test))):
        explanation = explainer.explain_instance(encoded_test.iloc[index].to_numpy(dtype=float),
                                                  predict_fn, num_features=10, labels=(1,))
        output.append({"test_row": int(index), "prediction": float(final_estimator.predict_proba(X_test.iloc[[index]])[0, 1]),
                       "explanation": json.dumps(explanation.as_list(label=1))})
    pd.DataFrame(output).to_csv(tables / "lime_individual_explanations.csv", index=False)


def run_experiment(data_path: str | Path, output_dir: str | Path = "results",
                   config: ExperimentConfig | None = None) -> dict[str, Any]:
    config = config or ExperimentConfig()
    np.random.seed(config.seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path, run_config = _prepare_run_config(output, config)
    log_path = output / "run.log"
    _log_line(log_path, f"RUN_START quick_run={config.quick_run}; output={output.resolve()}")
    tables, figures = output / "tables", output / "figures"
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    X, y, categorical, numeric = load_dataset(Path(data_path))
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=0.20, stratify=y,
                                                     random_state=config.seed)
    print(f"Rows={len(X)}; development={len(X_dev)}; final hold-out={len(X_test)}; "
          f"attrition={y.mean():.3f}; quick_run={config.quick_run}")
    X_search, X_compare, y_search, y_compare = train_test_split(
        X_dev, y_dev, test_size=0.20, stratify=y_dev, random_state=config.seed + 1)
    folds, curves = _nested_cv(X_search.reset_index(drop=True), y_search.reset_index(drop=True),
                               categorical, numeric, config, output, log_path)
    folds.to_csv(tables / "fold_level_results.csv", index=False)
    summary = _summary_table(folds)
    summary.to_csv(tables / "main_comparison.csv", index=False)
    summary.to_markdown(tables / "main_comparison.md", index=False)
    summary.to_latex(tables / "main_comparison.tex", index=False, float_format="%.3f")
    tests, ranks = _significance_tests(folds, summary)
    _save_table(tests, tables / "significance_tests.csv")
    _save_table(ranks, tables / "critical_difference_style_ranks.csv")
    costs = _cost_analysis(folds, summary)
    _save_table(costs, tables / "cost_analysis_development.csv")
    _plot_cost_curves(costs, figures)
    _plot_average_ranks(ranks, figures)
    _plot_cv_curves(curves, summary, figures)
    leakage = _leakage_demo(X_dev, y_dev, categorical, numeric, tables, config)

    selectable = summary[summary.model.isin(MODEL_NAMES)].sort_values("pr_auc_mean", ascending=False)
    mcnemar = _mcnemar_comparison(summary, X_search, y_search, X_compare, y_compare,
                                  categorical, numeric, config, tables)
    chosen = selectable.iloc[0]
    model_name, strategy = str(chosen.model), str(chosen.strategy)
    print(f"Selected using development nested-CV PR-AUC only: {model_name} / {strategy}")
    fitted, threshold, best_params = _fit_final(X_dev, y_dev, categorical, numeric,
                                                 model_name, strategy, config)
    thresholded = ThresholdedEstimator(fitted, threshold).fit(X_dev, y_dev)
    joblib.dump(thresholded, output / "final_pipeline.joblib")
    with (output / "final_model_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump({"model": model_name, "strategy": strategy, "threshold": threshold,
                   "selected_by": "nested development CV mean PR-AUC", "best_params": best_params,
                   "config": asdict(config), "categorical_features": categorical,
                   "numeric_features": numeric}, stream, indent=2)
    _holdout_outputs(fitted, threshold, X_test, y_test, output, figures, tables)
    _shap_explain(fitted, model_name, strategy, X_dev, X_test, categorical, figures, tables, config)
    _shap_strategy_comparison(model_name, X_dev, y_dev, categorical, numeric, tables, config)
    _lime_explain(fitted, X_dev, X_test, categorical, tables, config)
    costs.to_csv(tables / "cost_analysis_development.csv", index=False)
    run_config["end_time"] = _utc_timestamp()
    with config_path.open("w", encoding="utf-8") as stream:
        json.dump(run_config, stream, indent=2)
    _log_line(log_path, "RUN_FINISH")
    print(f"All artifacts saved beneath: {output.resolve()}")
    return {"folds": folds, "summary": summary, "costs": costs, "leakage": leakage,
            "mcnemar": mcnemar,
            "selected_model": model_name, "selected_strategy": strategy,
            "threshold": threshold, "final_model": thresholded}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="IBM attrition CSV path")
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="Use 2-fold/1-repeat CV and 2 search trials")
    args = parser.parse_args()
    run_experiment(args.data, args.output, ExperimentConfig.quick() if args.quick else ExperimentConfig())


if __name__ == "__main__":
    main()