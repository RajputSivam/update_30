"""Explain the selected model and compare the top five configurations."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import binomtest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import train_test_split

from src.employee_attrition import (
    MODEL_NAMES,
    SEED,
    ExperimentConfig,
    FoldBalancedXGBClassifier,
    _fit_final,
    load_dataset,
)


def _explain_pipeline(estimator: Any) -> tuple[ImbPipeline, Any, str]:
    if isinstance(estimator, CalibratedClassifierCV):
        estimator = estimator.calibrated_classifiers_[0].estimator
    if isinstance(estimator, StackingClassifier):
        estimator = dict(estimator.estimators_)['rf']
        component = 'stack_rf_component'
    else:
        component = 'selected_model'
    if not isinstance(estimator, ImbPipeline):
        raise TypeError(f"SHAP does not support the fitted estimator type {type(estimator).__name__}.")

    classifier = estimator.named_steps['classifier']
    if isinstance(classifier, FoldBalancedXGBClassifier):
        classifier = classifier.estimator_
    if not hasattr(classifier, 'feature_importances_') and not hasattr(classifier, 'coef_'):
        raise TypeError(f"SHAP does not support classifier type {type(classifier).__name__}.")
    return estimator, classifier, component


def _save_shap_outputs(
    estimator: Any,
    X_development: pd.DataFrame,
    X_holdout: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> pd.DataFrame:
    pipeline, classifier, component = _explain_pipeline(estimator)
    preprocessor = pipeline.named_steps['preprocessor']
    numeric_scaler = pipeline.named_steps['scale_numeric']
    transformed_holdout = preprocessor.transform(numeric_scaler.transform(X_holdout))
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(classifier, 'feature_importances_'):
        explainer = shap.TreeExplainer(classifier)
    else:
        background_rows = X_development.sample(min(100, len(X_development)), random_state=seed)
        background = preprocessor.transform(numeric_scaler.transform(background_rows))
        explainer = shap.LinearExplainer(classifier, background)

    values = explainer.shap_values(transformed_holdout)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.shape != (len(X_holdout), len(feature_names)):
        raise ValueError(
            f"Unexpected SHAP shape {values.shape}; expected "
            f"({len(X_holdout)}, {len(feature_names)})."
        )

    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.mean(np.abs(values), axis=0),
    }).sort_values('mean_abs_shap', ascending=False)
    top_15 = importance.head(15).reset_index(drop=True)
    top_15.to_csv(output_dir / 'shap_top15.csv', index=False)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        values, transformed_holdout, feature_names=feature_names,
        show=False, max_display=20,
    )
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()

    plot_data = top_15.iloc[::-1]
    _, axis = plt.subplots(figsize=(9, 6))
    axis.barh(plot_data['feature'], plot_data['mean_abs_shap'], color='#397a68')
    axis.set(xlabel='Mean absolute SHAP value', title='Top 15 hold-out feature importances')
    axis.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_dir / 'shap_top15.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP component: {component}")
    print(f"Saved {output_dir / 'shap_beeswarm.png'}")
    print(f"Saved {output_dir / 'shap_top15.png'} and {output_dir / 'shap_top15.csv'}")
    return top_15


def _holm_adjust(p_values: list[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _mcnemar_top_five(
    summary: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_compare: pd.DataFrame,
    y_compare: pd.Series,
    categorical: list[str],
    numeric: list[str],
    config: ExperimentConfig,
) -> pd.DataFrame:
    candidates = summary[summary['model'].isin(MODEL_NAMES)].nlargest(5, 'pr_auc_mean')
    if len(candidates) != 5:
        raise ValueError(f"Expected at least five model/strategy configurations; found {len(candidates)}.")

    predictions: dict[str, np.ndarray] = {}
    for row in candidates.itertuples(index=False):
        estimator, threshold, best_params = _fit_final(
            X_train, y_train, categorical, numeric, row.model, row.strategy, config
        )
        label = f"{row.model} / {row.strategy}"
        predictions[label] = (
            estimator.predict_proba(X_compare)[:, 1] >= threshold
        ).astype(int)
        print(f"McNemar model: {label}; best_params={json.dumps(best_params, sort_keys=True, default=str)}")

    records: list[dict[str, Any]] = []
    labels = list(predictions)
    actual = y_compare.to_numpy()
    for model_a, model_b in combinations(labels, 2):
        a_correct = predictions[model_a] == actual
        b_correct = predictions[model_b] == actual
        a_correct_b_wrong = int(np.sum(a_correct & ~b_correct))
        a_wrong_b_correct = int(np.sum(~a_correct & b_correct))
        discordant = a_correct_b_wrong + a_wrong_b_correct
        p_value = (
            float(binomtest(min(a_correct_b_wrong, a_wrong_b_correct), discordant,
                            0.5, alternative='two-sided').pvalue)
            if discordant else 1.0
        )
        records.append({
            'comparison_split': 'development-only',
            'model_a': model_a,
            'model_b': model_b,
            'comparison_rows': len(actual),
            'a_correct_b_wrong': a_correct_b_wrong,
            'a_wrong_b_correct': a_wrong_b_correct,
            'discordant_pairs': discordant,
            'mcnemar_exact_p_value': p_value,
        })

    results = pd.DataFrame(records)
    results['holm_adjusted_p_value'] = _holm_adjust(results['mcnemar_exact_p_value'].tolist())
    results['reject_at_0_05'] = results['holm_adjusted_p_value'] < 0.05
    return results


def run_explanations_and_tests(
    data_path: Path,
    summary_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(summary_path)
    required = {'model', 'strategy', 'pr_auc_mean'}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"CV summary is missing required columns: {', '.join(sorted(missing))}")
    candidate_summary = summary[summary['model'].isin(MODEL_NAMES)]
    if candidate_summary.empty:
        raise ValueError('CV summary has no candidate learner/strategy rows.')

    selected = candidate_summary.nlargest(1, 'pr_auc_mean').iloc[0]
    X, y, categorical, numeric = load_dataset(data_path)
    X_development, X_holdout, y_development, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=config.seed
    )
    X_search, X_compare, y_search, y_compare = train_test_split(
        X_development, y_development, test_size=0.20, stratify=y_development,
        random_state=config.seed + 1,
    )
    if len(X_compare) != 236:
        raise ValueError(f"Expected a 236-row development comparison subset; found {len(X_compare)}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_name, strategy = str(selected['model']), str(selected['strategy'])
    selected_estimator, _, best_params = _fit_final(
        X_development, y_development, categorical, numeric,
        model_name, strategy, config,
    )
    print(f"Selected configuration: {model_name} / {strategy}")
    print(f"Best hyperparameters: {json.dumps(best_params, sort_keys=True, default=str)}")
    shap_importance = _save_shap_outputs(
        selected_estimator, X_development, X_holdout, output_dir, config.seed
    )

    mcnemar = _mcnemar_top_five(
        candidate_summary, X_search, y_search, X_compare, y_compare,
        categorical, numeric, config,
    )
    mcnemar.to_csv(output_dir / 'mcnemar.csv', index=False)
    print(f"Saved {output_dir / 'mcnemar.csv'}")
    return shap_importance, mcnemar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('WA_Fn-UseC_-HR-Employee-Attrition.csv'))
    parser.add_argument('--summary', type=Path, default=Path('results/cv_summary.csv'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--trials', type=int, default=None,
                        help='Override the profile trial count (default: 30 expanded, 12 baseline)')
    parser.add_argument('--jobs', type=int, default=-1,
                        help='Parallel search jobs; use 1 to disable parallelism')
    parser.add_argument('--baseline-search', action='store_true',
                        help='Use the original 12-trial parameter spaces')
    args = parser.parse_args()
    if args.trials is not None and args.trials < 1:
        parser.error('--trials must be at least 1')
    if not args.summary.exists():
        parser.error(f'CV summary not found: {args.summary}')

    config = ExperimentConfig(
        search_trials=args.trials,
        expanded_search=not args.baseline_search,
        n_jobs=args.jobs,
    )
    run_explanations_and_tests(args.data, args.summary, args.output, config)


if __name__ == '__main__':
    main()