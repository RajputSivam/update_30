# IBM HR Employee Attrition Research Workflow

This project compares leakage-aware attrition classifiers and imbalance strategies on the IBM HR Analytics dataset. The implementation is in `src/employee_attrition.py`; the runnable, paper-oriented interface is `notebooks/employee_attrition_research.ipynb`.

## Setup

Use Python 3.10 or newer. Install the dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` separately and place it in this directory, or change `DATA_PATH` in the notebook. The raw dataset is not included.

## Run

Open `notebooks/employee_attrition_research.ipynb` and run all cells. Set `QUICK_RUN = True` for a short smoke run; leave it `False` for the specified 5-fold x 10-repeat nested CV and randomized search. The quick setting is for debugging only and is not suitable for paper results.

The script can also be run from this directory:

```powershell
python employee_attrition_experiment.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv
python employee_attrition_experiment.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv --quick
```

## Protocol

- Drops the four constant/identifier columns and retains nominal categories for one-hot encoding.
- Keeps imputation, one-hot encoding, scaling, and resampling inside the estimator pipeline. SMOTE-NC temporarily codes categories only for its categorical-aware sampler and restores their nominal labels before one-hot encoding.
- Tunes each model/strategy on inner folds with average precision; repeated stratified outer folds estimate performance. The fixed seed is 42.
- Chooses one configuration by development nested-CV mean PR-AUC. Only that selected configuration is evaluated on the untouched stratified 20% final hold-out. Cost minima for strategy comparisons therefore come from development CV; hold-out cost is reported for the selected model only.
- A separate development-only split supports a leakage demonstration and paired McNemar comparisons; neither can change the final configuration or touch the final hold-out.

Confidence intervals are conventional t intervals across repeated-CV fold scores. Repeated fold results are dependent, so these intervals are descriptive and should not be interpreted as independent-sample confidence intervals. Wilcoxon tests are paired on identical outer folds. Treat significance results as exploratory and report the dependence limitation in the paper.

## Outputs

All generated artifacts are placed under `results/`:

- `tables/`: fold results, mean/std/95% CI summaries, cost and leakage comparisons, significance tests, ranking, hold-out metrics, gain/lift, SHAP and LIME summaries. Main paper tables are also written as Markdown and LaTeX.
- `figures/`: development ROC/PR curves, final hold-out calibration and confusion matrix, cost-ratio and gain/lift plots, and SHAP figures.
- `final_pipeline.joblib` and `final_model_metadata.json`: selected fitted estimator, decision threshold, features, parameters, and run configuration.

The leakage demonstration intentionally oversamples before an internal development split. Its leaky scores are diagnostic only and are never used for model selection or reported as valid performance.