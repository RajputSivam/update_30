# IBM HR Employee Attrition Research Workflow

This project compares leakage-aware attrition classifiers and imbalance strategies on the IBM HR Analytics dataset. The implementation is in `src/employee_attrition.py`; the runnable, paper-oriented interface is `notebooks/employee_attrition_research.ipynb`.

## Setup

Use Python 3.10 or newer. Install the dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

Download `WA_Fn-UseC_-HR-Employee-Attrition.csv` separately and place it in this directory, or change `DATA_PATH` in the notebook. The raw dataset is not included.

## Run

Open `notebooks/employee_attrition_research.ipynb` and run all cells. The default protocol uses 5 outer folds x 10 repeats, 3 inner folds, and 30 randomized-search trials with the expanded search spaces. Use `--baseline-search` for the original 12-trial spaces; run the two settings into separate output directories to compare them. Use `--smoke` for 5 outer folds x 2 repeats and 2 search trials; write smoke outputs to a separate directory so they cannot be confused with the full results.

The script can also be run from this directory:

```powershell
python employee_attrition_experiment.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv
python employee_attrition_experiment.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv --baseline-search --output .\results_baseline
python employee_attrition_experiment.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv --smoke --output .\results_smoke
python final_holdout.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv --summary .\results\cv_summary.csv
python explain_and_tests.py --data .\WA_Fn-UseC_-HR-Employee-Attrition.csv --summary .\results\cv_summary.csv
python leakage_demo.py
```

`final_holdout.py` selects the development-CV leaders by mean PR-AUC and mean MCC, tunes each on the full development split, and evaluates each once on the untouched hold-out. It writes hold-out metrics with stratified 2,000-resample bootstrap 95% confidence intervals to `results/holdout_results.csv` and cost comparisons to `results/cost_table.csv`.

`explain_and_tests.py` refits the highest-PR-AUC configuration on development data, writes hold-out SHAP outputs (`results/shap_beeswarm.png`, `results/shap_top15.png`, and `results/shap_top15.csv`), and runs exact paired McNemar tests for the top five configurations on the 236-row development comparison subset with Holm-adjusted p-values in `results/mcnemar.csv`. LIME is not run.

## Protocol

- Drops the four constant/identifier columns and retains nominal categories for one-hot encoding.
- Keeps imputation, one-hot encoding, scaling, and resampling inside the estimator pipeline. SMOTE-NC temporarily codes categories only for its categorical-aware sampler and restores their nominal labels before one-hot encoding.
- Tunes each model/strategy on 3 stratified inner folds with average precision; each of the 10 stratified outer repeats uses its own recorded random state derived from master seed 42.
- The expanded search uses 30 trials and adds LR penalty, RF tree count, XGBoost `reg_lambda`/`gamma`, and LightGBM `reg_lambda`. Set `expanded_search=False` in `ExperimentConfig` or pass `--baseline-search` for the original parameter spaces and 12 trials. The run signature distinguishes settings so checkpoints cannot be mixed; `run.log` records elapsed time per learner/strategy configuration.
- Writes `results/cv_results_raw.csv` after each learner/strategy configuration and checkpoints each outer fold for resume. `results/cv_summary.csv` contains mean and standard deviation for the eight reported metrics; `results/run_config.json` records split counts, seeds, search trials, package versions, and run status.
- Uses `n_jobs=-1` for randomized search and displays a fold-level progress bar.
- Chooses one configuration by development nested-CV mean PR-AUC. Only that selected configuration is evaluated on the untouched stratified 20% final hold-out. Cost minima for strategy comparisons therefore come from development CV; hold-out cost is reported for the selected model only.
- A separate development-only split supports paired McNemar comparisons; it cannot change the final configuration or touch the final hold-out. All preprocessing, sampling, calibration, and threshold selection are fitted within training folds.

Confidence intervals are conventional t intervals across repeated-CV fold scores. Repeated fold results are dependent, so these intervals are descriptive and should not be interpreted as independent-sample confidence intervals. Wilcoxon tests are paired on identical outer folds. Treat significance results as exploratory and report the dependence limitation in the paper.

## Outputs

All generated artifacts are placed under `results/`:

- `tables/`: fold results, mean/std/95% CI summaries, cost and leakage comparisons, significance tests, ranking, hold-out metrics, gain/lift, SHAP and LIME summaries. Main paper tables are also written as Markdown and LaTeX.
- `figures/`: development ROC/PR curves, final hold-out calibration and confusion matrix, cost-ratio and gain/lift plots, and SHAP figures.
- `final_pipeline.joblib` and `final_model_metadata.json`: selected fitted estimator, decision threshold, features, parameters, and run configuration.
