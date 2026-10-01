| model               | strategy                        |   average_rank_across_folds |
|:--------------------|:--------------------------------|----------------------------:|
| LightGBM            | Calibrated isotonic + threshold |                       17.84 |
| LightGBM            | Calibrated sigmoid + threshold  |                       15.94 |
| LightGBM            | Class weight                    |                       21.04 |
| LightGBM            | No handling                     |                       20.92 |
| LightGBM            | Random undersampling            |                       30.58 |
| LightGBM            | SMOTE-NC                        |                       22.82 |
| LightGBM            | Threshold tuning                |                       21.6  |
| Logistic Regression | Calibrated isotonic + threshold |                       10.58 |
| Logistic Regression | Calibrated sigmoid + threshold  |                        6.6  |
| Logistic Regression | Class weight                    |                       10.22 |
| Logistic Regression | No handling                     |                        6.47 |
| Logistic Regression | Random undersampling            |                       17.92 |
| Logistic Regression | SMOTE-NC                        |                       12.84 |
| Logistic Regression | Threshold tuning                |                        6.61 |
| Random Forest       | Calibrated isotonic + threshold |                       24.56 |
| Random Forest       | Calibrated sigmoid + threshold  |                       22.24 |
| Random Forest       | Class weight                    |                       27.52 |
| Random Forest       | No handling                     |                       25.32 |
| Random Forest       | Random undersampling            |                       28.52 |
| Random Forest       | SMOTE-NC                        |                       25.48 |
| Random Forest       | Threshold tuning                |                       25.34 |
| Stacking            | Calibrated isotonic + threshold |                       16.42 |
| Stacking            | Calibrated sigmoid + threshold  |                       13.08 |
| Stacking            | Class weight                    |                       18.36 |
| Stacking            | No handling                     |                       17.62 |
| Stacking            | Random undersampling            |                       25.84 |
| Stacking            | SMOTE-NC                        |                       17.7  |
| Stacking            | Threshold tuning                |                       16.52 |
| XGBoost             | Calibrated isotonic + threshold |                       11.52 |
| XGBoost             | Calibrated sigmoid + threshold  |                       10.86 |
| XGBoost             | Class weight                    |                       15.06 |
| XGBoost             | No handling                     |                       12.34 |
| XGBoost             | Random undersampling            |                       24.06 |
| XGBoost             | SMOTE-NC                        |                       15.82 |
| XGBoost             | Threshold tuning                |                       13.84 |