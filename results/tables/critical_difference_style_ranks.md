| model               | strategy                        |   average_rank_across_folds |
|:--------------------|:--------------------------------|----------------------------:|
| LightGBM            | Calibrated isotonic + threshold |                        31   |
| LightGBM            | Calibrated sigmoid + threshold  |                        25.5 |
| LightGBM            | Class weight                    |                        15   |
| LightGBM            | No handling                     |                        15   |
| LightGBM            | Random undersampling            |                        34   |
| LightGBM            | SMOTE-NC                        |                        13   |
| LightGBM            | Threshold tuning                |                        15   |
| Logistic Regression | Calibrated isotonic + threshold |                        12   |
| Logistic Regression | Calibrated sigmoid + threshold  |                         6.5 |
| Logistic Regression | Class weight                    |                        16.5 |
| Logistic Regression | No handling                     |                         3   |
| Logistic Regression | Random undersampling            |                        18.5 |
| Logistic Regression | SMOTE-NC                        |                        20.5 |
| Logistic Regression | Threshold tuning                |                         3   |
| Random Forest       | Calibrated isotonic + threshold |                        31.5 |
| Random Forest       | Calibrated sigmoid + threshold  |                        21   |
| Random Forest       | Class weight                    |                        24.5 |
| Random Forest       | No handling                     |                        18.5 |
| Random Forest       | Random undersampling            |                        32   |
| Random Forest       | SMOTE-NC                        |                        22   |
| Random Forest       | Threshold tuning                |                        18.5 |
| Stacking            | Calibrated isotonic + threshold |                        19   |
| Stacking            | Calibrated sigmoid + threshold  |                        13.5 |
| Stacking            | Class weight                    |                         7.5 |
| Stacking            | No handling                     |                         9.5 |
| Stacking            | Random undersampling            |                        25   |
| Stacking            | SMOTE-NC                        |                         2   |
| Stacking            | Threshold tuning                |                         9.5 |
| XGBoost             | Calibrated isotonic + threshold |                        26.5 |
| XGBoost             | Calibrated sigmoid + threshold  |                        25   |
| XGBoost             | Class weight                    |                        18   |
| XGBoost             | No handling                     |                        16   |
| XGBoost             | Random undersampling            |                        31   |
| XGBoost             | SMOTE-NC                        |                        15   |
| XGBoost             | Threshold tuning                |                        16   |