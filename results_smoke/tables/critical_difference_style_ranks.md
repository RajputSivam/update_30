| model               | strategy                        |   average_rank_across_folds |
|:--------------------|:--------------------------------|----------------------------:|
| LightGBM            | Calibrated isotonic + threshold |                        20.6 |
| LightGBM            | Calibrated sigmoid + threshold  |                        13.7 |
| LightGBM            | Class weight                    |                        20.3 |
| LightGBM            | No handling                     |                        19   |
| LightGBM            | Random undersampling            |                        29.8 |
| LightGBM            | SMOTE-NC                        |                        24.4 |
| LightGBM            | Threshold tuning                |                        20.3 |
| Logistic Regression | Calibrated isotonic + threshold |                         7.7 |
| Logistic Regression | Calibrated sigmoid + threshold  |                         5.8 |
| Logistic Regression | Class weight                    |                         8.7 |
| Logistic Regression | No handling                     |                         6.6 |
| Logistic Regression | Random undersampling            |                        13.8 |
| Logistic Regression | SMOTE-NC                        |                        15.3 |
| Logistic Regression | Threshold tuning                |                         5.6 |
| Random Forest       | Calibrated isotonic + threshold |                        27   |
| Random Forest       | Calibrated sigmoid + threshold  |                        24.7 |
| Random Forest       | Class weight                    |                        30.2 |
| Random Forest       | No handling                     |                        23.8 |
| Random Forest       | Random undersampling            |                        25.1 |
| Random Forest       | SMOTE-NC                        |                        28.4 |
| Random Forest       | Threshold tuning                |                        25   |
| Stacking            | Calibrated isotonic + threshold |                        17.8 |
| Stacking            | Calibrated sigmoid + threshold  |                        14   |
| Stacking            | Class weight                    |                        18.2 |
| Stacking            | No handling                     |                        15.5 |
| Stacking            | Random undersampling            |                        23.8 |
| Stacking            | SMOTE-NC                        |                        17.7 |
| Stacking            | Threshold tuning                |                        15.8 |
| XGBoost             | Calibrated isotonic + threshold |                        16.1 |
| XGBoost             | Calibrated sigmoid + threshold  |                        11.6 |
| XGBoost             | Class weight                    |                        15.3 |
| XGBoost             | No handling                     |                        11.8 |
| XGBoost             | Random undersampling            |                        22.2 |
| XGBoost             | SMOTE-NC                        |                        18.7 |
| XGBoost             | Threshold tuning                |                        15.7 |