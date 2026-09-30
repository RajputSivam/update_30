| model               | comparison                                                        | metric   |   wilcoxon_statistic |    p_value |   bh_q_value |
|:--------------------|:------------------------------------------------------------------|:---------|---------------------:|-----------:|-------------:|
| Logistic Regression | No handling vs Class weight                                       | pr_auc   |                  7   | 0.0371094  |    0.125693  |
| Logistic Regression | No handling vs Class weight                                       | f1       |                 22   | 0.625      |    0.78125   |
| Logistic Regression | No handling vs SMOTE-NC                                           | pr_auc   |                  9   | 0.0644531  |    0.180469  |
| Logistic Regression | No handling vs SMOTE-NC                                           | f1       |                 23   | 0.695312   |    0.844021  |
| Logistic Regression | No handling vs Random undersampling                               | pr_auc   |                  5   | 0.0195312  |    0.0854492 |
| Logistic Regression | No handling vs Random undersampling                               | f1       |                 26   | 0.921875   |    0.972833  |
| Logistic Regression | No handling vs Threshold tuning                                   | pr_auc   |                 22   | 0.625      |    0.78125   |
| Logistic Regression | No handling vs Threshold tuning                                   | f1       |                  4   | 0.0136719  |    0.0755551 |
| Logistic Regression | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                 27   | 1          |    1         |
| Logistic Regression | No handling vs Calibrated sigmoid + threshold                     | f1       |                  6   | 0.0273438  |    0.10074   |
| Logistic Regression | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Logistic Regression | No handling vs Calibrated isotonic + threshold                    | f1       |                 10   | 0.0839844  |    0.217737  |
| Logistic Regression | Class weight vs SMOTE-NC                                          | pr_auc   |                 15   | 0.232422   |    0.413632  |
| Logistic Regression | Class weight vs SMOTE-NC                                          | f1       |                 27   | 1          |    1         |
| Logistic Regression | Class weight vs Random undersampling                              | pr_auc   |                 13   | 0.160156   |    0.332998  |
| Logistic Regression | Class weight vs Random undersampling                              | f1       |                  6   | 0.0273438  |    0.10074   |
| Logistic Regression | Class weight vs Threshold tuning                                  | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| Logistic Regression | Class weight vs Threshold tuning                                  | f1       |                  6   | 0.0273438  |    0.10074   |
| Logistic Regression | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| Logistic Regression | Class weight vs Calibrated sigmoid + threshold                    | f1       |                  0   | 0.00195312 |    0.0256348 |
| Logistic Regression | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                 12   | 0.130859   |    0.289268  |
| Logistic Regression | Class weight vs Calibrated isotonic + threshold                   | f1       |                  2   | 0.00585938 |    0.0512695 |
| Logistic Regression | SMOTE-NC vs Random undersampling                                  | pr_auc   |                 22   | 0.625      |    0.78125   |
| Logistic Regression | SMOTE-NC vs Random undersampling                                  | f1       |                 16   | 0.275391   |    0.451813  |
| Logistic Regression | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                  7   | 0.0371094  |    0.125693  |
| Logistic Regression | SMOTE-NC vs Threshold tuning                                      | f1       |                  5   | 0.0195312  |    0.0854492 |
| Logistic Regression | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                  8   | 0.0488281  |    0.146484  |
| Logistic Regression | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                  0   | 0.00195312 |    0.0256348 |
| Logistic Regression | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                  8   | 0.0488281  |    0.146484  |
| Logistic Regression | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                  2   | 0.00585938 |    0.0512695 |
| Logistic Regression | Random undersampling vs Threshold tuning                          | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| Logistic Regression | Random undersampling vs Threshold tuning                          | f1       |                  3   | 0.00976562 |    0.0585938 |
| Logistic Regression | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                  2   | 0.00585938 |    0.0512695 |
| Logistic Regression | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                  0   | 0.00195312 |    0.0256348 |
| Logistic Regression | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                  6   | 0.0273438  |    0.10074   |
| Logistic Regression | Random undersampling vs Calibrated isotonic + threshold           | f1       |                  1   | 0.00390625 |    0.0410156 |
| Logistic Regression | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Logistic Regression | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                 25   | 0.845703   |    0.944668  |
| Logistic Regression | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                 19   | 0.431641   |    0.616629  |
| Logistic Regression | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                 26.5 | 0.960938   |    1         |
| Logistic Regression | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                 21   | 0.556641   |    0.739839  |
| Logistic Regression | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                 24   | 0.769531   |    0.897786  |
| Random Forest       | No handling vs Class weight                                       | pr_auc   |                  9   | 0.0644531  |    0.180469  |
| Random Forest       | No handling vs Class weight                                       | f1       |                  1   | 0.00390625 |    0.0410156 |
| Random Forest       | No handling vs SMOTE-NC                                           | pr_auc   |                 12   | 0.130859   |    0.289268  |
| Random Forest       | No handling vs SMOTE-NC                                           | f1       |                  1   | 0.00390625 |    0.0410156 |
| Random Forest       | No handling vs Random undersampling                               | pr_auc   |                 27   | 1          |    1         |
| Random Forest       | No handling vs Random undersampling                               | f1       |                  3   | 0.00976562 |    0.0585938 |
| Random Forest       | No handling vs Threshold tuning                                   | pr_auc   |                 21   | 0.556641   |    0.739839  |
| Random Forest       | No handling vs Threshold tuning                                   | f1       |                  0   | 0.00195312 |    0.0256348 |
| Random Forest       | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                 24   | 0.769531   |    0.897786  |
| Random Forest       | No handling vs Calibrated sigmoid + threshold                     | f1       |                  0   | 0.00195312 |    0.0256348 |
| Random Forest       | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                 16   | 0.275391   |    0.451813  |
| Random Forest       | No handling vs Calibrated isotonic + threshold                    | f1       |                  0   | 0.00195312 |    0.0256348 |
| Random Forest       | Class weight vs SMOTE-NC                                          | pr_auc   |                 23   | 0.695312   |    0.844021  |
| Random Forest       | Class weight vs SMOTE-NC                                          | f1       |                 16   | 0.275391   |    0.451813  |
| Random Forest       | Class weight vs Random undersampling                              | pr_auc   |                 15   | 0.232422   |    0.413632  |
| Random Forest       | Class weight vs Random undersampling                              | f1       |                 27   | 1          |    1         |
| Random Forest       | Class weight vs Threshold tuning                                  | pr_auc   |                 15   | 0.232422   |    0.413632  |
| Random Forest       | Class weight vs Threshold tuning                                  | f1       |                 19   | 0.431641   |    0.616629  |
| Random Forest       | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                  6   | 0.0273438  |    0.10074   |
| Random Forest       | Class weight vs Calibrated sigmoid + threshold                    | f1       |                 23   | 0.695312   |    0.844021  |
| Random Forest       | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                 21   | 0.556641   |    0.739839  |
| Random Forest       | Class weight vs Calibrated isotonic + threshold                   | f1       |                 26   | 0.921875   |    0.972833  |
| Random Forest       | SMOTE-NC vs Random undersampling                                  | pr_auc   |                 14   | 0.193359   |    0.372527  |
| Random Forest       | SMOTE-NC vs Random undersampling                                  | f1       |                 17   | 0.322266   |    0.501302  |
| Random Forest       | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                 14   | 0.193359   |    0.372527  |
| Random Forest       | SMOTE-NC vs Threshold tuning                                      | f1       |                  3   | 0.00976562 |    0.0585938 |
| Random Forest       | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                 14   | 0.193359   |    0.372527  |
| Random Forest       | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                 12   | 0.130859   |    0.289268  |
| Random Forest       | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                 17   | 0.322266   |    0.501302  |
| Random Forest       | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                  5   | 0.0195312  |    0.0854492 |
| Random Forest       | Random undersampling vs Threshold tuning                          | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Random Forest       | Random undersampling vs Threshold tuning                          | f1       |                 10   | 0.0839844  |    0.217737  |
| Random Forest       | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Random Forest       | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                 25   | 0.845703   |    0.944668  |
| Random Forest       | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 19   | 0.431641   |    0.616629  |
| Random Forest       | Random undersampling vs Calibrated isotonic + threshold           | f1       |                 21   | 0.556641   |    0.739839  |
| Random Forest       | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                 22   | 0.625      |    0.78125   |
| Random Forest       | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                 19   | 0.431641   |    0.616629  |
| Random Forest       | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                 25   | 0.845703   |    0.944668  |
| Random Forest       | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                  9   | 0.0644531  |    0.180469  |
| Random Forest       | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                 16   | 0.275391   |    0.451813  |
| Random Forest       | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                 26   | 0.921875   |    0.972833  |
| XGBoost             | No handling vs Class weight                                       | pr_auc   |                 20   | 0.492188   |    0.684499  |
| XGBoost             | No handling vs Class weight                                       | f1       |                 11   | 0.105469   |    0.25458   |
| XGBoost             | No handling vs SMOTE-NC                                           | pr_auc   |                 17   | 0.322266   |    0.501302  |
| XGBoost             | No handling vs SMOTE-NC                                           | f1       |                 23   | 0.695312   |    0.844021  |
| XGBoost             | No handling vs Random undersampling                               | pr_auc   |                  8   | 0.0488281  |    0.146484  |
| XGBoost             | No handling vs Random undersampling                               | f1       |                 19   | 0.431641   |    0.616629  |
| XGBoost             | No handling vs Threshold tuning                                   | pr_auc   |                 17   | 0.322266   |    0.501302  |
| XGBoost             | No handling vs Threshold tuning                                   | f1       |                  8   | 0.0488281  |    0.146484  |
| XGBoost             | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                 21   | 0.556641   |    0.739839  |
| XGBoost             | No handling vs Calibrated sigmoid + threshold                     | f1       |                  9   | 0.0644531  |    0.180469  |
| XGBoost             | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                 20   | 0.492188   |    0.684499  |
| XGBoost             | No handling vs Calibrated isotonic + threshold                    | f1       |                 14   | 0.193359   |    0.372527  |
| XGBoost             | Class weight vs SMOTE-NC                                          | pr_auc   |                 19   | 0.431641   |    0.616629  |
| XGBoost             | Class weight vs SMOTE-NC                                          | f1       |                 15   | 0.232422   |    0.413632  |
| XGBoost             | Class weight vs Random undersampling                              | pr_auc   |                 14   | 0.193359   |    0.372527  |
| XGBoost             | Class weight vs Random undersampling                              | f1       |                  5.5 | 0.0234375  |    0.0984375 |
| XGBoost             | Class weight vs Threshold tuning                                  | pr_auc   |                 25   | 0.845703   |    0.944668  |
| XGBoost             | Class weight vs Threshold tuning                                  | f1       |                 27   | 1          |    1         |
| XGBoost             | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                 19   | 0.431641   |    0.616629  |
| XGBoost             | Class weight vs Calibrated sigmoid + threshold                    | f1       |                 25.5 | 0.882812   |    0.972833  |
| XGBoost             | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                 26   | 0.921875   |    0.972833  |
| XGBoost             | Class weight vs Calibrated isotonic + threshold                   | f1       |                 23   | 0.695312   |    0.844021  |
| XGBoost             | SMOTE-NC vs Random undersampling                                  | pr_auc   |                 21   | 0.556641   |    0.739839  |
| XGBoost             | SMOTE-NC vs Random undersampling                                  | f1       |                  5.5 | 0.0234375  |    0.0984375 |
| XGBoost             | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                 22   | 0.625      |    0.78125   |
| XGBoost             | SMOTE-NC vs Threshold tuning                                      | f1       |                 14.5 | 0.210938   |    0.402699  |
| XGBoost             | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                 16   | 0.275391   |    0.451813  |
| XGBoost             | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                  9   | 0.0644531  |    0.180469  |
| XGBoost             | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                 22   | 0.625      |    0.78125   |
| XGBoost             | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                 17   | 0.322266   |    0.501302  |
| XGBoost             | Random undersampling vs Threshold tuning                          | pr_auc   |                 16   | 0.275391   |    0.451813  |
| XGBoost             | Random undersampling vs Threshold tuning                          | f1       |                  0   | 0.00195312 |    0.0256348 |
| XGBoost             | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| XGBoost             | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                  0   | 0.00195312 |    0.0256348 |
| XGBoost             | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 13   | 0.160156   |    0.332998  |
| XGBoost             | Random undersampling vs Calibrated isotonic + threshold           | f1       |                  5   | 0.0195312  |    0.0854492 |
| XGBoost             | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                 14   | 0.193359   |    0.372527  |
| XGBoost             | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                 26.5 | 0.960938   |    1         |
| XGBoost             | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                 24   | 0.769531   |    0.897786  |
| XGBoost             | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                 18.5 | 0.402344   |    0.612262  |
| XGBoost             | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                 22   | 0.625      |    0.78125   |
| XGBoost             | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                 16   | 0.275391   |    0.451813  |
| LightGBM            | No handling vs Class weight                                       | pr_auc   |                 22   | 0.625      |    0.78125   |
| LightGBM            | No handling vs Class weight                                       | f1       |                 13   | 0.160156   |    0.332998  |
| LightGBM            | No handling vs SMOTE-NC                                           | pr_auc   |                 13   | 0.160156   |    0.332998  |
| LightGBM            | No handling vs SMOTE-NC                                           | f1       |                 23.5 | 0.730469   |    0.876563  |
| LightGBM            | No handling vs Random undersampling                               | pr_auc   |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | No handling vs Random undersampling                               | f1       |                 16   | 0.275391   |    0.451813  |
| LightGBM            | No handling vs Threshold tuning                                   | pr_auc   |                 15   | 0.232422   |    0.413632  |
| LightGBM            | No handling vs Threshold tuning                                   | f1       |                 23.5 | 0.730469   |    0.876563  |
| LightGBM            | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                 14   | 0.193359   |    0.372527  |
| LightGBM            | No handling vs Calibrated sigmoid + threshold                     | f1       |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                 22   | 0.625      |    0.78125   |
| LightGBM            | No handling vs Calibrated isotonic + threshold                    | f1       |                 11   | 0.105469   |    0.25458   |
| LightGBM            | Class weight vs SMOTE-NC                                          | pr_auc   |                 19   | 0.431641   |    0.616629  |
| LightGBM            | Class weight vs SMOTE-NC                                          | f1       |                 11   | 0.105469   |    0.25458   |
| LightGBM            | Class weight vs Random undersampling                              | pr_auc   |                  4   | 0.0136719  |    0.0755551 |
| LightGBM            | Class weight vs Random undersampling                              | f1       |                  6   | 0.0273438  |    0.10074   |
| LightGBM            | Class weight vs Threshold tuning                                  | pr_auc   |                 21   | 0.556641   |    0.739839  |
| LightGBM            | Class weight vs Threshold tuning                                  | f1       |                 27   | 1          |    1         |
| LightGBM            | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                 13   | 0.160156   |    0.332998  |
| LightGBM            | Class weight vs Calibrated sigmoid + threshold                    | f1       |                  6   | 0.0273438  |    0.10074   |
| LightGBM            | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                 27   | 1          |    1         |
| LightGBM            | Class weight vs Calibrated isotonic + threshold                   | f1       |                 25   | 0.845703   |    0.944668  |
| LightGBM            | SMOTE-NC vs Random undersampling                                  | pr_auc   |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | SMOTE-NC vs Random undersampling                                  | f1       |                 10   | 0.0839844  |    0.217737  |
| LightGBM            | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                 18   | 0.375      |    0.574818  |
| LightGBM            | SMOTE-NC vs Threshold tuning                                      | f1       |                 22   | 0.625      |    0.78125   |
| LightGBM            | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| LightGBM            | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                  1   | 0.00390625 |    0.0410156 |
| LightGBM            | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                  8   | 0.0488281  |    0.146484  |
| LightGBM            | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | Random undersampling vs Threshold tuning                          | pr_auc   |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | Random undersampling vs Threshold tuning                          | f1       |                 12   | 0.130859   |    0.289268  |
| LightGBM            | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| LightGBM            | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                  0   | 0.00195312 |    0.0256348 |
| LightGBM            | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                  4   | 0.0136719  |    0.0755551 |
| LightGBM            | Random undersampling vs Calibrated isotonic + threshold           | f1       |                  0   | 0.00195312 |    0.0256348 |
| LightGBM            | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                 11   | 0.105469   |    0.25458   |
| LightGBM            | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                 10   | 0.0839844  |    0.217737  |
| LightGBM            | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                 27   | 1          |    1         |
| LightGBM            | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                 17   | 0.322266   |    0.501302  |
| LightGBM            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                  5   | 0.0195312  |    0.0854492 |
| LightGBM            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                 15   | 0.232422   |    0.413632  |
| Stacking            | No handling vs Class weight                                       | pr_auc   |                 17   | 0.322266   |    0.501302  |
| Stacking            | No handling vs Class weight                                       | f1       |                 13.5 | 0.175781   |    0.361903  |
| Stacking            | No handling vs SMOTE-NC                                           | pr_auc   |                 24   | 0.769531   |    0.897786  |
| Stacking            | No handling vs SMOTE-NC                                           | f1       |                 16   | 0.275391   |    0.451813  |
| Stacking            | No handling vs Random undersampling                               | pr_auc   |                 11   | 0.105469   |    0.25458   |
| Stacking            | No handling vs Random undersampling                               | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | No handling vs Threshold tuning                                   | pr_auc   |                 24   | 0.769531   |    0.897786  |
| Stacking            | No handling vs Threshold tuning                                   | f1       |                 12   | 0.130859   |    0.289268  |
| Stacking            | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                 16   | 0.275391   |    0.451813  |
| Stacking            | No handling vs Calibrated sigmoid + threshold                     | f1       |                 15   | 0.232422   |    0.413632  |
| Stacking            | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                 20   | 0.492188   |    0.684499  |
| Stacking            | No handling vs Calibrated isotonic + threshold                    | f1       |                 11.5 | 0.117188   |    0.279652  |
| Stacking            | Class weight vs SMOTE-NC                                          | pr_auc   |                 25   | 0.845703   |    0.944668  |
| Stacking            | Class weight vs SMOTE-NC                                          | f1       |                  6.5 | 0.03125    |    0.113147  |
| Stacking            | Class weight vs Random undersampling                              | pr_auc   |                 15   | 0.232422   |    0.413632  |
| Stacking            | Class weight vs Random undersampling                              | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | Class weight vs Threshold tuning                                  | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Stacking            | Class weight vs Threshold tuning                                  | f1       |                  8   | 0.0488281  |    0.146484  |
| Stacking            | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                 12   | 0.130859   |    0.289268  |
| Stacking            | Class weight vs Calibrated sigmoid + threshold                    | f1       |                 25   | 0.845703   |    0.944668  |
| Stacking            | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                 27   | 1          |    1         |
| Stacking            | Class weight vs Calibrated isotonic + threshold                   | f1       |                  9.5 | 0.0742188  |    0.205078  |
| Stacking            | SMOTE-NC vs Random undersampling                                  | pr_auc   |                  3   | 0.00976562 |    0.0585938 |
| Stacking            | SMOTE-NC vs Random undersampling                                  | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                 25   | 0.845703   |    0.944668  |
| Stacking            | SMOTE-NC vs Threshold tuning                                      | f1       |                  2   | 0.00585938 |    0.0512695 |
| Stacking            | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                 13   | 0.160156   |    0.332998  |
| Stacking            | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                  7   | 0.0371094  |    0.125693  |
| Stacking            | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                 26   | 0.921875   |    0.972833  |
| Stacking            | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                  3   | 0.00976562 |    0.0585938 |
| Stacking            | Random undersampling vs Threshold tuning                          | pr_auc   |                  8   | 0.0488281  |    0.146484  |
| Stacking            | Random undersampling vs Threshold tuning                          | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                  7   | 0.0371094  |    0.125693  |
| Stacking            | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 12   | 0.130859   |    0.289268  |
| Stacking            | Random undersampling vs Calibrated isotonic + threshold           | f1       |                  0   | 0.00195312 |    0.0256348 |
| Stacking            | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                 19   | 0.431641   |    0.616629  |
| Stacking            | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                 11   | 0.105469   |    0.25458   |
| Stacking            | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                 20   | 0.492188   |    0.684499  |
| Stacking            | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                 17.5 | 0.347656   |    0.536822  |
| Stacking            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                 10   | 0.0839844  |    0.217737  |
| Stacking            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                  8   | 0.0488281  |    0.146484  |