| model               | comparison                                                        | metric   |   wilcoxon_statistic |     p_value |   bh_q_value |
|:--------------------|:------------------------------------------------------------------|:---------|---------------------:|------------:|-------------:|
| Logistic Regression | No handling vs Class weight                                       | pr_auc   |                175   | 1.97672e-06 |  5.76543e-06 |
| Logistic Regression | No handling vs Class weight                                       | f1       |                399.5 | 0.0215914   |  0.0357023   |
| Logistic Regression | No handling vs SMOTE-NC                                           | pr_auc   |                166   | 1.13262e-06 |  3.49781e-06 |
| Logistic Regression | No handling vs SMOTE-NC                                           | f1       |                366   | 0.00808562  |  0.0143897   |
| Logistic Regression | No handling vs Random undersampling                               | pr_auc   |                 22   | 9.52127e-13 |  1.33298e-11 |
| Logistic Regression | No handling vs Random undersampling                               | f1       |                625.5 | 0.90778     |  0.920936    |
| Logistic Regression | No handling vs Threshold tuning                                   | pr_auc   |                584.5 | 0.608914    |  0.702593    |
| Logistic Regression | No handling vs Threshold tuning                                   | f1       |                 42   | 2.02203e-11 |  1.8462e-10  |
| Logistic Regression | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                601   | 0.730448    |  0.774718    |
| Logistic Regression | No handling vs Calibrated sigmoid + threshold                     | f1       |                 75   | 8.85528e-10 |  5.63518e-09 |
| Logistic Regression | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                138   | 1.73945e-07 |  6.29803e-07 |
| Logistic Regression | No handling vs Calibrated isotonic + threshold                    | f1       |                 75.5 | 5.7898e-08  |  2.48134e-07 |
| Logistic Regression | Class weight vs SMOTE-NC                                          | pr_auc   |                345   | 0.00417845  |  0.00769714  |
| Logistic Regression | Class weight vs SMOTE-NC                                          | f1       |                589.5 | 0.643108    |  0.729215    |
| Logistic Regression | Class weight vs Random undersampling                              | pr_auc   |                133   | 1.2141e-07  |  4.84658e-07 |
| Logistic Regression | Class weight vs Random undersampling                              | f1       |                145.5 | 2.04008e-06 |  5.86872e-06 |
| Logistic Regression | Class weight vs Threshold tuning                                  | pr_auc   |                172   | 1.64553e-06 |  4.93658e-06 |
| Logistic Regression | Class weight vs Threshold tuning                                  | f1       |                 54   | 9.07523e-11 |  7.32999e-10 |
| Logistic Regression | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                212   | 1.60312e-05 |  4.00781e-05 |
| Logistic Regression | Class weight vs Calibrated sigmoid + threshold                    | f1       |                 36   | 8.8427e-12  |  8.8427e-11  |
| Logistic Regression | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                591   | 0.659698    |  0.740724    |
| Logistic Regression | Class weight vs Calibrated isotonic + threshold                   | f1       |                 49   | 1.33919e-08 |  6.69252e-08 |
| Logistic Regression | SMOTE-NC vs Random undersampling                                  | pr_auc   |                330   | 0.00252569  |  0.004866    |
| Logistic Regression | SMOTE-NC vs Random undersampling                                  | f1       |                178   | 2.36937e-06 |  6.63423e-06 |
| Logistic Regression | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                191   | 5.0702e-06  |  1.34778e-05 |
| Logistic Regression | SMOTE-NC vs Threshold tuning                                      | f1       |                 51   | 6.33786e-11 |  5.54563e-10 |
| Logistic Regression | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                200   | 8.39809e-06 |  2.15073e-05 |
| Logistic Regression | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                 27   | 2.23999e-12 |  2.93998e-11 |
| Logistic Regression | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                469   | 0.105255    |  0.148345    |
| Logistic Regression | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                 52   | 7.15143e-11 |  6.00721e-10 |
| Logistic Regression | Random undersampling vs Threshold tuning                          | pr_auc   |                 29   | 3.08908e-12 |  3.60393e-11 |
| Logistic Regression | Random undersampling vs Threshold tuning                          | f1       |                 22   | 9.52127e-13 |  1.33298e-11 |
| Logistic Regression | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                 40   | 1.5449e-11  |  1.47467e-10 |
| Logistic Regression | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                 13   | 1.56319e-13 |  2.98428e-12 |
| Logistic Regression | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                158   | 6.78355e-07 |  2.19161e-06 |
| Logistic Regression | Random undersampling vs Calibrated isotonic + threshold           | f1       |                 17   | 3.67706e-13 |  5.93986e-12 |
| Logistic Regression | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                624   | 0.901018    |  0.918513    |
| Logistic Regression | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                622   | 0.885886    |  0.911941    |
| Logistic Regression | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                156   | 5.95147e-07 |  1.95283e-06 |
| Logistic Regression | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                556.5 | 0.434264    |  0.536444    |
| Logistic Regression | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                134   | 1.30549e-07 |  4.9846e-07  |
| Logistic Regression | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                600   | 0.717352    |  0.768591    |
| Random Forest       | No handling vs Class weight                                       | pr_auc   |                474   | 0.116195    |  0.161596    |
| Random Forest       | No handling vs Class weight                                       | f1       |                  2   | 5.32907e-15 |  1.59872e-13 |
| Random Forest       | No handling vs SMOTE-NC                                           | pr_auc   |                592   | 0.666652    |  0.740724    |
| Random Forest       | No handling vs SMOTE-NC                                           | f1       |                  6   | 2.4869e-14  |  6.52811e-13 |
| Random Forest       | No handling vs Random undersampling                               | pr_auc   |                325   | 0.00212243  |  0.00412696  |
| Random Forest       | No handling vs Random undersampling                               | f1       |                 15   | 2.43361e-13 |  4.25882e-12 |
| Random Forest       | No handling vs Threshold tuning                                   | pr_auc   |                578   | 0.572086    |  0.678746    |
| Random Forest       | No handling vs Threshold tuning                                   | f1       |                  0   | 1.77636e-15 |  1.24345e-13 |
| Random Forest       | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                225   | 3.12816e-05 |  7.46493e-05 |
| Random Forest       | No handling vs Calibrated sigmoid + threshold                     | f1       |                  0   | 1.77636e-15 |  1.24345e-13 |
| Random Forest       | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                579   | 0.578629    |  0.682652    |
| Random Forest       | No handling vs Calibrated isotonic + threshold                    | f1       |                  0   | 1.77636e-15 |  1.24345e-13 |
| Random Forest       | Class weight vs SMOTE-NC                                          | pr_auc   |                541   | 0.357187    |  0.451863    |
| Random Forest       | Class weight vs SMOTE-NC                                          | f1       |                347.5 | 0.00511896  |  0.00926709  |
| Random Forest       | Class weight vs Random undersampling                              | pr_auc   |                399   | 0.0206659   |  0.0344431   |
| Random Forest       | Class weight vs Random undersampling                              | f1       |                265   | 0.000202797 |  0.000443618 |
| Random Forest       | Class weight vs Threshold tuning                                  | pr_auc   |                415   | 0.0312617   |  0.0482717   |
| Random Forest       | Class weight vs Threshold tuning                                  | f1       |                336.5 | 0.00366507  |  0.006872    |
| Random Forest       | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                222   | 2.68859e-05 |  6.48969e-05 |
| Random Forest       | Class weight vs Calibrated sigmoid + threshold                    | f1       |                390.5 | 0.0171089   |  0.0287429   |
| Random Forest       | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                451   | 0.0723775   |  0.10555     |
| Random Forest       | Class weight vs Calibrated isotonic + threshold                   | f1       |                409.5 | 0.0277397   |  0.0431507   |
| Random Forest       | SMOTE-NC vs Random undersampling                                  | pr_auc   |                406   | 0.0248477   |  0.0395304   |
| Random Forest       | SMOTE-NC vs Random undersampling                                  | f1       |                570   | 0.521025    |  0.628823    |
| Random Forest       | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                550   | 0.404258    |  0.505322    |
| Random Forest       | SMOTE-NC vs Threshold tuning                                      | f1       |                203   | 9.89861e-06 |  2.50447e-05 |
| Random Forest       | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                403   | 0.0229749   |  0.0376932   |
| Random Forest       | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                248.5 | 0.000173255 |  0.000387058 |
| Random Forest       | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                578   | 0.572086    |  0.678746    |
| Random Forest       | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                199   | 7.94697e-06 |  2.06032e-05 |
| Random Forest       | Random undersampling vs Threshold tuning                          | pr_auc   |                311   | 0.00128224  |  0.00256448  |
| Random Forest       | Random undersampling vs Threshold tuning                          | f1       |                 89.5 | 1.22319e-07 |  4.84658e-07 |
| Random Forest       | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                137   | 1.61982e-07 |  5.96775e-07 |
| Random Forest       | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                134   | 1.30549e-07 |  4.9846e-07  |
| Random Forest       | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                280   | 0.000382585 |  0.000803429 |
| Random Forest       | Random undersampling vs Calibrated isotonic + threshold           | f1       |                131   | 1.049e-07   |  4.40582e-07 |
| Random Forest       | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                268   | 0.000230875 |  0.000499832 |
| Random Forest       | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                613   | 0.813036    |  0.849441    |
| Random Forest       | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                635   | 0.984734    |  0.984734    |
| Random Forest       | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                557   | 0.437096    |  0.536784    |
| Random Forest       | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                292   | 0.000620771 |  0.00127806  |
| Random Forest       | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                592   | 0.666652    |  0.740724    |
| XGBoost             | No handling vs Class weight                                       | pr_auc   |                464   | 0.0951333   |  0.135905    |
| XGBoost             | No handling vs Class weight                                       | f1       |                257.5 | 0.00024422  |  0.000523329 |
| XGBoost             | No handling vs SMOTE-NC                                           | pr_auc   |                405   | 0.0242095   |  0.0391077   |
| XGBoost             | No handling vs SMOTE-NC                                           | f1       |                214.5 | 4.43936e-05 |  0.000104749 |
| XGBoost             | No handling vs Random undersampling                               | pr_auc   |                113   | 2.63781e-08 |  1.23098e-07 |
| XGBoost             | No handling vs Random undersampling                               | f1       |                628.5 | 0.930767    |  0.939717    |
| XGBoost             | No handling vs Threshold tuning                                   | pr_auc   |                580   | 0.585207    |  0.686555    |
| XGBoost             | No handling vs Threshold tuning                                   | f1       |                226.5 | 7.26313e-05 |  0.000169473 |
| XGBoost             | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                519   | 0.257124    |  0.333309    |
| XGBoost             | No handling vs Calibrated sigmoid + threshold                     | f1       |                152   | 2.77653e-06 |  7.672e-06   |
| XGBoost             | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                601   | 0.730448    |  0.774718    |
| XGBoost             | No handling vs Calibrated isotonic + threshold                    | f1       |                164   | 9.97976e-07 |  3.12798e-06 |
| XGBoost             | Class weight vs SMOTE-NC                                          | pr_auc   |                573   | 0.5399      |  0.64788     |
| XGBoost             | Class weight vs SMOTE-NC                                          | f1       |                617   | 0.848244    |  0.881838    |
| XGBoost             | Class weight vs Random undersampling                              | pr_auc   |                182   | 3.00675e-06 |  8.20022e-06 |
| XGBoost             | Class weight vs Random undersampling                              | f1       |                105   | 1.37037e-08 |  6.69252e-08 |
| XGBoost             | Class weight vs Threshold tuning                                  | pr_auc   |                551   | 0.409704    |  0.509099    |
| XGBoost             | Class weight vs Threshold tuning                                  | f1       |                623   | 0.893447    |  0.915238    |
| XGBoost             | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                345   | 0.00417845  |  0.00769714  |
| XGBoost             | Class weight vs Calibrated sigmoid + threshold                    | f1       |                463   | 0.0932041   |  0.134061    |
| XGBoost             | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                372   | 0.0096745   |  0.0170727   |
| XGBoost             | Class weight vs Calibrated isotonic + threshold                   | f1       |                407.5 | 0.0264015   |  0.0413755   |
| XGBoost             | SMOTE-NC vs Random undersampling                                  | pr_auc   |                222   | 2.68859e-05 |  6.48969e-05 |
| XGBoost             | SMOTE-NC vs Random undersampling                                  | f1       |                123   | 5.76496e-08 |  2.48134e-07 |
| XGBoost             | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                540   | 0.352175    |  0.448223    |
| XGBoost             | SMOTE-NC vs Threshold tuning                                      | f1       |                596   | 0.694746    |  0.75972     |
| XGBoost             | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                378   | 0.0115297   |  0.0196848   |
| XGBoost             | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                548   | 0.387602    |  0.487404    |
| XGBoost             | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                404   | 0.0235853   |  0.0383947   |
| XGBoost             | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                507.5 | 0.209503    |  0.274973    |
| XGBoost             | Random undersampling vs Threshold tuning                          | pr_auc   |                172   | 1.64553e-06 |  4.93658e-06 |
| XGBoost             | Random undersampling vs Threshold tuning                          | f1       |                132   | 1.12873e-07 |  4.64771e-07 |
| XGBoost             | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                 77   | 1.08053e-09 |  6.67386e-09 |
| XGBoost             | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                 60   | 1.80947e-10 |  1.40736e-09 |
| XGBoost             | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 74   | 8.00824e-10 |  5.42494e-09 |
| XGBoost             | Random undersampling vs Calibrated isotonic + threshold           | f1       |                 30.5 | 4.64055e-09 |  2.49876e-08 |
| XGBoost             | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                456   | 0.0805483   |  0.116656    |
| XGBoost             | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                535.5 | 0.324804    |  0.415907    |
| XGBoost             | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                469   | 0.105255    |  0.148345    |
| XGBoost             | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                429   | 0.0441441   |  0.0671757   |
| XGBoost             | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                597   | 0.701837    |  0.75972     |
| XGBoost             | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                570   | 0.521025    |  0.628823    |
| LightGBM            | No handling vs Class weight                                       | pr_auc   |                598   | 0.708952    |  0.763487    |
| LightGBM            | No handling vs Class weight                                       | f1       |                118   | 5.30509e-07 |  1.76836e-06 |
| LightGBM            | No handling vs SMOTE-NC                                           | pr_auc   |                481   | 0.132961    |  0.182495    |
| LightGBM            | No handling vs SMOTE-NC                                           | f1       |                241.5 | 0.000131987 |  0.000298036 |
| LightGBM            | No handling vs Random undersampling                               | pr_auc   |                101   | 9.77286e-09 |  5.13075e-08 |
| LightGBM            | No handling vs Random undersampling                               | f1       |                433   | 0.0483095   |  0.0724643   |
| LightGBM            | No handling vs Threshold tuning                                   | pr_auc   |                629   | 0.938992    |  0.943485    |
| LightGBM            | No handling vs Threshold tuning                                   | f1       |                161.5 | 4.32831e-06 |  1.16531e-05 |
| LightGBM            | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                334   | 0.00289631  |  0.00552932  |
| LightGBM            | No handling vs Calibrated sigmoid + threshold                     | f1       |                119   | 4.2364e-08  |  1.93401e-07 |
| LightGBM            | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                407   | 0.0255001   |  0.0402633   |
| LightGBM            | No handling vs Calibrated isotonic + threshold                    | f1       |                110   | 2.07007e-08 |  9.87986e-08 |
| LightGBM            | Class weight vs SMOTE-NC                                          | pr_auc   |                520   | 0.261209    |  0.336527    |
| LightGBM            | Class weight vs SMOTE-NC                                          | f1       |                434.5 | 0.0500407   |  0.0745288   |
| LightGBM            | Class weight vs Random undersampling                              | pr_auc   |                 89   | 3.38221e-09 |  1.86912e-08 |
| LightGBM            | Class weight vs Random undersampling                              | f1       |                 75   | 8.85528e-10 |  5.63518e-09 |
| LightGBM            | Class weight vs Threshold tuning                                  | pr_auc   |                585   | 0.618594    |  0.709861    |
| LightGBM            | Class weight vs Threshold tuning                                  | f1       |                582.5 | 0.595465    |  0.690871    |
| LightGBM            | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                249   | 9.9154e-05  |  0.00022633  |
| LightGBM            | Class weight vs Calibrated sigmoid + threshold                    | f1       |                486.5 | 0.144938    |  0.197643    |
| LightGBM            | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                416   | 0.0320535   |  0.0491331   |
| LightGBM            | Class weight vs Calibrated isotonic + threshold                   | f1       |                472.5 | 0.111207    |  0.15569     |
| LightGBM            | SMOTE-NC vs Random undersampling                                  | pr_auc   |                163   | 9.36411e-07 |  2.97949e-06 |
| LightGBM            | SMOTE-NC vs Random undersampling                                  | f1       |                173   | 1.74968e-06 |  5.17511e-06 |
| LightGBM            | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                495   | 0.171914    |  0.231423    |
| LightGBM            | SMOTE-NC vs Threshold tuning                                      | f1       |                481.5 | 0.13209     |  0.182493    |
| LightGBM            | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                337   | 0.00320542  |  0.00606431  |
| LightGBM            | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                378   | 0.0115297   |  0.0196848   |
| LightGBM            | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                378   | 0.0115297   |  0.0196848   |
| LightGBM            | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                347   | 0.00445942  |  0.00814329  |
| LightGBM            | Random undersampling vs Threshold tuning                          | pr_auc   |                 61   | 2.02322e-10 |  1.51741e-09 |
| LightGBM            | Random undersampling vs Threshold tuning                          | f1       |                 31   | 4.21885e-12 |  4.42979e-11 |
| LightGBM            | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                 30   | 3.61489e-12 |  3.9954e-11  |
| LightGBM            | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                 29   | 3.08908e-12 |  3.60393e-11 |
| LightGBM            | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 68   | 4.31486e-10 |  3.12455e-09 |
| LightGBM            | Random undersampling vs Calibrated isotonic + threshold           | f1       |                 11   | 1.46824e-09 |  8.80944e-09 |
| LightGBM            | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                303   | 0.000950408 |  0.00191909  |
| LightGBM            | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                497   | 0.178101    |  0.238224    |
| LightGBM            | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                391   | 0.0166377   |  0.0281768   |
| LightGBM            | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                448.5 | 0.0680811   |  0.0999792   |
| LightGBM            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                505   | 0.204476    |  0.271772    |
| LightGBM            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                611.5 | 0.801826    |  0.841917    |
| Stacking            | No handling vs Class weight                                       | pr_auc   |                495   | 0.171914    |  0.231423    |
| Stacking            | No handling vs Class weight                                       | f1       |                245   | 8.23756e-05 |  0.000190098 |
| Stacking            | No handling vs SMOTE-NC                                           | pr_auc   |                609   | 0.788753    |  0.832352    |
| Stacking            | No handling vs SMOTE-NC                                           | f1       |                582.5 | 0.595465    |  0.690871    |
| Stacking            | No handling vs Random undersampling                               | pr_auc   |                151   | 4.26982e-07 |  1.46994e-06 |
| Stacking            | No handling vs Random undersampling                               | f1       |                 13   | 1.56319e-13 |  2.98428e-12 |
| Stacking            | No handling vs Threshold tuning                                   | pr_auc   |                506   | 0.20796     |  0.274664    |
| Stacking            | No handling vs Threshold tuning                                   | f1       |                101   | 2.23062e-07 |  7.93948e-07 |
| Stacking            | No handling vs Calibrated sigmoid + threshold                     | pr_auc   |                281   | 0.000398648 |  0.000828871 |
| Stacking            | No handling vs Calibrated sigmoid + threshold                     | f1       |                104   | 2.60411e-07 |  9.11439e-07 |
| Stacking            | No handling vs Calibrated isotonic + threshold                    | pr_auc   |                597   | 0.701837    |  0.75972     |
| Stacking            | No handling vs Calibrated isotonic + threshold                    | f1       |                 72   | 4.78929e-08 |  2.1399e-07  |
| Stacking            | Class weight vs SMOTE-NC                                          | pr_auc   |                595   | 0.687682    |  0.75609     |
| Stacking            | Class weight vs SMOTE-NC                                          | f1       |                315.5 | 0.0018812   |  0.0037269   |
| Stacking            | Class weight vs Random undersampling                              | pr_auc   |                213   | 1.68966e-05 |  4.17446e-05 |
| Stacking            | Class weight vs Random undersampling                              | f1       |                 12   | 1.24345e-13 |  2.90138e-12 |
| Stacking            | Class weight vs Threshold tuning                                  | pr_auc   |                430   | 0.0450275   |  0.0680272   |
| Stacking            | Class weight vs Threshold tuning                                  | f1       |                323   | 0.00197804  |  0.00388213  |
| Stacking            | Class weight vs Calibrated sigmoid + threshold                    | pr_auc   |                264   | 0.000194158 |  0.000429191 |
| Stacking            | Class weight vs Calibrated sigmoid + threshold                    | f1       |                259.5 | 0.000263303 |  0.000558523 |
| Stacking            | Class weight vs Calibrated isotonic + threshold                   | pr_auc   |                446   | 0.0648862   |  0.0959585   |
| Stacking            | Class weight vs Calibrated isotonic + threshold                   | f1       |                293   | 0.000645726 |  0.00131653  |
| Stacking            | SMOTE-NC vs Random undersampling                                  | pr_auc   |                178   | 2.36937e-06 |  6.63423e-06 |
| Stacking            | SMOTE-NC vs Random undersampling                                  | f1       |                 21.5 | 2.73955e-09 |  1.55488e-08 |
| Stacking            | SMOTE-NC vs Threshold tuning                                      | pr_auc   |                561   | 0.466473    |  0.569531    |
| Stacking            | SMOTE-NC vs Threshold tuning                                      | f1       |                135   | 1.40329e-07 |  5.26234e-07 |
| Stacking            | SMOTE-NC vs Calibrated sigmoid + threshold                        | pr_auc   |                406   | 0.0248477   |  0.0395304   |
| Stacking            | SMOTE-NC vs Calibrated sigmoid + threshold                        | f1       |                169.5 | 6.25048e-06 |  1.64075e-05 |
| Stacking            | SMOTE-NC vs Calibrated isotonic + threshold                       | pr_auc   |                587   | 0.632173    |  0.721502    |
| Stacking            | SMOTE-NC vs Calibrated isotonic + threshold                       | f1       |                102   | 1.06421e-08 |  5.45083e-08 |
| Stacking            | Random undersampling vs Threshold tuning                          | pr_auc   |                154   | 5.21565e-07 |  1.76659e-06 |
| Stacking            | Random undersampling vs Threshold tuning                          | f1       |                  1   | 3.55271e-15 |  1.49214e-13 |
| Stacking            | Random undersampling vs Calibrated sigmoid + threshold            | pr_auc   |                 70   | 5.31845e-10 |  3.72291e-09 |
| Stacking            | Random undersampling vs Calibrated sigmoid + threshold            | f1       |                  2   | 5.32907e-15 |  1.59872e-13 |
| Stacking            | Random undersampling vs Calibrated isotonic + threshold           | pr_auc   |                 85   | 2.33441e-09 |  1.36174e-08 |
| Stacking            | Random undersampling vs Calibrated isotonic + threshold           | f1       |                  1   | 3.55271e-15 |  1.49214e-13 |
| Stacking            | Threshold tuning vs Calibrated sigmoid + threshold                | pr_auc   |                358   | 0.00632549  |  0.0113534   |
| Stacking            | Threshold tuning vs Calibrated sigmoid + threshold                | f1       |                620   | 0.865847    |  0.895704    |
| Stacking            | Threshold tuning vs Calibrated isotonic + threshold               | pr_auc   |                589   | 0.645876    |  0.729215    |
| Stacking            | Threshold tuning vs Calibrated isotonic + threshold               | f1       |                519.5 | 0.254664    |  0.332171    |
| Stacking            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | pr_auc   |                373   | 0.00996428  |  0.0174375   |
| Stacking            | Calibrated sigmoid + threshold vs Calibrated isotonic + threshold | f1       |                593.5 | 0.671023    |  0.741657    |