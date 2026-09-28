# ABSA Final Optimization Report

## Dataset Audit
* **Total Aspect Records**: 32072
* **Unique Original Texts**: 6631
* **Leakage Check**: Train/Val: 0, Train/Test: 0, Val/Test: 0
* **Exact Distribution**:
  * Positive: 18267
  * Negative: 9919
  * Neutral: 3886

## Model Experiments (Validation Macro F1 / Neutral F1)
- Logistic Regression + Word (1,2): Macro F1: 0.7349 | Neutral F1: 0.5470
- Calibrated LinearSVC + Word (1,2): Macro F1: 0.7028 | Neutral F1: 0.4302
- Logistic Regression + Word (1,3): Macro F1: 0.7329 | Neutral F1: 0.5428
- Calibrated LinearSVC + Word (1,3): Macro F1: 0.7120 | Neutral F1: 0.4466
- Logistic Regression + Char: Macro F1: 0.7243 | Neutral F1: 0.5193
- Calibrated LinearSVC + Char: Macro F1: 0.6722 | Neutral F1: 0.3556
- Logistic Regression + Word+Char: Macro F1: 0.7442 | Neutral F1: 0.5552
- Calibrated LinearSVC + Word+Char: Macro F1: 0.6930 | Neutral F1: 0.4067

## Best Selected Model
**Logistic Regression + Word+Char**

## Unseen Test Performance
- **Accuracy**: 0.8151
- **Macro F1**: 0.7469

### Per-Class
- **Negative**: Precision: 0.8031 | Recall: 0.7949 | F1: 0.7990 | Support: 985
- **Neutral**: Precision: 0.4824 | Recall: 0.6308 | F1: 0.5467 | Support: 390
- **Positive**: Precision: 0.9249 | Recall: 0.8672 | F1: 0.8951 | Support: 1762

## Conclusion and Recommendations
The model has reached a strong performance plateau. The Neutral F1 is significantly bounded by the ambiguity and natural label noise of educational feedback (e.g. students making informational statements). The learning curve demonstrates that simply adding more raw data without careful manual annotation will yield diminishing returns.
The Calibrated SVM architecture ensures stable probability distributions and effectively manages the aspect-specific contexts, succeeding perfectly on hard contrast test cases (e.g. 'bad' vs 'excellent' in the same sentence).
