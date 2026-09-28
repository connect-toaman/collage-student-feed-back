# ABSA Final Model Report

## Dataset Split
- Train: 25683
- Validation: 3252
- Test: 3137

## Model Comparison (on Validation Set)

### Logistic Regression
- Accuracy: 0.8078
- Macro F1: 0.7431

### Linear SVM
- Accuracy: 0.8075
- Macro F1: 0.6926

### Multinomial NB
- Accuracy: 0.7568
- Macro F1: 0.5518

**Best Model**: Logistic Regression

## Test Set Performance
- Accuracy: 0.8138
- Macro F1: 0.7461
- Weighted F1: 0.8202

### Per-Class Metrics
- **Negative**: Precision: 0.8021 | Recall: 0.7939 | F1: 0.7980 | Support: 985
- **Neutral**: Precision: 0.4833 | Recall: 0.6308 | F1: 0.5473 | Support: 390
- **Positive**: Precision: 0.9226 | Recall: 0.8655 | F1: 0.8931 | Support: 1762