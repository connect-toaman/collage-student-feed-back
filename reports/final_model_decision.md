# Final Model Decision

## Dataset
- **Dataset size**: 6466 records (Real sourced feedback ONLY, NO SYNTHETIC DATA)
- **Sources**: ['EduRABSA']
- **Class distribution**: 
  - Positive: 3379 (52.3%)
  - Neutral: 1493 (23.1%)
  - Negative: 1594 (24.7%)
- **Data cleaning**: 
  - Total Initial: 6491
  - Duplicates/Empty Removed: 25
  - Unique Valid: 6466
  - Train/test split: 80% / 20% (Stratified)

## Model Comparison
- **Logistic Regression**: Accuracy=0.729, Macro F1=0.698
- **Linear SVC**: Accuracy=0.723, Macro F1=0.677

**Final Selected Model**: Logistic Regression (Saved to `model/sentiment_pipeline.joblib`)

## Confidence Calibration
- **0-20%**: 0 predictions | 0.0% accuracy
- **20-40%**: 73 predictions | 47.9% accuracy
- **40-60%**: 594 predictions | 59.9% accuracy
- **60-80%**: 415 predictions | 82.4% accuracy
- **80-100%**: 212 predictions | 99.1% accuracy

## Learning Curve Analysis
- F1 at 20%: 0.640
- F1 at 100%: 0.699
- **Conclusion**: Additional real educational feedback is likely beneficial.

## Test Sentence Results
- **Text**: 'ifra is fucked'
  - Prediction: Neutral (Conf: 35.1% | Pos: 0.35, Neu: 0.35, Neg: 0.30)
- **Text**: 'teachers are shit'
  - Prediction: Positive (Conf: 42.3% | Pos: 0.42, Neu: 0.19, Neg: 0.39)
- **Text**: 'The teachers are very helpful and the laboratory facilities are excellent.'
  - Prediction: Positive (Conf: 85.0% | Pos: 0.85, Neu: 0.11, Neg: 0.04)
- **Text**: 'The teachers are very nice and the librarians are very helpful.'
  - Prediction: Positive (Conf: 88.0% | Pos: 0.88, Neu: 0.09, Neg: 0.03)
- **Text**: 'The classrooms are poorly maintained and the internet connection is unreliable.'
  - Prediction: Negative (Conf: 35.5% | Pos: 0.32, Neu: 0.33, Neg: 0.36)
- **Text**: 'The library is open from 9 AM to 5 PM.'
  - Prediction: Neutral (Conf: 37.2% | Pos: 0.35, Neu: 0.37, Neg: 0.28)
- **Text**: 'The teachers are good but the laboratory equipment needs improvement.'
  - Prediction: Neutral (Conf: 51.1% | Pos: 0.33, Neu: 0.51, Neg: 0.16)
- **Text**: 'The course was excellent and very informative.'
  - Prediction: Positive (Conf: 64.3% | Pos: 0.64, Neu: 0.19, Neg: 0.17)
- **Text**: 'The course was terrible and poorly organized.'
  - Prediction: Negative (Conf: 78.8% | Pos: 0.09, Neu: 0.13, Neg: 0.79)
- **Text**: 'The teacher is okay.'
  - Prediction: Positive (Conf: 38.6% | Pos: 0.39, Neu: 0.32, Neg: 0.29)
- **Text**: 'The teacher is not good.'
  - Prediction: Negative (Conf: 55.2% | Pos: 0.24, Neu: 0.21, Neg: 0.55)
- **Text**: 'The teacher is not bad.'
  - Prediction: Negative (Conf: 41.8% | Pos: 0.19, Neu: 0.39, Neg: 0.42)
- **Text**: 'Everything was fine.'
  - Prediction: Neutral (Conf: 55.6% | Pos: 0.25, Neu: 0.56, Neg: 0.19)
- **Text**: 'Worst teaching experience ever.'
  - Prediction: Negative (Conf: 80.2% | Pos: 0.15, Neu: 0.05, Neg: 0.80)
- **Text**: 'Amazing faculty and excellent facilities.'
  - Prediction: Positive (Conf: 77.5% | Pos: 0.78, Neu: 0.17, Neg: 0.06)
- **Text**: 'The faculty members are supportive.'
  - Prediction: Neutral (Conf: 42.6% | Pos: 0.32, Neu: 0.43, Neg: 0.26)
- **Text**: 'The faculty members are not supportive.'
  - Prediction: Negative (Conf: 43.1% | Pos: 0.17, Neu: 0.40, Neg: 0.43)
- **Text**: 'The teaching quality needs improvement.'
  - Prediction: Negative (Conf: 40.4% | Pos: 0.31, Neu: 0.29, Neg: 0.40)
- **Text**: 'The teaching quality is outstanding.'
  - Prediction: Negative (Conf: 36.1% | Pos: 0.32, Neu: 0.32, Neg: 0.36)
- **Text**: 'The classes are average.'
  - Prediction: Positive (Conf: 48.6% | Pos: 0.49, Neu: 0.21, Neg: 0.30)
