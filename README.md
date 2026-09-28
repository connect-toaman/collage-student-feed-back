# College Student Feedback Sentiment Analysis

A full-stack, Vercel-ready Flask web application for analyzing and classifying student feedback sentiment using a genuine NLP pipeline. 

## Dataset Sources & Preparation
This project leverages **high-quality, publicly available educational datasets** rather than synthetic or hardcoded data. 
The core dataset is built using:
- **EduRABSA (Education Review ABSA dataset)**: ~6,500 tertiary student review texts (HuggingFace: `yhua219/EduRABSA_SA`)
- **Student Feedback Dataset**: Available on Kaggle for manual integration.
- **Mendeley Teacher Performance Dataset**: Available on Mendeley Data for manual integration (CC BY-NC 3.0).

For full licensing details and links, please see `data/DATASET_SOURCES.md`.

### Dataset Preparation
The `ml/prepare_dataset.py` script automatically downloads accessible datasets (like EduRABSA), normalizes the text (removing HTML, whitespace), standardizes the sentiment labels to `Positive`, `Neutral`, `Negative`, and removes duplicate records. 
The cleaned, unified dataset is output to `data/processed/college_student_feedback_sentiment.csv`.

**Current Dataset Size**: ~6,491 genuine records (after deduping).
**Class Distribution**: 
- Positive: ~40%
- Negative: ~38%
- Neutral: ~22%

## Model Training & Evaluation
The `ml/train_model.py` script splits the data (80% Train, 20% Test, stratified) and evaluates three models:
1. **Logistic Regression** (Balanced Class Weights)
2. **Linear SVM** 
3. **Multinomial Naive Bayes**

### Evaluation Results (on unseen test set)
- **Logistic Regression (Best Model)**: 
  - Accuracy: ~72%
  - Macro F1: ~0.69
  - Weighted F1: ~0.73
- **Linear SVM**: ~72% Accuracy
- **Multinomial Naive Bayes**: ~64% Accuracy

*Note: 72% accuracy on genuine textual reviews reflects real-world language complexity, slang, and mixed sentiments.* 

### Error Analysis
The `ml/error_analysis.py` script isolates and groups misclassifications. It identifies weak points, such as the model struggling with sarcastic "Positive -> Negative" inversions or interpreting highly mixed reviews (e.g., "The teacher is great but the exam is terrible") as Neutral when human graders marked it as Negative.

## Limitations
- **Sarcasm & Mixed Sentiments**: The Bag-of-Words TF-IDF approach sometimes struggles with complex negations over long sentences or highly mixed sentiments.
- **Domain Specificity**: The model performs exceptionally well on "Educational" feedback (teachers, labs, exams) but may degrade if used for general e-commerce reviews. 

## Running Locally
```bash
pip install -r requirements.txt
python ml/prepare_dataset.py
python ml/train_model.py
python api/index.py
```

## Aspect-Based Sentiment Analysis (ABSA)
This project now supports ABSA to analyze multi-aspect feedback (e.g. "The teachers are good but the lab is bad"). For more details on the NLP architecture, see [docs/ABSA.md](docs/ABSA.md).
