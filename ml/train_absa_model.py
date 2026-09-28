import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib

from ml.preprocess import preprocess_text

def train_absa_model():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    df = pd.read_csv('data/processed/absa_dataset.csv')
    df = df.dropna(subset=['text', 'aspect', 'aspect_context', 'sentiment'])
    
    df['aspect_context_clean'] = df['aspect_context'].astype(str).apply(preprocess_text)
    
    # Construct Aspect-Aware feature
    df['absa_feature'] = "[ASPECT] " + df['aspect'].astype(str) + " [CONTEXT] " + df['aspect_context_clean'].astype(str)
    
    # DATA SPLIT: Split by original text to avoid leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_val_idx = next(gss.split(df, groups=df['text']))
    
    train_df = df.iloc[train_idx]
    test_val_df = df.iloc[test_val_idx]
    
    gss_val = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, test_idx = next(gss_val.split(test_val_df, groups=test_val_df['text']))
    
    val_df = test_val_df.iloc[val_idx]
    test_df = test_val_df.iloc[test_idx]
    
    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Combined TF-IDF: Words + Characters
    word_tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=2, max_df=0.9)
    char_tfidf = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), min_df=2, max_df=0.9)
    
    combined_features = FeatureUnion([
        ('word', word_tfidf),
        ('char', char_tfidf)
    ])
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Linear SVM': CalibratedClassifierCV(LinearSVC(max_iter=2000, class_weight='balanced', dual=False), cv=3),
        'Multinomial NB': MultinomialNB()
    }
    
    best_f1 = 0
    best_model_name = None
    best_pipeline = None
    best_preds = None
    best_probs = None
    
    report_lines = ["# ABSA Final Model Report\n"]
    report_lines.append(f"## Dataset Split\n- Train: {len(train_df)}\n- Validation: {len(val_df)}\n- Test: {len(test_df)}\n")
    report_lines.append("## Model Comparison (on Validation Set)\n")
    
    X_train = train_df['absa_feature']
    y_train = train_df['sentiment']
    X_val = val_df['absa_feature']
    y_val = val_df['sentiment']
    
    for name, clf in models.items():
        pipeline = Pipeline([
            ('features', combined_features),
            ('classifier', clf)
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_val)
        
        mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_val, preds, average='macro', zero_division=0)
        acc = accuracy_score(y_val, preds)
        
        report_lines.append(f"### {name}")
        report_lines.append(f"- Accuracy: {acc:.4f}")
        report_lines.append(f"- Macro F1: {mac_f1:.4f}\n")
        
        if mac_f1 > best_f1:
            best_f1 = mac_f1
            best_model_name = name
            best_pipeline = pipeline

    report_lines.append(f"**Best Model**: {best_model_name}")
    
    # Test Evaluation
    X_test = test_df['absa_feature']
    y_test = test_df['sentiment']
    test_preds = best_pipeline.predict(X_test)
    test_probs = best_pipeline.predict_proba(X_test)
    
    mac_p, mac_r, mac_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='macro', zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_test, test_preds, average='weighted', zero_division=0)
    acc = accuracy_score(y_test, test_preds)
    
    report_lines.append("\n## Test Set Performance")
    report_lines.append(f"- Accuracy: {acc:.4f}")
    report_lines.append(f"- Macro F1: {mac_f1:.4f}")
    report_lines.append(f"- Weighted F1: {wt_f1:.4f}\n")
    
    classes = best_pipeline.classes_
    p, r, f1, sup = precision_recall_fscore_support(y_test, test_preds, labels=classes, zero_division=0)
    
    report_lines.append("### Per-Class Metrics")
    for i, cls in enumerate(classes):
        report_lines.append(f"- **{cls}**: Precision: {p[i]:.4f} | Recall: {r[i]:.4f} | F1: {f1[i]:.4f} | Support: {sup[i]}")
        
    # Error Analysis
    errors = []
    for i in range(len(test_df)):
        if y_test.iloc[i] != test_preds[i]:
            errors.append({
                'text': test_df.iloc[i]['text'],
                'aspect': test_df.iloc[i]['aspect'],
                'aspect_context': test_df.iloc[i]['aspect_context'],
                'actual': y_test.iloc[i],
                'predicted': test_preds[i],
                'confidence': max(test_probs[i])
            })
            
    err_df = pd.DataFrame(errors)
    err_df.to_csv('reports/absa_error_analysis.csv', index=False)
    
    with open('reports/absa_final_report.md', 'w') as f:
        f.write("\n".join(report_lines))
        
    joblib.dump(best_pipeline, 'model/absa_pipeline.joblib')
    print("ABSA Model training complete! Saved to model/absa_pipeline.joblib")

if __name__ == "__main__":
    train_absa_model()
