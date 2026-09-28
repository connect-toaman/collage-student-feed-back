import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_text
from ml.prepare_dataset import prepare_dataset
import joblib

def run_validation():
    print("Preparing dataset (no synthetic)...")
    prepare_dataset()
    
    data_path = 'data/processed/real_educational_feedback.csv'
    df = pd.read_csv(data_path)
    
    total_initial = len(df)
    empty_rows = len(df[df['text'].isna() | (df['text'].str.strip() == '')])
    
    # Remove empty/duplicate just in case, though prepare_dataset does it.
    df['cleaned_text'] = df['text'].astype(str).apply(preprocess_text)
    df = df[df['cleaned_text'].str.strip() != '']
    df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
    
    unique_rows = len(df)
    duplicate_rows = total_initial - unique_rows
    
    # Class counts
    counts = df['sentiment'].value_counts()
    pos = counts.get('Positive', 0)
    neu = counts.get('Neutral', 0)
    neg = counts.get('Negative', 0)
    tot = pos + neu + neg
    
    # Stratified Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], 
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    # Ensure no leakage
    train_set = set(X_train)
    test_set = set(X_test)
    leakage = train_set.intersection(test_set)
    if len(leakage) > 0:
        print(f"WARNING: {len(leakage)} examples leaked from train to test!")
        
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    models = {
        'Logistic Regression': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        'LinearSVC': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', LinearSVC(max_iter=2000, class_weight='balanced', dual=False))
        ])
    }
    
    results = {}
    best_model_name = None
    best_pipeline = None
    best_f1 = 0
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        mac_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        results[name] = {
            'accuracy': acc,
            'macro_f1': mac_f1,
            'preds': y_pred
        }
        
        if mac_f1 > best_f1:
            best_f1 = mac_f1
            best_model_name = name
            best_pipeline = pipeline

    print(f"Selected Model: {best_model_name} with Macro F1: {best_f1}")
    
    # Save Classification Report for best model
    best_preds = results[best_model_name]['preds']
    report_dict = classification_report(y_test, best_preds, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv('reports/classification_report.csv')
    
    # Confusion Matrix
    labels = best_pipeline.classes_
    cm = confusion_matrix(y_test, best_preds, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
    cm_df.to_csv('reports/confusion_matrix.csv')
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('reports/confusion_matrix.png')
    plt.close()
    
    # Error Analysis (Top 50)
    errors = []
    
    # For SVC we must use decision_function if predict_proba is not available
    has_proba = hasattr(best_pipeline.named_steps['clf'], 'predict_proba')
    
    if has_proba:
        probs = best_pipeline.predict_proba(X_test)
    else:
        # Fake probs using decision_function for SVC
        dists = best_pipeline.decision_function(X_test)
        probs = np.exp(dists) / np.sum(np.exp(dists), axis=1, keepdims=True)
        
    for i, (idx, txt) in enumerate(X_test.items()):
        actual = y_test.loc[idx]
        pred = best_preds[i]
        if actual != pred:
            prob_dict = {labels[j]: probs[i][j] for j in range(len(labels))}
            
            # Simple grouping heuristic
            if len(txt.split()) < 5: group = "short text"
            elif 'not_' in txt or 'never_' in txt or 'no_' in txt: group = "negation"
            elif ('fuck' in txt or 'shit' in txt or 'damn' in txt): group = "profanity"
            else: group = "mixed sentiment or ambiguous"
            
            # Revert the clean text slightly to show original
            orig_txt = df.loc[idx, 'text']
            errors.append({
                'Text': orig_txt,
                'Actual Sentiment': actual,
                'Predicted Sentiment': pred,
                'Positive Probability': prob_dict.get('Positive', 0),
                'Neutral Probability': prob_dict.get('Neutral', 0),
                'Negative Probability': prob_dict.get('Negative', 0),
                'Group': group
            })
            if len(errors) >= 50:
                break
                
    err_df = pd.DataFrame(errors)
    err_df.to_csv('reports/error_analysis.csv', index=False)
    
    # Confidence Calibration
    buckets = {'0-20': [0,0], '20-40': [0,0], '40-60': [0,0], '60-80': [0,0], '80-100': [0,0]}
    for i in range(len(best_preds)):
        actual = y_test.iloc[i]
        pred = best_preds[i]
        conf = max(probs[i])
        
        is_correct = 1 if actual == pred else 0
        
        if conf <= 0.2: b = '0-20'
        elif conf <= 0.4: b = '20-40'
        elif conf <= 0.6: b = '40-60'
        elif conf <= 0.8: b = '60-80'
        else: b = '80-100'
        
        buckets[b][0] += 1
        buckets[b][1] += is_correct
        
    calib_lines = []
    for k, v in buckets.items():
        acc = (v[1]/v[0]*100) if v[0] > 0 else 0
        calib_lines.append(f"- **{k}%**: {v[0]} predictions | {acc:.1f}% accuracy")
        
    # Learning Curves
    train_sizes, train_scores, test_scores = learning_curve(
        best_pipeline, df['cleaned_text'], df['sentiment'], cv=3, scoring='f1_macro',
        train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0], n_jobs=-1
    )
    test_mean = test_scores.mean(axis=1)
    lc_plateau = test_mean[-1] - test_mean[-2] < 0.01 # If last jump is < 1%
    lc_status = "Additional real educational feedback is likely beneficial." if not lc_plateau else "Performance has plateaued; more data may have diminishing returns."

    # Final Model Decision Report
    report = f"""# Final Model Decision

## Dataset
- **Dataset size**: {tot} records (Real sourced feedback ONLY, NO SYNTHETIC DATA)
- **Sources**: {list(df['source'].unique())}
- **Class distribution**: 
  - Positive: {pos} ({pos/tot*100:.1f}%)
  - Neutral: {neu} ({neu/tot*100:.1f}%)
  - Negative: {neg} ({neg/tot*100:.1f}%)
- **Data cleaning**: 
  - Total Initial: {total_initial}
  - Duplicates/Empty Removed: {duplicate_rows}
  - Unique Valid: {unique_rows}
  - Train/test split: 80% / 20% (Stratified)

## Model Comparison
- **Logistic Regression**: Accuracy={results['Logistic Regression']['accuracy']:.3f}, Macro F1={results['Logistic Regression']['macro_f1']:.3f}
- **Linear SVC**: Accuracy={results['LinearSVC']['accuracy']:.3f}, Macro F1={results['LinearSVC']['macro_f1']:.3f}

**Final Selected Model**: {best_model_name} (Saved to `model/sentiment_pipeline.joblib`)

## Confidence Calibration
{chr(10).join(calib_lines)}

## Learning Curve Analysis
- F1 at 20%: {test_mean[0]:.3f}
- F1 at 100%: {test_mean[-1]:.3f}
- **Conclusion**: {lc_status}

## Test Sentence Results
"""
    # Test sentences
    sentences = [
        "ifra is fucked", "teachers are shit", 
        "The teachers are very helpful and the laboratory facilities are excellent.",
        "The teachers are very nice and the librarians are very helpful.",
        "The classrooms are poorly maintained and the internet connection is unreliable.",
        "The library is open from 9 AM to 5 PM.",
        "The teachers are good but the laboratory equipment needs improvement.",
        "The course was excellent and very informative.",
        "The course was terrible and poorly organized.",
        "The teacher is okay.", "The teacher is not good.", "The teacher is not bad.",
        "Everything was fine.", "Worst teaching experience ever.",
        "Amazing faculty and excellent facilities.", "The faculty members are supportive.",
        "The faculty members are not supportive.", "The teaching quality needs improvement.",
        "The teaching quality is outstanding.", "The classes are average."
    ]
    
    for s in sentences:
        cl = preprocess_text(s)
        pred = best_pipeline.predict([cl])[0]
        if has_proba:
            pr = best_pipeline.predict_proba([cl])[0]
        else:
            d = best_pipeline.decision_function([cl])
            pr = np.exp(d) / np.sum(np.exp(d), axis=1, keepdims=True)[0]
        
        pr_dict = {labels[j]: pr[j] for j in range(len(labels))}
        report += f"- **Text**: '{s}'\n  - Prediction: {pred} (Conf: {max(pr)*100:.1f}% | Pos: {pr_dict.get('Positive',0):.2f}, Neu: {pr_dict.get('Neutral',0):.2f}, Neg: {pr_dict.get('Negative',0):.2f})\n"

    with open('reports/final_model_decision.md', 'w') as f:
        f.write(report)
        
    # Save the pipeline
    joblib.dump(best_pipeline, 'model/sentiment_pipeline.joblib')
    print("Done! Check reports directory.")

if __name__ == "__main__":
    run_validation()
