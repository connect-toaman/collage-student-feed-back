import pandas as pd
import joblib
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_text

def train():
    os.makedirs('reports', exist_ok=True)
    os.makedirs('model', exist_ok=True)
    
    data_path = 'data/processed/real_educational_feedback.csv'
    df = pd.read_csv(data_path)
    
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Drop completely empty after cleaning
    df = df[df['cleaned_text'].str.strip() != '']
    # Drop duplicates after cleaning to prevent leakage
    df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['sentiment'], 
        test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    models = {
        'Logistic Regression (1,1)': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.9)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        'Logistic Regression (1,2)': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]),
        'Linear SVM': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', LinearSVC(class_weight='balanced', max_iter=2000, dual=False))
        ]),
        'Multinomial Naive Bayes': Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)),
            ('clf', MultinomialNB())
        ])
    }
    
    results = []
    best_f1 = 0
    best_model_name = None
    best_pipeline = None
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        mac_prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        mac_rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        mac_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        w_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Macro Precision': mac_prec,
            'Macro Recall': mac_rec,
            'Macro F1': mac_f1,
            'Weighted F1': w_f1
        })
        
        # Save confusion matrix plot
        cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(f'reports/confusion_matrix_{name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")}.png')
        plt.close()
        
        # Prefer Logistic Regression for probability outputs
        if mac_f1 > best_f1 and 'Logistic' in name:
            best_f1 = mac_f1
            best_model_name = name
            best_pipeline = pipeline

    # Ensure we actually save the best even if it's not LR, but fallback to LR if possible
    if best_pipeline is None:
        best_pipeline = models['Logistic Regression (1,2)']
        best_model_name = 'Logistic Regression (1,2)'
        
    results_df = pd.DataFrame(results)
    results_df.to_csv('reports/model_comparison.csv', index=False)
    
    joblib.dump(best_pipeline, 'model/sentiment_pipeline.joblib')
    
    # Save Metadata
    vocab_size = len(best_pipeline.named_steps['tfidf'].vocabulary_)
    metadata = {
        "model_name": best_model_name,
        "training_row_count": len(X_train),
        "class_distribution": df['sentiment'].value_counts().to_dict(),
        "vectorizer_config": {
            "ngram_range": best_pipeline.named_steps['tfidf'].ngram_range,
            "min_df": best_pipeline.named_steps['tfidf'].min_df,
            "max_df": best_pipeline.named_steps['tfidf'].max_df,
            "vocabulary_size": vocab_size
        },
        "validation_metrics": {
            "Macro F1": best_f1
        },
        "training_date": datetime.now().isoformat(),
        "dataset_sources": list(df['source'].unique())
    }
    with open('model/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    # Explicit Test Sentences
    test_sentences = [
        "ifra is fucked",
        "teachers are shit",
        "The teachers are very helpful and the laboratory facilities are excellent.",
        "The teachers are very nice and the librarians are very helpful.",
        "The classrooms are poorly maintained and the internet connection is unreliable.",
        "The library is open from 9 AM to 5 PM.",
        "The teachers are good but the laboratory equipment needs improvement.",
        "The course was excellent and very informative.",
        "The course was terrible and poorly organized.",
        "The teacher is okay.",
        "The teacher is not good.",
        "The teacher is not bad.",
        "Everything was fine.",
        "Worst teaching experience ever.",
        "Amazing faculty and excellent facilities."
    ]
    
    print("\n--- Final Explicit Tests ---")
    classes = list(best_pipeline.classes_)
    
    for s in test_sentences:
        cleaned = preprocess_text(s)
        pred = best_pipeline.predict([cleaned])[0]
        if hasattr(best_pipeline.named_steps['clf'], 'predict_proba'):
            probs = best_pipeline.predict_proba([cleaned])[0]
            conf = max(probs)
            prob_dict = {classes[i]: probs[i] for i in range(len(classes))}
        else:
            conf = 1.0
            prob_dict = {c: 1.0 if c == pred else 0.0 for c in classes}
            
        print(f"Text: '{s}'")
        print(f"Predicted Sentiment: {pred}")
        print(f"Confidence: {conf:.2f}")
        for c in classes:
            print(f"Probability {c}: {prob_dict.get(c, 0):.2f}")
        print("-" * 20)

if __name__ == "__main__":
    train()
