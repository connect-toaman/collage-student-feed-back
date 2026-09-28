import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import joblib

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocess import preprocess_text

def evaluate():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'dataset.csv')
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'sentiment_pipeline.joblib')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please train it first.")
        return
        
    print("Loading model...")
    pipeline = joblib.load(model_path)
    
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("Preprocessing text...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['label']
    
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Making predictions on test set...")
    y_pred = pipeline.predict(X_test)
    
    print("\n--- Evaluation Results ---")
    print(f"Dataset size: {len(df)} total, {len(X_train)} train, {len(X_test)} test")
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
