import pandas as pd
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML
    text = re.sub(r'\s+', ' ', text)      # Remove excessive whitespace
    return text.strip()

def prepare_dataset():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    combined_data = []

    # 1. HuggingFace EduRABSA_SA
    try:
        import datasets
        print("Downloading EduRABSA from HuggingFace...")
        ds = datasets.load_dataset('yhua219/EduRABSA_SA')
        for split in ds.keys():
            df_hf = ds[split].to_pandas()
            for _, row in df_hf.iterrows():
                # HF label mapping varies, assume standard positive/neutral/negative if available
                # Fallback to string representation if needed
                label_val = str(row.get('output', row.get('label', 'Neutral'))).lower()
                if 'pos' in label_val or label_val == '1' or label_val == '2':
                    sentiment = 'Positive'
                elif 'neg' in label_val or label_val == '0':
                    sentiment = 'Negative'
                else:
                    sentiment = 'Neutral'
                    
                combined_data.append({
                    'text': clean_text(row.get('text', '')),
                    'sentiment': sentiment,
                    'source': 'EduRABSA',
                    'category': 'General' # Fallback if specific category isn't clear
                })
    except Exception as e:
        print(f"Could not load HuggingFace dataset: {e}")

    # 2. Kaggle Student Feedback
    kaggle_path = 'data/raw/kaggle_student_feedback.csv'
    if os.path.exists(kaggle_path):
        try:
            df_k = pd.read_csv(kaggle_path)
            # Detect text column
            text_col = next((col for col in df_k.columns if 'text' in col.lower() or 'feedback' in col.lower()), None)
            # Detect label column
            label_col = next((col for col in df_k.columns if 'sentiment' in col.lower() or 'label' in col.lower() or 'score' in col.lower()), None)
            
            if text_col and label_col:
                for _, row in df_k.iterrows():
                    val = str(row[label_col]).strip()
                    if val == '1' or 'pos' in val.lower():
                        sentiment = 'Positive'
                    elif val == '-1' or 'neg' in val.lower():
                        sentiment = 'Negative'
                    else:
                        sentiment = 'Neutral'
                        
                    combined_data.append({
                        'text': clean_text(str(row[text_col])),
                        'sentiment': sentiment,
                        'source': 'StudentFeedback_Kaggle',
                        'category': 'General'
                    })
        except Exception as e:
            print(f"Error reading Kaggle dataset: {e}")

    # 3. Mendeley Teacher Performance
    mendeley_path = 'data/raw/mendeley_teacher_performance.csv'
    if os.path.exists(mendeley_path):
        try:
            df_m = pd.read_csv(mendeley_path)
            text_col = next((col for col in df_m.columns if 'text' in col.lower() or 'review' in col.lower()), None)
            label_col = next((col for col in df_m.columns if 'sentiment' in col.lower() or 'class' in col.lower()), None)
            
            if text_col and label_col:
                for _, row in df_m.iterrows():
                    val = str(row[label_col]).strip().lower()
                    if 'pos' in val or val == '1': sentiment = 'Positive'
                    elif 'neg' in val or val == '-1': sentiment = 'Negative'
                    else: sentiment = 'Neutral'
                        
                    combined_data.append({
                        'text': clean_text(str(row[text_col])),
                        'sentiment': sentiment,
                        'source': 'MendeleyTeacherFeedback',
                        'category': 'Teacher'
                    })
        except Exception as e:
            print(f"Error reading Mendeley dataset: {e}")

    if len(combined_data) == 0:
        raise ValueError("No external datasets found! Synthetic fallback removed per requirements.")

    df = pd.DataFrame(combined_data)
    
    # Cleaning
    initial_len = len(df)
    df = df[df['text'].str.strip() != '']
    df = df.drop_duplicates(subset=['text'], keep='first')
    print(f"Removed {initial_len - len(df)} duplicates/empty rows.")
    
    # Generate Report
    report = []
    report.append("# Dataset Report\n")
    report.append(f"**Total Records:** {len(df)}\n")
    
    for source in df['source'].unique():
        report.append(f"- **{source}**: {len(df[df['source'] == source])} records")
    
    report.append(f"\n### Sentiment Distribution")
    dist = df['sentiment'].value_counts()
    for k, v in dist.items():
        report.append(f"- {k}: {v} ({(v/len(df))*100:.1f}%)")
        
    df['text_len'] = df['text'].apply(len)
    report.append(f"\n### Text Statistics")
    report.append(f"- Average Length: {df['text_len'].mean():.1f} chars")
    report.append(f"- Median Length: {df['text_len'].median():.1f} chars")
    report.append(f"- Min Length: {df['text_len'].min()} chars")
    report.append(f"- Max Length: {df['text_len'].max()} chars")
    
    df = df.drop(columns=['text_len'])
    df.to_csv('data/processed/real_educational_feedback.csv', index=False)
    
    with open('data/DATASET_REPORT.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(report))
        
    print("Dataset preparation complete! Check data/DATASET_REPORT.md")

if __name__ == "__main__":
    prepare_dataset()
