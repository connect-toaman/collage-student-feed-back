import pandas as pd
import numpy as np
import os

def audit_dataset():
    data_path = 'data/processed/college_student_feedback_sentiment.csv'
    if not os.path.exists(data_path):
        print("Processed dataset not found.")
        return

    df = pd.read_csv(data_path)
    
    total_rows = len(df)
    
    # Check for empty/invalid
    df['text_str'] = df['text'].astype(str)
    empty_rows = len(df[df['text_str'].str.strip() == ''])
    
    # Check for duplicates
    dup_rows = len(df[df.duplicated(subset=['text_str'], keep=False)])
    
    # Conflicting labels (same text, different sentiment)
    text_sent_grouped = df.groupby('text_str')['sentiment'].nunique()
    conflicting_rows = len(text_sent_grouped[text_sent_grouped > 1])
    
    # Value counts
    vc = df['sentiment'].value_counts()
    pos_count = vc.get('Positive', 0)
    neu_count = vc.get('Neutral', 0)
    neg_count = vc.get('Negative', 0)
    
    # Valid rows
    valid_rows = total_rows - empty_rows
    
    # Profanity check
    profanity_words = ['fuck', 'fucked', 'shit', 'bitch', 'ass', 'damn', 'crap', 'bullshit']
    profanity_count = df['text_str'].str.contains('|'.join(profanity_words), case=False, na=False).sum()
    
    report = [
        "# Data Quality Report",
        f"- Total rows: {total_rows}",
        f"- Valid rows: {valid_rows}",
        f"- Empty/invalid text rows: {empty_rows}",
        f"- Duplicate rows: {dup_rows}",
        f"- Conflicting-label rows: {conflicting_rows}",
        f"- Rows containing profanity: {profanity_count}",
        "",
        "## Class Distribution",
        f"- Positive count: {pos_count} ({(pos_count/total_rows)*100:.1f}%)",
        f"- Neutral count: {neu_count} ({(neu_count/total_rows)*100:.1f}%)",
        f"- Negative count: {neg_count} ({(neg_count/total_rows)*100:.1f}%)"
    ]
    
    with open('data_quality_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    print("data_quality_report.md generated.")
    
    # Specific words inspection
    test_words = ['fucked', 'fuck', 'shit', 'terrible', 'horrible', 'worst', 'bad', 'poor', 'useless', 'excellent', 'amazing', 'helpful', 'good']
    print("\n--- Word Inspection ---")
    for w in test_words:
        subset = df[df['text_str'].str.contains(rf'\b{w}\b', case=False, na=False)]
        print(f"Word '{w}': found {len(subset)} times.")
        if len(subset) > 0:
            samp = subset.sample(min(2, len(subset)))
            for _, r in samp.iterrows():
                print(f"  [{r['sentiment']}] {r['source']} | {r['text'][:80]}...")

    print("\n--- 20 Random Examples per Class ---")
    for sent in ['Positive', 'Neutral', 'Negative']:
        print(f"\n{sent}:")
        samp = df[df['sentiment'] == sent]
        samp = samp.sample(min(20, len(samp)))
        for _, r in samp.iterrows():
            print(f"  - {r['text'][:80]}...")

if __name__ == "__main__":
    audit_dataset()
