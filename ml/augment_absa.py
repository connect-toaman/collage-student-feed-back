import pandas as pd
import random
import os

def augment_dataset():
    df = pd.read_csv('data/processed/absa_dataset.csv')
    
    aug_rows = []
    
    pos_words = ['excellent', 'good', 'great', 'wonderful', 'amazing', 'fantastic', 'superb', 'outstanding', 'brilliant', 'perfect']
    neg_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'poor', 'unacceptable', 'disgusting', 'dreadful', 'useless']
    
    aspects = ['Teachers / Faculty', 'Infrastructure / Facilities', 'Library', 'Internet / Wi-Fi', 'Laboratory / Lab Equipment', 'Course Content', 'Administration']
    
    # Generate 500 clean positive
    for _ in range(500):
        w = random.choice(pos_words)
        a = random.choice(aspects)
        text = f"The {a.split(' / ')[0].lower()} is {w}."
        aug_rows.append({
            'original_id': 'aug_pos',
            'text': text,
            'aspect': a,
            'aspect_category': 'General',
            'aspect_context': text,
            'sentiment': 'Positive',
            'source': 'Augmented_Clean'
        })
        
    # Generate 500 clean negative
    for _ in range(500):
        w = random.choice(neg_words)
        a = random.choice(aspects)
        text = f"The {a.split(' / ')[0].lower()} is {w}."
        aug_rows.append({
            'original_id': 'aug_neg',
            'text': text,
            'aspect': a,
            'aspect_category': 'General',
            'aspect_context': text,
            'sentiment': 'Negative',
            'source': 'Augmented_Clean'
        })
        
    aug_df = pd.DataFrame(aug_rows)
    df = pd.concat([df, aug_df], ignore_index=True)
    
    df.to_csv('data/processed/absa_dataset.csv', index=False)
    print("Augmented with 1000 clean examples.")

if __name__ == "__main__":
    augment_dataset()
