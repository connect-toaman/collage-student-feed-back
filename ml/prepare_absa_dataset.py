import pandas as pd
import json
import os

def prepare_absa_dataset():
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('data', exist_ok=True)

    print("Downloading EduRABSA_ASTE from HuggingFace...")
    import datasets
    ds = datasets.load_dataset('yhua219/EduRABSA_ASTE')

    absa_rows = []
    
    # We want to process both train and test to get all available data
    for split in ds.keys():
        for row in ds[split]:
            text = row['text']
            # output is a list of dicts, but it's a string if we convert to pandas, so we iterate through huggingface dataset object
            outputs = row['output']
            if isinstance(outputs, str):
                try:
                    outputs = eval(outputs) # It might be a string representation of list
                except:
                    continue
            
            if not isinstance(outputs, list):
                continue
                
            for aspect_data in outputs:
                aspect = aspect_data.get('aspect')
                opinion = aspect_data.get('opinion')
                sentiment = aspect_data.get('sentiment')
                
                # We need to map to our aspect categories roughly, or just use the raw aspect
                if not aspect or not sentiment:
                    continue
                    
                # Format exactly as requested
                absa_rows.append({
                    'original_id': row.get('id', ''),
                    'text': text,
                    'aspect': aspect,
                    'aspect_category': 'General', # We can improve this later
                    'aspect_context': opinion if opinion else text,
                    'sentiment': sentiment.capitalize(),
                    'source': 'EduRABSA_ASTE'
                })

    df = pd.DataFrame(absa_rows)
    print(f"Created {len(df)} ABSA records from ASTE.")
    
    # 2. Augment with document-level data to learn general sentiment words
    # The user required: "Build a data strategy using: 3. Existing sentiment-labeled educational datasets"
    doc_path = 'data/processed/real_educational_feedback.csv'
    if os.path.exists(doc_path):
        df_doc = pd.read_csv(doc_path)
        print(f"Augmenting with {len(df_doc)} document-level records...")
        
        doc_rows = []
        for _, row in df_doc.iterrows():
            doc_rows.append({
                'original_id': '',
                'text': row['text'],
                'aspect': 'General Feedback',
                'aspect_category': 'General',
                'aspect_context': row['text'],
                'sentiment': str(row['sentiment']).capitalize(),
                'source': row.get('source', 'Document_Level')
            })
            
        df = pd.concat([df, pd.DataFrame(doc_rows)], ignore_index=True)
    
    # Clean up empty
    df = df[df['sentiment'].isin(['Positive', 'Negative', 'Neutral'])]
    
    output_path = 'data/processed/absa_dataset.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path} (Total: {len(df)})")

    # Generate DATASET_SOURCES.md
    report = """# Dataset Sources

## EduRABSA_ASTE
* **Name**: EduRABSA_ASTE
* **URL/Source**: https://huggingface.co/datasets/yhua219/EduRABSA_ASTE
* **Number of Records**: Extracted {0} ABSA aspect annotations.
* **License**: Unspecified on HF, assume research/educational use.
* **Fields Used**: `text`, `output` (parsed for aspect, opinion, sentiment).
* **Sentiment Labels**: Positive, Negative, Neutral.
* **Aspect Labels**: Raw aspects provided by the dataset.
* **Redistribution Allowed**: Check HF repository for explicit license details.
* **Transformations Performed**: Parsed ASTE triplet array into flattened CSV rows (one per aspect).
""".format(len(df))

    with open('data/DATASET_SOURCES.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    prepare_absa_dataset()
