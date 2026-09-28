import pandas as pd
import os

def analyze_errors():
    test_res_path = 'reports/test_predictions.csv'
    if not os.path.exists(test_res_path):
        print("Run ml/train_model.py first to generate test predictions.")
        return

    df = pd.read_csv(test_res_path)
    errors = df[df['actual'] != df['predicted']]
    
    print(f"Total Errors Found: {len(errors)} out of {len(df)} test samples ({(len(errors)/len(df))*100:.1f}%)")
    
    if len(errors) == 0:
        print("No errors found in test set!")
        return

    print("\n--- Error Grouping ---")
    error_types = errors.groupby(['actual', 'predicted']).size().reset_index(name='count')
    for _, row in error_types.iterrows():
        print(f"{row['actual']} -> {row['predicted']}: {row['count']} instances")

    print("\n--- Sample Misclassifications ---")
    for actual in ['Positive', 'Negative', 'Neutral']:
        for predicted in ['Positive', 'Negative', 'Neutral']:
            if actual == predicted: continue
            
            subset = errors[(errors['actual'] == actual) & (errors['predicted'] == predicted)]
            if len(subset) > 0:
                print(f"\n{actual} -> {predicted}:")
                for _, row in subset.head(3).iterrows():
                    print(f"Conf: {row['confidence']:.2f} | Text: {row['text']}")

if __name__ == "__main__":
    analyze_errors()
