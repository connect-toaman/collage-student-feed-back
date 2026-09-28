import json
import os
from ml.aspect_sentiment import analyze_aspects
from app.sentiment import predict_sentiment

examples = [
    "The teacher is bad but the facilities are excellent.",
    "The teacher is good but the facilities are terrible.",
    "The teachers are excellent and very supportive.",
    "The teachers are terrible and unhelpful.",
    "The library is open from 9 AM to 5 PM.",
    "The laboratory has twenty computers.",
    "The laboratory equipment needs improvement.",
    "The laboratory equipment is excellent.",
    "The teacher is not good.",
    "The teacher is not bad.",
    "The teacher is good, but the laboratory is poor.",
    "The teachers are helpful, although the administration is slow.",
    "The library is excellent but the Wi-Fi is terrible.",
    "The campus is clean and well maintained.",
    "The campus is dirty and poorly maintained.",
    "The course was okay.",
    "The course was neither good nor bad."
]

def run_evaluation():
    print("=" * 60)
    print("SPECIAL TEST EVALUATION")
    print("=" * 60)
    for text in examples:
        print(f"\nOriginal text: {text}")
        absa_result = analyze_aspects(text)
        doc_result = predict_sentiment(text)
        
        print("Detected aspects:")
        for a in absa_result['aspects']:
            print(f"- Aspect: {a['aspect']}")
            print(f"  Context: {a['context']}")
            print(f"  Sentiment: {a['sentiment']}")
            print(f"  Confidence: {a['confidence']*100:.1f}%")
            
        print(f"Document sentiment: {doc_result['sentiment']} ({doc_result['confidence']*100:.1f}%)")
        print(f"ABSA overall sentiment: {absa_result['overall']}")
        print("-" * 60)

if __name__ == "__main__":
    run_evaluation()
