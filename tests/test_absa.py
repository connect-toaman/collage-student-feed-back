import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.aspect_sentiment import analyze_aspects

test_cases = [
    {
        "text": "The teachers are good but the laboratory equipment needs improvement.",
        "expected_overall": "Mixed",
        "expected_aspects": ["Teachers / Faculty", "Laboratory / Lab Equipment"]
    },
    {
        "text": "The teachers are excellent and very supportive.",
        "expected_overall": "Positive",
        "expected_aspects": ["Teachers / Faculty"]
    },
    {
        "text": "The laboratory equipment is outdated and poorly maintained.",
        "expected_overall": "Negative",
        "expected_aspects": ["Laboratory / Lab Equipment"]
    },
    {
        "text": "The library is excellent but the internet connection is terrible.",
        "expected_overall": "Mixed",
        "expected_aspects": ["Library", "Internet / Wi-Fi"]
    },
    {
        "text": "The library is open from 9 AM to 5 PM.",
        "expected_overall": "Neutral",
        "expected_aspects": ["Library"]
    },
    {
        "text": "The teachers are terrible and the classrooms are poorly maintained.",
        "expected_overall": "Negative",
        "expected_aspects": ["Teachers / Faculty", "Classrooms"]
    },
    {
        "text": "The teachers are helpful, the library is excellent, and the laboratory is well equipped.",
        "expected_overall": "Positive",
        "expected_aspects": ["Teachers / Faculty", "Library", "Laboratory / Lab Equipment"]
    },
    {
        "text": "The teacher is good but the facilities are terrible.",
        "expected_overall": "Mixed",
        "expected_aspects": ["Teachers / Faculty", "Infrastructure / Facilities"]
    },
    {
        "text": "I like the teachers but I hate the cafeteria food.",
        "expected_overall": "Mixed",
        "expected_aspects": ["Teachers / Faculty", "Canteen / Cafeteria"]
    },
    {
        "text": "The campus is okay.",
        "expected_overall": "Neutral",
        "expected_aspects": ["Infrastructure / Facilities"] # Added campus to Infrastructure
    },
    {
        "text": "Overall, I had a good experience.",
        "expected_overall": "Positive",
        "expected_aspects": ["General Feedback"]
    }
]

def run_tests():
    for i, tc in enumerate(test_cases):
        res = analyze_aspects(tc['text'])
        overall = res['overall']
        aspects = [a['aspect'] for a in res['aspects']]
        
        print(f"Test {i+1}: {tc['text']}")
        print(f"  Expected Overall: {tc['expected_overall']} | Actual: {overall}")
        print(f"  Expected Aspects: {tc['expected_aspects']} | Actual: {aspects}")
        print(f"  Aspect Details: {res['aspects']}")
        print("-" * 40)

if __name__ == "__main__":
    run_tests()
