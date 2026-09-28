import os
import joblib
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocess import preprocess_text

_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'sentiment_pipeline.joblib')
        try:
            _pipeline = joblib.load(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            _pipeline = None
    return _pipeline

def predict_sentiment(text):
    pipeline = get_pipeline()
    if pipeline is None:
        return {"sentiment": "Unknown (Model missing)", "confidence": 0.0}
        
    cleaned_text = preprocess_text(text)
    prediction = pipeline.predict([cleaned_text])[0]
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    
    # Get index of prediction
    class_index = list(pipeline.classes_).index(prediction)
    confidence = probabilities[class_index]
    
    prob_dict = {}
    for i, cls in enumerate(pipeline.classes_):
        prob_dict[cls.lower()] = round(float(probabilities[i]), 4)
    
    return {
        "sentiment": prediction,
        "confidence": round(float(confidence), 4),
        "probabilities": prob_dict
    }
