import joblib
import os
from ml.aspect_extractor import AspectExtractor
from app.sentiment import predict_sentiment as document_predict

_extractor = None
_absa_model = None

def get_extractor():
    global _extractor
    if _extractor is None:
        _extractor = AspectExtractor()
    return _extractor

def get_absa_model():
    global _absa_model
    if _absa_model is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'absa_pipeline.joblib')
        if os.path.exists(model_path):
            _absa_model = joblib.load(model_path)
    return _absa_model

from ml.preprocess import preprocess_text

def predict_absa_sentiment(aspect, context):
    model = get_absa_model()
    if not model:
        # Fallback to document model if ABSA model is missing
        return document_predict(context)
        
    context_clean = preprocess_text(context)
    feature_string = f"[ASPECT] {aspect} [CONTEXT] {context_clean}"
    pred = model.predict([feature_string])[0]
    
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba([feature_string])[0]
        confidence = float(max(probs))
    else:
        confidence = 0.5
        
    return {
        "sentiment": pred,
        "confidence": confidence
    }

def determine_overall_absa_sentiment(aspects):
    if not aspects:
        return "Neutral"
        
    pos_count = sum(1 for a in aspects if a['sentiment'] == 'Positive')
    neg_count = sum(1 for a in aspects if a['sentiment'] == 'Negative')
    neu_count = sum(1 for a in aspects if a['sentiment'] == 'Neutral')
    
    if pos_count > 0 and neg_count > 0:
        return "Mixed"
    elif pos_count > 0 and neg_count == 0:
        return "Positive"
    elif neg_count > 0 and pos_count == 0:
        return "Negative"
    else:
        return "Neutral"

def analyze_aspects(text):
    extractor = get_extractor()
    extracted_aspects = extractor.extract_aspects(text)
    
    # If no aspects are found, use General Feedback
    if not extracted_aspects:
        # Fallback to document level sentiment for the general feedback
        doc_prediction = document_predict(text)
        return {
            "overall": doc_prediction['sentiment'],
            "aspects": [
                {
                    "aspect": "General Feedback",
                    "sentiment": doc_prediction['sentiment'],
                    "confidence": doc_prediction['confidence'],
                    "context": text
                }
            ]
        }
        
    final_aspects = []
    for item in extracted_aspects:
        prediction = predict_absa_sentiment(item['aspect'], item['context'])
        final_aspects.append({
            "aspect": item['aspect'],
            "sentiment": prediction['sentiment'],
            "confidence": prediction['confidence'],
            "context": item['context']
        })
        
    overall = determine_overall_absa_sentiment(final_aspects)
    
    return {
        "overall": overall,
        "aspects": final_aspects
    }
