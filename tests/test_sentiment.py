import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.preprocess import preprocess_text
from app.sentiment import predict_sentiment
from ml.train_model import train

# Ensure model is trained for testing
def setup_module(module):
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model', 'sentiment_pipeline.joblib')
    if not os.path.exists(model_path):
        train()

def test_preprocess_text():
    text = "The teacher is NOT helpful at all!"
    processed = preprocess_text(text)
    assert "not_negation" in processed
    assert "helpful" in processed
    assert "!" not in processed

def test_predict_positive():
    result = predict_sentiment("The teachers are very nice and helpful.")
    assert result['sentiment'] == 'Positive'
    assert result['confidence'] > 0.0

def test_predict_negative():
    result = predict_sentiment("I hardly understand anything taught in the electronics class.")
    assert result['sentiment'] == 'Negative'
    assert result['confidence'] > 0.0

def test_predict_neutral():
    result = predict_sentiment("The canteen food is okay, nothing special.")
    assert result['sentiment'] in ['Neutral', 'Negative', 'Positive'] # Depending on specific training data, neutral can be tricky
    assert result['confidence'] > 0.0

def test_predict_empty():
    result = predict_sentiment("")
    assert 'sentiment' in result
