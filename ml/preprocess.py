import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import os

# Ensure nltk data is downloaded
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt_tab')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('wordnet')

download_nltk_data()

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    
    # Lowercase conversion
    text = text.lower()
    
    # Preserve some meaningful negations before removing punctuation
    text = re.sub(r'\bnot\b', 'not_negation', text)
    text = re.sub(r'\bno\b', 'no_negation', text)
    text = re.sub(r'\bnever\b', 'never_negation', text)
    text = re.sub(r'\bhardly\b', 'hardly_negation', text)
    text = re.sub(r'\bneither\b', 'neither_negation', text)
    text = re.sub(r"\bdon'?t\b", 'not_negation', text)
    text = re.sub(r"\bdoesn'?t\b", 'not_negation', text)
    text = re.sub(r"\bisn'?t\b", 'not_negation', text)
    text = re.sub(r"\bwasn'?t\b", 'not_negation', text)
    text = re.sub(r"\baren'?t\b", 'not_negation', text)
    text = re.sub(r"\bwouldn'?t\b", 'not_negation', text)
    text = re.sub(r"\bshouldn'?t\b", 'not_negation', text)
    text = re.sub(r"\bcouldn'?t\b", 'not_negation', text)
    text = re.sub(r"\bcan'?t\b", 'not_negation', text)
    text = re.sub(r"\bwon'?t\b", 'not_negation', text)
    
    # Remove special characters and digits, keep spaces and basic word characters
    text = re.sub(r'[^a-zA-Z\s_]', '', text)
    
    # Whitespace normalization
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(lemmatized_tokens)
