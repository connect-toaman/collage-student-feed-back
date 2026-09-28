# Aspect-Based Sentiment Analysis (ABSA)

## 1. What is ABSA?
Aspect-Based Sentiment Analysis (ABSA) breaks down text into specific components (aspects) and determines the sentiment for each component individually, rather than assigning a single sentiment to the entire text.

## 2. Document-level vs ABSA
- **Document-level**: Takes a string (e.g. "The teachers are good but the laboratory equipment needs improvement.") and outputs a single sentiment (e.g., Neutral/Mixed).
- **ABSA**: Parses the string, isolates the aspects, and returns:
  - Teachers -> Positive
  - Laboratory -> Negative

## 3. Aspect Extraction
This project uses a controlled domain-specific vocabulary located in `config/aspects.json`. The extraction module (`ml/aspect_extractor.py`) parses the input text, splits it around grammatical conjunctions (e.g. "but", "however", "although"), and matches known synonyms to a canonical aspect category.

## 4. Aspect Sentiment Classification
Rather than building an entirely new ABSA-specific ML model, the system smartly reuses the highly-trained existing document-level classifier (`predict_sentiment`). It does this by extracting only the *surrounding clause/context* of the aspect and passing that specific sub-string to the model.

## 5. Mixed Sentiment Logic
The overall ABSA sentiment aggregation logic is:
- **Positive**: Contains positive aspects, zero negative aspects.
- **Negative**: Contains negative aspects, zero positive aspects.
- **Mixed**: Contains at least one positive AND at least one negative aspect.
- **Neutral**: All extracted aspects are neutral/factual.

## 6. Confidence
The confidence scores shown for each aspect are inherited directly from the underlying Logistic Regression model's probability distribution for that specific clause. If confidence drops below 60%, the UI flags it with a "Low confidence — interpret with caution" warning.

## 7. Aspect Taxonomy
Configurable via `config/aspects.json`. Current categories include:
- Teachers / Faculty
- Teaching Quality
- Course Content
- Curriculum
- Laboratory / Lab Equipment
- Classrooms
- Library
- Internet / Wi-Fi
- Infrastructure / Facilities
... and more.

## 8. Architecture
- **`config/aspects.json`**: Taxonomy definitions.
- **`ml/aspect_extractor.py`**: Splits sentences into clauses and extracts aspects.
- **`ml/aspect_sentiment.py`**: Coordinates extraction and ML model invocation.
- **`app/routes.py`**: Exposes `/api/analyze`.

## 9. API Format
```json
POST /api/analyze
{
  "text": "The teachers are good but the laboratory equipment needs improvement."
}
```
**Response**:
```json
{
  "text": "The teachers are good but the laboratory equipment needs improvement.",
  "overall_sentiment": {
    "label": "Neutral",
    "confidence": 0.51,
    "probabilities": {
      "positive": 0.33,
      "neutral": 0.51,
      "negative": 0.16
    }
  },
  "aspect_based_analysis": {
    "overall": "Mixed",
    "aspects": [
      {
        "aspect": "Teachers / Faculty",
        "sentiment": "Positive",
        "confidence": 0.66,
        "context": "The teachers are good"
      },
      {
        "aspect": "Laboratory / Lab Equipment",
        "sentiment": "Positive",
        "confidence": 0.38,
        "context": "the laboratory equipment needs improvement"
      }
    ]
  }
}
```

## 10. Limitations
- The aspect extractor relies on regex dictionary matching. It cannot identify zero-shot/implicit aspects (e.g. inferring "Infrastructure" from "the roof is leaking" without "roof" being in the taxonomy).
- The existing document model is trained on full sentences. Sometimes, very short isolated clauses lack enough signal for high-confidence predictions.
