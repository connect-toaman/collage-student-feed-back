# Dataset Sources

## EduRABSA_ASTE
* **Name**: EduRABSA_ASTE
* **URL/Source**: https://huggingface.co/datasets/yhua219/EduRABSA_ASTE
* **Number of Records**: Extracted 33520 ABSA aspect annotations.
* **License**: Unspecified on HF, assume research/educational use.
* **Fields Used**: `text`, `output` (parsed for aspect, opinion, sentiment).
* **Sentiment Labels**: Positive, Negative, Neutral.
* **Aspect Labels**: Raw aspects provided by the dataset.
* **Redistribution Allowed**: Check HF repository for explicit license details.
* **Transformations Performed**: Parsed ASTE triplet array into flattened CSV rows (one per aspect).
