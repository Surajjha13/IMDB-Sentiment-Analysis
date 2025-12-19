import os
import joblib

from src.preprocessing.text_cleaning import clean_text

# Resolve base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Load artifacts once (at import time)
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single input text.
    Returns sentiment label and confidence score.
    """

    cleaned_text = clean_text(text)

    X = vectorizer.transform([cleaned_text])

    prediction = model.predict(X)[0]
    probability = model.predict_proba(X).max()

    sentiment = "positive" if prediction == 1 else "negative"

    return {
        "sentiment": sentiment,
        "confidence": round(float(probability), 4)
    }

if __name__ == "__main__":
    print(predict_sentiment("The movie was boring and too long"))
