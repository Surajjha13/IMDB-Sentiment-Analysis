import os
import joblib

from src.inference.predict import predict_sentiment


def test_model_files_exist():
    """
    Test that trained model artifacts exist and load successfully.
    """
    assert os.path.exists("models/sentiment_model.pkl")
    assert os.path.exists("models/tfidf_vectorizer.pkl")

    model = joblib.load("models/sentiment_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    assert model is not None
    assert vectorizer is not None


def test_prediction_output():
    """
    Test that prediction returns expected output format.
    """
    result = predict_sentiment("The movie was amazing and thrilling")

    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result

    assert result["sentiment"] in ["positive", "negative"]
    assert 0.0 <= result["confidence"] <= 1.0
