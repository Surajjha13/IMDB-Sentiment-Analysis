from fastapi import FastAPI
from pydantic import BaseModel

from src.inference.predict import predict_sentiment

app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Sentiment analysis using TF-IDF + Logistic Regression",
    version="1.0.0"
)


# ---------- Request & Response Schemas ----------

class ReviewRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float


# ---------- Health Check ----------

@app.get("/")
def health_check():
    return {"status": "API is running"}


# ---------- Prediction Endpoint ----------

@app.post("/predict", response_model=PredictionResponse)
def predict(review: ReviewRequest):
    """
    Predict sentiment for a given review text.
    """
    result = predict_sentiment(review.text)
    return result
