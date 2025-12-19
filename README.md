<think>
First, the user wants me to convert the entire provided text into a downloadable README.md file. The text is a description of a project, structured like a README.

- The user needs the project description converted into a downloadable README.md file.
</think># IMDB Sentiment Analysis — End-to-End ML + CI/CD

This project implements an end-to-end sentiment analysis system using classical NLP techniques and demonstrates production-ready ML engineering practices, including API deployment, Docker containerization, and CI/CD automation.

## Project Overview

**Problem:** Classify movie reviews as positive or negative

**Dataset:** IMDB Movie Reviews (50K samples)

**Model:** TF-IDF + Logistic Regression

**API:** FastAPI

**Deployment:** Docker

**CI/CD:** GitHub Actions

This project was developed as part of a structured learning task to understand ML workflows, deployment, and CI/CD concepts.

## Model & Approach

### 🔹 Dataset

IMDB movie reviews

Balanced binary sentiment labels:

- positive
- negative

### Preprocessing

- Lowercasing
- HTML tag removal
- URL removal
- Whitespace normalization

### Feature Engineering

- TF-IDF vectorization
- Word n-grams (1–3)
- Sublinear TF scaling

### Models Tested

- Logistic Regression (used for deployment)
- Linear SVM (used for comparison)

**Why Logistic Regression?**

- Comparable accuracy to SVM
- Provides probability scores (useful for APIs)
- Faster and easier to deploy

## Model Performance

| Model                      | Accuracy |
|----------------------------|----------|
| Logistic Regression (baseline) | ~90%    |
| Linear SVM (comparison)    | ~91–92% |

## Project Structure

```
sentiment_analysis/
│
├── api/
│   └── main.py                # FastAPI application
│
├── src/
│   ├── preprocessing/
│   │   └── text_cleaning.py
│   ├── features/
│   │   └── tfidf.py
│   ├── training/
│   │   └── train.py
│   └── inference/
│       └── predict.py
│
├── models/
│   ├── sentiment_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── tests/
│   └── test_model.py          # Automated tests
│
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
│
├── Dockerfile
├── requirements.txt
├── README.md
```

## Running the Project Locally

### Train the Model

```bash
python -m src.training.train
```

This will:

- Preprocess data
- Train the model
- Save artifacts to models/

### Run FastAPI Server

```bash
uvicorn api.main:app --reload
```

API will be available at:

http://127.0.0.1:8000

Swagger UI:

http://127.0.0.1:8000/docs

### Test Prediction Endpoint

**Request**

```
POST /predict
{
  "text": "The movie was absolutely fantastic!"
}
```

**Response**

```json
{
  "sentiment": "positive",
  "confidence": 0.94
}
```

## Docker Usage

### Build Image

```bash
docker build -t imdb-sentiment-api .
```

### Run Container

```bash
docker run -p 8000:8000 imdb-sentiment-api
```

API will be available at:

http://localhost:8000

## Automated Tests

Basic tests ensure:

- Model files load successfully
- Prediction output format is valid

Run locally:

```bash
pytest
```

## CI/CD Pipeline (GitHub Actions)

On every push to main:

- Code is checked out
- Dependencies are installed
- Tests are executed
- Docker image is built

Pipeline file:

.github/workflows/ci.yml

This ensures:

- Model integrity
- Reproducible builds
- Early failure detection

## Tech Stack

- Python 3.10
- Pandas
- Scikit-learn
- FastAPI
- Docker
- Pytest
- GitHub Actions

## Future Improvements

- Add model versioning
- Add API test cases in CI
- Push Docker image to GHCR
- Add logging and monitoring
- Try transformer-based models

## Key Learnings

- Treat ML models as software artifacts
- Validate models before deployment
- Use Docker for reproducibility
- Automate ML workflows using CI/CD
- Prefer simplicity and clarity over overengineering

## Author

Your Name  : Suraj Kumar Jha