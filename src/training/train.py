import os
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocessing.text_cleaning import clean_text
from src.features.tfidf import build_tfidf_vectorizer


def load_data(data_path: str):
    df = pd.read_csv(data_path)
    return df


def preprocess_data(df: pd.DataFrame):
    # Clean text
    df["review_clean"] = df["review"].apply(clean_text)

    # Encode labels
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})

    return df


def train_model(X_train, y_train):
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def main():
    # Paths
    DATA_DIR = "data"
    MODEL_DIR = "models"

    DATA_PATH = os.path.join(DATA_DIR, "imdb.csv")

    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Preprocessing text...")
    df = preprocess_data(df)

    print("Splitting train/test data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["review_clean"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )

    print("Building TF-IDF features...")
    vectorizer = build_tfidf_vectorizer()

    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    print("Training Logistic Regression model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=["negative", "positive"]))

    print("Saving model artifacts...")
    joblib.dump(model, os.path.join(MODEL_DIR, "sentiment_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

    print("Training complete. Model and vectorizer saved.")


if __name__ == "__main__":
    main()
