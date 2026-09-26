import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import joblib
import os

DATA_PATH = "data/faqs.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    df = pd.read_csv(DATA_PATH)

    corpus = (df["question"] + " " + df["category"]).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    vectors = vectorizer.fit_transform(corpus)
    vectors = normalize(vectors)

    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.joblib")
    joblib.dump(vectors, f"{MODEL_DIR}/vectors.joblib")
    joblib.dump(df, f"{MODEL_DIR}/faqs.joblib")

    print("Saved vectorizer and faq vectors.")

if __name__ == "__main__":
    train()