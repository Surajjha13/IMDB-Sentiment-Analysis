from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf_vectorizer():
    """
    Creates and returns a TF-IDF vectorizer
    configured for sentiment analysis.
    """
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=5,
        stop_words="english"
    )
