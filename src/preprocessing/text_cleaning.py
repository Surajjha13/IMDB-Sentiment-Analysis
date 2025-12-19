import re

def clean_text(text: str) -> str:
    """
    Basic text cleaning for sentiment analysis.
    - Lowercases text
    - Removes HTML tags
    - Removes URLs
    - Normalizes whitespace
    """

    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text