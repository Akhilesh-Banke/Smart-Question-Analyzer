# src/text_cleaner.py
import re
from collections import Counter

def normalize_question(text: str) -> str:
    """
    Normalize a question by removing punctuation, lowercasing, and trimming spaces.
    """
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s?]', '', text)
    return text.strip()

def get_frequent_questions(questions: list[str], top_k: int = 10):
    """
    Given a list of question strings, return the most frequent ones.
    Returns: list of (question, count)
    """
    if not questions:
        return []

    normalized = [normalize_question(q) for q in questions if q.strip()]
    freq_counter = Counter(normalized)
    return freq_counter.most_common(top_k)
