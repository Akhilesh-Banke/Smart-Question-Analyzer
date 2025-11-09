
"""
Module for extracting question-like sentences from OCR text.
Uses regex patterns and simple NLP preprocessing to detect questions.
"""

import re


def extract_questions_from_text(text: str) -> list[str]:
    """
    Extracts question-like sentences from a block of text.

    Args:
        text (str): OCR-extracted text from PDF page.

    Returns:
        list[str]: Extracted questions.
    """

    if not text or not isinstance(text, str):
        return []

    # --- Normalize text ---
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\n', ' ').strip()

    # --- Common question patterns ---
    patterns = [
        r"(Q\d+[\).:-]?\s*)(.*?)(?=(Q\d+[\).:-])|$)",          # Q1, Q2... style
        r"(\d+[\).:-]?\s*)(what|why|how|when|explain|define|describe|differentiate|write|list|state|discuss|draw).*?\?",  # numbered + WH
        r"\b(what|why|how|when|explain|define|describe|differentiate|write|list|state|discuss|draw)\b.*?[\?\.]",  # plain WH questions
    ]

    # --- Extract matches ---
    questions = []
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for m in matches:
            # Some patterns return tuples, handle both cases
            q = m if isinstance(m, str) else " ".join([x for x in m if x])
            q = q.strip()
            if len(q.split()) > 2 and len(q) < 400:  # reasonable question length
                questions.append(q)

    # --- Remove duplicates & clean ---
    unique_questions = []
    seen = set()
    for q in questions:
        q_clean = re.sub(r'^(q\d+[\).:-]?\s*)', '', q, flags=re.IGNORECASE).strip()
        q_clean = re.sub(r'\s+', ' ', q_clean)
        if q_clean.lower() not in seen:
            unique_questions.append(q_clean)
            seen.add(q_clean.lower())

    return unique_questions


def extract_all_questions(text_blocks: list[str]) -> list[str]:
    """
    Process multiple text blocks (pages) and extract questions from each.

    Args:
        text_blocks (list[str]): List of OCR-extracted texts.

    Returns:
        list[str]: Combined list of questions from all pages.
    """
    all_questions = []
    for block in text_blocks:
        qs = extract_questions_from_text(block)
        all_questions.extend(qs)
    return all_questions
