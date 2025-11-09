import re

def extract_questions_from_text(text: str):
    """
    Extract questions from text using flexible patterns that handle:
    - Numbered lists (1. Question, 1) Question, 1 Question)
    - Lines ending with '?' or typical question keywords
    - Supports both short and long questions
    """
    # Normalize spacing
    text = re.sub(r'\s+', ' ', text.strip())

    # Split into potential question blocks
    # Breaks on newlines, numbered sections, or question marks
    potential_questions = re.split(r'(?=\n?\d+[\). ]|[\?\n])', text)

    questions = []
    for q in potential_questions:
        q = q.strip()
        # Flexible detection logic
        if re.match(r'^\d+[\). ]', q) or q.endswith('?') or re.search(r'\b(explain|discuss|define|what|how|why|when|write|describe)\b', q, re.IGNORECASE):
            # Clean up numbering
            q = re.sub(r'^\d+[\). ]\s*', '', q)
            if 8 < len(q) < 500:  # filter out junk
                questions.append(q.strip())

    # Deduplicate
    seen = set()
    unique_questions = []
    for q in questions:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique_questions.append(q)

    return unique_questions
