import re


NUM_PATTERN = re.compile(r'^\s*(?:Q(?:uestion)?\.?\s*)?(\d+)[\)\.:\-]\s*(.*)', re.I)


# Very simple extractor: collects lines that look like questions


def extract_questions_from_text(text):
    lines = text.splitlines()
    out = []
    buffer = ''
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = NUM_PATTERN.match(line)
        if m:
            # new numbered question
            q = m.group(2).strip()
            if q:
                out.append(q)
            continue
        # lines ending with question mark
        if line.endswith('?'):
            out.append(line)
            continue
        # fallback: lines that start with 'Question'
        if line.lower().startswith('question'):
            # pick rest
            parts = line.split(None, 1)
            if len(parts) > 1:
                out.append(parts[1])
    return out