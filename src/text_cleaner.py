import re


def normalize_question(q):
    if not q: return q
    q = q.strip()
    q = re.sub(r'\s+', ' ', q)
    # remove leading numbering like '1.' or '(i)'
    q = re.sub(r'^\(?\d+\)?[\.)\-:\s]+', '', q)
    # remove page footers like 'Page 3 of 10'
    q = re.sub(r'page\s*\d+(\s*of\s*\d+)?', '', q, flags=re.I)
    return q