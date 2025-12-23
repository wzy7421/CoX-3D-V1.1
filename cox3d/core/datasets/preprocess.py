import re

STOP = {"的","了","是","在","我","你","他","她","它","我们","你们","他们"}

def tokenize_zh(text: str):
    """
    Minimal tokenizer for reproducibility; replace with jieba if needed.
    """
    text = re.sub(r"\s+", " ", text.strip())
    return [c for c in text if c.strip()]

def denoise(tokens):
    return [t for t in tokens if t not in STOP]

def semantic_normalize(tokens):
    # Placeholder hook
    return tokens
