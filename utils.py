from typing import Any, Dict, List, Tuple
from pathlib import Path
from bs4 import BeautifulSoup
import re, json, unicodedata, html, os, random
import numpy as np

def get_combine_text(doc):
    title = doc.get("title") or ""
    description = doc.get("description") or ""
    post_content = doc.get("post_content") or ""
    content = doc.get("content") or ""
    return "\n".join([p for p in [title , description , post_content , content] if p]).strip()

_sentence_splitter = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])")

def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _sentence_splitter.split(text)
    if len(parts) == 1:
        parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]

def estimate_tokens(text: str) -> int:
    w = max(1, len(re.findall(r"\S+", text)))
    return int(w / 0.75)

def clean_text(text: str) -> str:
    """Remove HTML, extra spaces, and non-alphanumeric chars."""
    # 1. Parse HTML & extract text
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text()

    # 2. Decode HTML entities
    clean_text = html.unescape(clean_text)

    # 3. Remove inline attributes like style="..."
    clean_text = re.sub(r'\b\w+=(["\']).*?\1', '', clean_text)

    # 4. Normalize symbols
    clean_text = clean_text.replace('&', 'and').replace('%', 'percent')

    # 5. Remove control characters and emojis
    clean_text = ''.join(
        c for c in clean_text 
        if not unicodedata.category(c).startswith('C') and unicodedata.category(c) != 'So'
    )

    # 6. Remove unwanted symbols
    clean_text = re.sub(r'[~=/]', ' ', clean_text)

    # 8. Collapse multiple spaces
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()

    return clean_text.lower().strip()

def query_processor(query: str) -> str:
    if "//" in query:
        target = query.split("//", 1)[0].strip()
        query = query.split("//", 1)[1].strip()
        query = f"{target} {query}".strip()
    return query

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)