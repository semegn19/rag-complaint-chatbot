from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# Common CFPB product label variants mapped to the target categories above.
PRODUCT_NORMALIZATION_MAP: Dict[str, str] = {
    "credit card": "Credit Card",
    "credit card or prepaid card": "Credit Card",
    "credit cards": "Credit Card",
    "personal loan": "Personal Loan",
    "payday loan": "Personal Loan",
    "payday loan, title loan, or personal loan": "Personal Loan",
    "student loan": "Personal Loan",
    "savings account": "Savings Account",
    "checking or savings account": "Savings Account",
    "deposit account": "Savings Account",
    "bank account or service": "Savings Account",
    "money transfer": "Money Transfer",
    "money transfers": "Money Transfer",
    "remittances": "Money Transfer",
    "international money transfer": "Money Transfer",
}

# Boilerplate phrases to strip from complaint narratives.
BOILERPLATE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bi am writing to file a complaint[^.]*\.?") ,
    re.compile(r"\bi am writing this complaint[^.]*\.?") ,
    re.compile(r"\bi would like to file a complaint[^.]*\.?") ,
    re.compile(r"\bplease help[^.]*\.?") ,
    re.compile(r"\bthis is regarding[^.]*\.?") ,
    re.compile(r"\bi am contacting you regarding[^.]*\.?") ,
    re.compile(r"\bi am writing because[^.]*\.?") ,
]

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """Clean complaint narrative text while preserving semantics."""
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    if not text:
        return ""

    # Remove boilerplate phrases.
    for pattern in BOILERPLATE_PATTERNS:
        text = pattern.sub(" ", text)

    # Remove URLs, email addresses, and HTML entities.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"&[a-z]+;", " ", text)

    # Replace punctuation / special characters with spaces, keeping apostrophes inside words.
    text = re.sub(r"[^a-z0-9\s']+", " ", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def word_count(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(text.split())


def normalize_product_label(product: str) -> Optional[str]:
    """Map raw CFPB product labels to the four target categories."""
    if pd.isna(product):
        return None

    raw = str(product).strip().lower()
    raw = re.sub(r"\s+", " ", raw)

    # Direct match first.
    if raw in PRODUCT_NORMALIZATION_MAP:
        return PRODUCT_NORMALIZATION_MAP[raw]

    # Fuzzy containment-based fallback for common variants.
    if "credit card" in raw:
        return "Credit Card"
    if "personal loan" in raw or "payday loan" in raw or "student loan" in raw:
        return "Personal Loan"
    if "savings account" in raw or "checking or savings account" in raw or "deposit account" in raw:
        return "Savings Account"
    if "money transfer" in raw or "money transfers" in raw or "remitt" in raw:
        return "Money Transfer"

    return None