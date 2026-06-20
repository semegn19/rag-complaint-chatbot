from __future__ import annotations
from src.helpers import normalize_product_label, normalize_text, word_count


import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

TARGET_PRODUCTS = {
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
}

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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to target products, remove empty narratives, and clean text."""
    required_cols = ["Product", "Consumer complaint narrative"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()

    # Normalize product labels.
    work["product_normalized"] = work["Product"].apply(normalize_product_label)

    # Retain only the four requested categories.
    work = work[work["product_normalized"].isin(TARGET_PRODUCTS)].copy()

    # Remove missing / empty narratives.
    work["Consumer complaint narrative"] = work["Consumer complaint narrative"].astype("string")
    work = work[work["Consumer complaint narrative"].notna()]
    work = work[work["Consumer complaint narrative"].str.strip().ne("")]

    # Clean narrative text.
    work["narrative_clean"] = work["Consumer complaint narrative"].apply(normalize_text)
    work["narrative_word_count_clean"] = work["narrative_clean"].apply(word_count)

    # Drop rows that became empty after cleaning.
    work = work[work["narrative_clean"].str.strip().ne("")].copy()

    # Keep useful columns for downstream RAG, plus any metadata that exists.
    preferred_cols = [
        "Complaint ID",
        "Date received",
        "product_normalized",
        "Product",
        "Sub-product",
        "Issue",
        "Sub-issue",
        "narrative_clean",
        "narrative_word_count_clean",
        "Company",
        "State",
        "Submitted via",
        "Date sent to company",
        "Company response to consumer",
        "Timely response?",
        "Consumer disputed?",
    ]
    existing_cols = [c for c in preferred_cols if c in work.columns]
    work = work[existing_cols].copy()

    return work