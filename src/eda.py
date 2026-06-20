from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.helpers import ensure_dir, word_count


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate summary stats and plots."""
    ensure_dir(output_dir)
    
    narrative_col = "Consumer complaint narrative"
    product_col = "Product"

    # Count complaints with and without narratives.
    df["has_narrative"] = df[narrative_col].notna() & df[narrative_col].astype(str).str.strip().ne("")
    narrative_counts = df["has_narrative"].value_counts(dropna=False).rename(index={True: "With narrative", False: "Without narrative"})
    narrative_counts.to_csv(output_dir / "narrative_presence_counts.csv")

    # Narrative length.
    df["narrative_word_count"] = df[narrative_col].fillna("").apply(word_count)
    df[["narrative_word_count"]].describe().to_csv(output_dir / "narrative_length_summary.csv")

    # Distribution by product.
    if product_col in df.columns:
        product_counts = df[product_col].value_counts(dropna=False)
        product_counts.to_csv(output_dir / "complaints_by_product_raw.csv")

        plt.figure(figsize=(12, 6))
        product_counts.head(20).sort_values().plot(kind="barh")
        plt.title("Complaints by Product (Top 20)")
        plt.xlabel("Complaint count")
        plt.ylabel("Product")
        plt.tight_layout()
        plt.savefig(output_dir / "complaints_by_product.png", dpi=200)
        plt.close()

    # Narrative length histogram.
    wc = df["narrative_word_count"]
    plt.figure(figsize=(12, 6))
    plt.hist(wc[wc > 0], bins=50)
    plt.title("Consumer Narrative Word Count Distribution")
    plt.xlabel("Word count")
    plt.ylabel("Number of complaints")
    plt.tight_layout()
    plt.savefig(output_dir / "narrative_word_count_histogram.png", dpi=200)
    plt.close()

    # Very short / very long entries.
    short_entries = df.loc[(df["narrative_word_count"] > 0) & (df["narrative_word_count"] <= 5), [product_col, narrative_col, "narrative_word_count"]].copy()
    long_threshold = df["narrative_word_count"].quantile(0.99)
    long_entries = df.loc[df["narrative_word_count"] >= long_threshold, [product_col, narrative_col, "narrative_word_count"]].copy()

    short_entries.to_csv(output_dir / "very_short_narratives.csv", index=False)
    long_entries.to_csv(output_dir / "very_long_narratives.csv", index=False)

    print("EDA complete")
    print("- Complaint rows:", len(df))
    print("- With narrative:", int(narrative_counts.get("With narrative", 0)))
    print("- Without narrative:", int(narrative_counts.get("Without narrative", 0)))
    print("- Median narrative word count:", float(df["narrative_word_count"].median()))
    print("- 99th percentile narrative word count:", float(long_threshold))