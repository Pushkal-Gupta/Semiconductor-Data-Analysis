"""Preprocessing steps used in notebook 02.

These are split into small functions so the wrangling notebook can call them
one at a time and report how many columns each step removed. That kind of
running commentary is more useful than a single black-box pipeline for an
EDA-focused write-up.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns where the fraction of NaN exceeds `threshold` (default 50%)."""
    frac = df.isna().mean()
    dropped = frac[frac > threshold].index.tolist()
    return df.drop(columns=dropped), dropped


def drop_constant(df: pd.DataFrame, variance_threshold: float = 1e-12) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns whose variance is effectively zero.

    SECOM has a surprising number of these — sensors that were on but never
    moved during the measurement window.
    """
    variances = df.var(numeric_only=True)
    dropped = variances[variances <= variance_threshold].index.tolist()
    return df.drop(columns=dropped), dropped


def drop_correlated(df: pd.DataFrame, threshold: float = 0.95) -> Tuple[pd.DataFrame, List[str]]:
    """For each pair of features with |corr| > threshold, drop one of the two.

    Computed on already-imputed numeric data — call this after imputation, not before.
    """
    corr = df.corr().abs()
    # upper triangle only, so we look at each pair once
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [c for c in upper.columns if (upper[c] > threshold).any()]
    return df.drop(columns=dropped), dropped
