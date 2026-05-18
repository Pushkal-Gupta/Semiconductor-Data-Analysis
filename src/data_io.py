"""Dataset loaders.

Kept deliberately small — the goal is to keep the notebooks readable, not to
build a framework. Each loader returns plain pandas / numpy objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_secom(raw_dir: str | Path = "data/raw/secom") -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load the SECOM dataset from UCI's two-file format.

    Returns
    -------
    X : DataFrame of shape (1567, 590), columns named sensor_0..sensor_589
    y : Series of 0 (pass) / 1 (fail). UCI uses -1/+1; we relabel for sklearn.
    ts : Series of timestamps from the labels file (kept in case we want to
         look at time-ordering, e.g. for sensor drift)
    """
    raw_dir = Path(raw_dir)
    data_path = raw_dir / "secom.data"
    labels_path = raw_dir / "secom_labels.data"

    if not data_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Expected SECOM files in {raw_dir}. See data/raw/secom/README.md "
            "for download instructions."
        )

    # The .data file is whitespace-separated, with literal 'NaN' for missing values.
    X = pd.read_csv(data_path, sep=r"\s+", header=None, na_values=["NaN"])
    X.columns = [f"sensor_{i}" for i in range(X.shape[1])]

    # Labels file format:  -1 "19/07/2008 11:55:00"
    labels = pd.read_csv(
        labels_path,
        sep=" ",
        header=None,
        names=["label", "timestamp"],
        quotechar='"',
    )
    ts = pd.to_datetime(labels["timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    y = labels["label"].map({-1: 0, 1: 1}).astype(int)

    return X, y, ts


def load_wm811k(pkl_path: str | Path = "data/raw/wm811k/LSWMD.pkl",
                labeled_only: bool = True) -> pd.DataFrame:
    """Load the WM-811K wafer-map dataset.

    The original pickle is ~400 MB and contains 811,457 wafer maps; most are
    unlabeled. Setting `labeled_only=True` (the default) keeps only the
    ~172k rows that have a `failureType` annotation, which is what we model
    against.
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Expected {pkl_path}. See data/raw/wm811k/README.md for the Kaggle download link."
        )

    df = pd.read_pickle(pkl_path)

    # `failureType` is stored inconsistently: sometimes a string, sometimes a
    # numpy array wrapping a string, sometimes an empty array. Normalize to str.
    def _flatten_label(v):
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return ""
            return str(v.flatten()[0])
        return str(v) if v is not None else ""

    df["label"] = df["failureType"].apply(_flatten_label).str.strip()

    if labeled_only:
        df = df[df["label"].ne("") & df["label"].notna()].reset_index(drop=True)

    return df
