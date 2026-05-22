"""Chronological train/val/test splits (no shuffle) for time series."""

from __future__ import annotations

import pandas as pd


def time_series_fraction_split(
    n: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[slice, slice, slice]:
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    if n < 3:
        raise ValueError("Need at least 3 rows for a meaningful split.")
    i_train_end = max(1, int(n * train_frac))
    i_val_end = max(i_train_end + 1, int(n * (train_frac + val_frac)))
    if i_val_end >= n:
        i_val_end = n - 1
    return slice(0, i_train_end), slice(i_train_end, i_val_end), slice(i_val_end, n)


def split_by_date(
    df: pd.DataFrame,
    date_col: str = "Date",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    Sort by date, then contiguous index splits (70/15/15 default).
    Returns train, val, test and a small metadata dict with cut dates.
    """
    out = df.sort_values(date_col).reset_index(drop=True)
    tr_sl, va_sl, te_sl = time_series_fraction_split(len(out), train_frac, val_frac, test_frac)
    train, val, test = out.iloc[tr_sl], out.iloc[va_sl], out.iloc[te_sl]
    meta = {
        "n_total": int(len(out)),
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "train_date_end": train[date_col].max(),
        "val_date_end": val[date_col].max(),
        "test_date_end": test[date_col].max(),
        "split": "chronological_70_15_15",
    }
    return train, val, test, meta
