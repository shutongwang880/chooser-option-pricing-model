"""Feature matrices and targets for Week 5 ML pipelines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from chooser_pricing import black_scholes_call

ROOT = Path(__file__).resolve().parents[1]


def normalize_div_yield(dy: float, default: float = 0.025) -> float:
    if dy is None or (isinstance(dy, float) and np.isnan(dy)):
        return default
    dy = float(dy)
    if dy > 0.2:
        return dy / 100.0
    return dy


def load_cleaned(path: Path | None = None) -> pd.DataFrame:
    p = path or ROOT / "data" / "cleaned_dataset.csv"
    df = pd.read_csv(p, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def load_config(path: Path | None = None) -> dict:
    p = path or ROOT / "config.json"
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# Columns aligned with week2 cleaned_dataset
VOL_FEATURE_COLS: list[str] = [
    "log_return",
    "rolling_vol_21",
    "div_yield",
    "yield_slope",
    "rate_momentum",
    "vix_jpm_corr",
    "sentiment_score",
    "VIX_Close",
    "Treasury_3M",
    "Treasury_10Y",
    "JPM_Close",
]

# End-to-end pricing: omit same-day realized vol so the model cannot trivially invert BSM
PRICE_FEATURE_COLS: list[str] = [c for c in VOL_FEATURE_COLS if c != "rolling_vol_21"]


def build_volatility_supervised_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Predict next row's 21d rolling vol from features at t (no overlap with target window).
    Drops tail rows with missing target.
    """
    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["y_vol_next"] = d["rolling_vol_21"].shift(-1)
    d = d.dropna(subset=["y_vol_next"] + VOL_FEATURE_COLS)
    return d, "y_vol_next", VOL_FEATURE_COLS


def build_daily_teacher_bsm_frame(
    df: pd.DataFrame,
    strike: float,
    expiration: pd.Timestamp,
) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Each row: X = PRICE_FEATURE_COLS; y = BSM call using same-day realized vol
    (rolling_vol_21 annualized) as teacher sigma — supervised pricing on a dense calendar.
    """
    d = df.sort_values("Date").reset_index(drop=True).copy()
    d["Date"] = pd.to_datetime(d["Date"]).dt.normalize()
    exp = pd.Timestamp(expiration).normalize()
    days = (exp - d["Date"]).dt.days.astype(float).clip(lower=1.0)
    d["T_years"] = days / 365.25
    need = PRICE_FEATURE_COLS + ["rolling_vol_21", "JPM_Close", "Treasury_3M", "div_yield"]
    d = d.dropna(subset=need)
    sig = np.clip(d["rolling_vol_21"].astype(float).values * np.sqrt(252.0), 0.01, 2.0)
    r = d["Treasury_3M"].astype(float).values / 100.0
    q = np.array([normalize_div_yield(v) for v in d["div_yield"].astype(float).values], dtype=float)
    S = d["JPM_Close"].astype(float).values
    T = d["T_years"].astype(float).values
    d["y_bs_call"] = black_scholes_call(S, strike, T, r, q, sig)
    return d, "y_bs_call", PRICE_FEATURE_COLS


def enrich_panel_for_week5(panel: pd.DataFrame, cleaned_path: Path) -> pd.DataFrame:
    """
    Panel rows (often post-cleaned calendar) need the same feature columns as `cleaned_dataset`.
    Rebuild a Yahoo-backed daily slice and derived macro features, then left-merge onto panel.
    """
    from week4_baseline_evaluation import (
        build_extended_features,
        merge_cleaned_where_possible,
        yf_history_close,
    )

    dates = pd.to_datetime(panel["Date"]).dt.normalize()
    ext = build_extended_features(dates)
    ext = merge_cleaned_where_possible(ext, cleaned_path)
    ext = ext.sort_index()
    d0, d1 = ext.index.min(), ext.index.max()
    tnx = yf_history_close("^TNX", d0 - pd.Timedelta(days=60), d1 + pd.Timedelta(days=2))
    ext = ext.copy()
    ext["Treasury_10Y"] = tnx.reindex(ext.index).ffill().bfill()
    ext["yield_slope"] = ext["Treasury_10Y"] - ext["Treasury_3M"]
    ext["rate_momentum"] = ext["Treasury_10Y"].diff()
    ext["vix_jpm_corr"] = ext["JPM_Close"].rolling(21, min_periods=5).corr(ext["VIX_Close"])
    ext = ext.reset_index()
    if "Date" not in ext.columns:
        ext = ext.rename(columns={ext.columns[0]: "Date"})
    fe = ext[["Date"] + VOL_FEATURE_COLS].copy()
    out = panel.copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    saved = out[["Date"] + [c for c in VOL_FEATURE_COLS if c in out.columns]].copy()
    drop_cols = [c for c in VOL_FEATURE_COLS if c in out.columns]
    out = out.drop(columns=drop_cols, errors="ignore")
    out = out.merge(fe, on="Date", how="left")
    saved = saved.rename(columns={c: f"{c}__panel" for c in VOL_FEATURE_COLS if c in saved.columns})
    out = out.merge(saved, on="Date", how="left")
    for c in VOL_FEATURE_COLS:
        panel_col = f"{c}__panel"
        if panel_col in out.columns:
            out[c] = out[panel_col].combine_first(out[c])
            out = out.drop(columns=[panel_col])
    out[VOL_FEATURE_COLS] = out[VOL_FEATURE_COLS].bfill().ffill()
    if "Treasury_10Y" in out.columns and "Treasury_3M" in out.columns:
        out["Treasury_10Y"] = out["Treasury_10Y"].fillna(out["Treasury_3M"])
    if "yield_slope" in out.columns and {"Treasury_10Y", "Treasury_3M"}.issubset(out.columns):
        out["yield_slope"] = out["yield_slope"].fillna(out["Treasury_10Y"] - out["Treasury_3M"])
    if "rate_momentum" in out.columns:
        out["rate_momentum"] = out["rate_momentum"].fillna(0.0)
    if "vix_jpm_corr" in out.columns:
        out["vix_jpm_corr"] = out["vix_jpm_corr"].fillna(0.0)
    return out


def panel_feature_matrix(panel: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    missing = [c for c in feature_cols if c not in panel.columns]
    if missing:
        raise KeyError(f"Panel missing columns: {missing}")
    return panel[feature_cols].astype(float).values
