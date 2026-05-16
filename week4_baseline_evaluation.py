# ============================================================
# Week 4 — Baseline performance vs market option prices
#
# - Row-level errors: Black–Scholes *European call* vs Yahoo option
#   close (liquidity proxy; not CME tape).
# - Chooser MC (paper-style contract) reported as scalar benchmark.
# - Limitation splits: high VIX vs rest; sentiment tertiles when available.
#
# Requires: dates in market CSV may be outside cleaned_dataset range;
# missing inputs are filled from Yahoo (^IRX, JPM, ^VIX) + local vol estimate.
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from chooser_pricing import black_scholes_call, chooser_option_mc

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DEFAULT_MARKET = DATA_DIR / "week4_market_proxy.csv"
FALLBACK_MARKET = DATA_DIR / "week4_liquid_call.csv"
DEFAULT_CLEANED = DATA_DIR / "cleaned_dataset.csv"
DEFAULT_CONFIG = ROOT / "config.json"
OUT_PANEL = DATA_DIR
OUT_BENCHMARK = DATA_DIR


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_div_yield(dy: float, default: float = 0.025) -> float:
    """Yahoo may return 2.0 for 2%; BS formula needs decimal (0.02)."""
    if dy is None or (isinstance(dy, float) and np.isnan(dy)):
        return default
    dy = float(dy)
    if dy > 0.2:
        return dy / 100.0
    return dy


def _flatten_yf(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] if isinstance(c, tuple) else c for c in out.columns]
    return out


def yf_history_close(ticker: str, start, end) -> pd.Series:
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    raw = _flatten_yf(raw)
    if raw.empty or "Close" not in raw.columns:
        return pd.Series(dtype=float)
    s = raw["Close"].copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def build_extended_features(dates: pd.Series, buffer_days: int = 50) -> pd.DataFrame:
    """Daily S, r, VIX, rolling vol, q for calendar spanning `dates` (+ buffer)."""
    dates = pd.to_datetime(dates).dt.normalize()
    d0, d1 = dates.min(), dates.max()
    start = d0 - pd.Timedelta(days=buffer_days)
    end = d1 + pd.Timedelta(days=1)

    jpm = yf_history_close("JPM", start, end)
    vix = yf_history_close("^VIX", start, end)
    irx = yf_history_close("^IRX", start, end)  # 13-week yield, quoted in %

    df = pd.DataFrame({"JPM_Close": jpm, "VIX_Close": vix, "Treasury_3M": irx})
    df = df.sort_index().ffill().bfill()
    df["log_return"] = np.log(df["JPM_Close"] / df["JPM_Close"].shift(1))
    df["rolling_vol_21"] = df["log_return"].rolling(21, min_periods=5).std()
    # Dividend yield: optional from Yahoo trailing; fallback constant for short panels
    try:
        t = yf.Ticker("JPM")
        dy = t.info.get("dividendYield") or t.fast_info.get("dividend_yield")
        if dy is not None and dy == dy:
            q_const = normalize_div_yield(float(dy))
        else:
            q_const = 0.025
    except Exception:
        q_const = 0.025
    df["div_yield"] = q_const
    df["r_decimal"] = df["Treasury_3M"] / 100.0
    df["sigma_annual"] = df["rolling_vol_21"] * np.sqrt(252)
    # Sentiment proxy on available VIX window (min-max on extended sample)
    vmin, vmax = df["VIX_Close"].min(), df["VIX_Close"].max()
    denom = max(vmax - vmin, 1e-9)
    df["sentiment_score"] = (df["VIX_Close"] - vmin) / denom
    return df


def merge_cleaned_where_possible(
    feats: pd.DataFrame, cleaned_path: Path
) -> pd.DataFrame:
    if not cleaned_path.exists():
        feats.index.name = feats.index.name or "Date"
        return feats
    cl = pd.read_csv(cleaned_path, parse_dates=["Date"])
    cl["Date"] = cl["Date"].dt.normalize()
    feats = feats.copy()
    feats.index.name = feats.index.name or "Date"
    feats = feats.reset_index()
    merged = feats.merge(
        cl[
            [
                "Date",
                "rolling_vol_21",
                "div_yield",
                "Treasury_3M",
                "VIX_Close",
                "sentiment_score",
            ]
        ],
        on="Date",
        how="left",
        suffixes=("", "_cleaned"),
    )
    # Prefer cleaned columns when present
    for c in ["rolling_vol_21", "div_yield", "Treasury_3M", "VIX_Close", "sentiment_score"]:
        alt = f"{c}_cleaned"
        if alt in merged.columns:
            merged[c] = merged[alt].combine_first(merged[c])
            merged.drop(columns=[alt], inplace=True)
    merged["sigma_annual"] = merged["rolling_vol_21"] * np.sqrt(252)
    merged["r_decimal"] = merged["Treasury_3M"] / 100.0
    merged["div_yield"] = merged["div_yield"].apply(normalize_div_yield)
    return merged.set_index("Date")


def mae_rmse(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    e = yhat - y
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "n": int(len(y)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 4 baseline evaluation")
    ap.add_argument("--market-csv", type=Path, default=DEFAULT_MARKET)
    ap.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "--out-panel",
        type=Path,
        default=OUT_PANEL,
        help="Folder for evaluation_panel.csv",
    )
    ap.add_argument(
        "--out-benchmark",
        type=Path,
        default=OUT_BENCHMARK,
        help="Folder for metrics.json",
    )
    ap.add_argument("--n-sims-chooser", type=int, default=40_000)
    ap.add_argument(
        "--make-report-artifacts",
        action="store_true",
        help="Also regenerate optional markdown reports and figures.",
    )
    args = ap.parse_args()

    if not args.market_csv.exists() and FALLBACK_MARKET.exists():
        args.market_csv = FALLBACK_MARKET

    if not args.market_csv.exists():
        print(f"Missing {args.market_csv}; run week4_cme_data.py yahoo-history --auto-liquid ...", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    X_paper = float(cfg["contract_terms"]["X"])
    T1 = float(cfg["contract_terms"]["T1_years"])
    T2 = float(cfg["contract_terms"]["T2_years"])

    mkt = pd.read_csv(args.market_csv, parse_dates=["Date"])
    mkt["Date"] = mkt["Date"].dt.normalize()
    strike = float(mkt["strike"].iloc[0])
    exp = pd.Timestamp(mkt["expiration"].iloc[0])

    feats = build_extended_features(mkt["Date"])
    feats = merge_cleaned_where_possible(feats, args.cleaned)
    panel = mkt.merge(feats.reset_index(), on="Date", how="inner")
    if panel.empty:
        print("No overlapping dates between market CSV and built features.", file=sys.stderr)
        sys.exit(1)

    # Time to expiry (years) for the *listed vanilla call*
    panel["T_years"] = (exp - panel["Date"]).dt.days.clip(lower=1) / 365.25
    y = panel["market_price_close"].astype(float).values
    S = panel["JPM_Close"].astype(float).values
    r = panel["r_decimal"].astype(float).values
    q = panel["div_yield"].astype(float).values
    sig = panel["sigma_annual"].astype(float).values
    q = np.array([normalize_div_yield(v) for v in q], dtype=float)
    # Clip degenerate vol
    sig = np.clip(sig, 0.01, 2.0)

    yhat_bs = black_scholes_call(S, strike, panel["T_years"].values, r, q, sig)
    panel["bs_call_model"] = yhat_bs
    panel["residual"] = yhat_bs - y

    overall = mae_rmse(y, yhat_bs)

    vix = panel["VIX_Close"].astype(float).values
    v75 = np.quantile(vix, 0.75)
    hi = vix >= v75
    lo = ~hi
    by_vol = {
        "high_VIX": mae_rmse(y[hi], yhat_bs[hi]) if hi.any() else {"MAE": None, "RMSE": None, "n": 0},
        "rest": mae_rmse(y[lo], yhat_bs[lo]) if lo.any() else {"MAE": None, "RMSE": None, "n": 0},
        "VIX_threshold": float(v75),
    }

    sent = panel["sentiment_score"].astype(float).values
    t33, t66 = np.quantile(sent, (1 / 3, 2 / 3))
    low_s, mid_s, hi_s = sent <= t33, (sent > t33) & (sent <= t66), sent > t66
    by_sent = {
        "low_sentiment_tercile": mae_rmse(y[low_s], yhat_bs[low_s]) if low_s.any() else {},
        "mid_sentiment_tercile": mae_rmse(y[mid_s], yhat_bs[mid_s]) if mid_s.any() else {},
        "high_sentiment_tercile": mae_rmse(y[hi_s], yhat_bs[hi_s]) if hi_s.any() else {},
    }

    # Chooser scalar benchmarks (not comparable row-wise to vanilla call)
    med = panel.median(numeric_only=True)
    chooser_paper = chooser_option_mc(
        float(med["JPM_Close"]),
        X_paper,
        T1,
        T2,
        float(med["r_decimal"]),
        float(med["div_yield"]),
        float(med["sigma_annual"]),
        n_sims=args.n_sims_chooser,
        seed=int(cfg["simulation"]["random_seed"]),
    )
    chooser_at_listed_k = chooser_option_mc(
        float(med["JPM_Close"]),
        strike,
        T1,
        T2,
        float(med["r_decimal"]),
        float(med["div_yield"]),
        float(med["sigma_annual"]),
        n_sims=args.n_sims_chooser,
        seed=int(cfg["simulation"]["random_seed"]) + 1,
    )

    args.out_panel.mkdir(parents=True, exist_ok=True)
    args.out_benchmark.mkdir(parents=True, exist_ok=True)
    panel_path = args.out_panel / "week4_evaluation_panel.csv"
    metrics_path = args.out_benchmark / "week4_metrics.json"
    panel.to_csv(panel_path, index=False)

    metrics = {
        "instrument": {
            "description": "Yahoo OCC last/close for listed call (CME transaction proxy if course allows).",
            "occ_symbol": str(panel["occ_symbol"].iloc[0]),
            "strike": strike,
            "expiration": str(exp.date()),
        },
        "vanilla_bs_call_vs_market": {"overall": overall, "by_VIX_regime": by_vol, "by_sentiment_terciles": by_sent},
        "chooser_mc_benchmark_scalar": {
            "paper_strike_X": X_paper,
            "T1_years": T1,
            "T2_years": T2,
            "at_median_panel_inputs": chooser_paper,
            "at_median_panel_inputs_strike_equals_listed": chooser_at_listed_k,
            "n_sims": args.n_sims_chooser,
        },
        "notes": [
            "MAE/RMSE use European BS call vs the same strike/expiry as the Yahoo series.",
            "Chooser price is not the same object as a vanilla call; scalar MC is for BSM-style baseline documentation only.",
        ],
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(overall, indent=2))
    print(f"Wrote {panel_path}")
    print(f"Wrote {metrics_path}")

    if args.make_report_artifacts:
        try:
            from scripts.generate_part_b_figures import main as gen_deliverable_reports

            gen_deliverable_reports()
            _merge_validation_report(ROOT / "artifacts" / "week4" / "01_Model_Validation_Report", metrics)
        except Exception as exc:
            print(f"Note: deliverable charts/reports skipped ({exc})", file=sys.stderr)


def _merge_validation_report(report_dir: Path, metrics: dict) -> None:
    """Deliverable 1 = Section 0 (paper) + short JPM MAE/RMSE only."""
    from scripts.generate_part_b_figures import write_deliverable1_part_b

    path = report_dir / "MODEL_VALIDATION_REPORT.md"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for marker in ("# Part B —", "# Part 1 —", "# Part 2 —"):
        if marker in text:
            text = text.split(marker)[0].rstrip()
    path.write_text(text + "\n\n---\n\n" + write_deliverable1_part_b(metrics), encoding="utf-8")
    print(f"Merged deliverable-1 summary into {path}")


if __name__ == "__main__":
    main()
