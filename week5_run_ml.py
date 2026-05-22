#!/usr/bin/env python3
# ============================================================
# Week 5 — ML model pipelines (two approaches)
#
# Approach 1: supervised next-step rolling vol → annualize → BSM call vs panel market.
# Approach 2: end-to-end supervised call price (teacher = BSM on dense calendar),
#             then panel market check (out-of-sample dates vs cleaned training range).
#
# Splits: chronological 70% / 15% / 15% on each modeling frame (no shuffle).
#
# Run from repository root:
#   python3 week5_run_ml.py
# Uses repo-root chooser_pricing.py, week4_baseline_evaluation.py, week5_ml/features.py & splits.py,
# and data/*.csv / config.json alongside this file.
# ============================================================

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from week5_ml.approach1_vol_bsm import (
    build_vol_estimators,
    evaluate_all_vol_models_on_test,
    evaluate_vol_test,
    fit_and_select_vol_model,
    panel_bsm_from_predicted_vol,
    refit_on_trainval,
)
from week5_ml.approach2_supervised import (
    build_price_estimators,
    evaluate_all_price_models_on_test,
    evaluate_price_test,
    fit_and_select_price_model,
    panel_price_predictions,
    refit_price_on_trainval,
)
from week5_ml.features import (
    PRICE_FEATURE_COLS,
    build_daily_teacher_bsm_frame,
    build_volatility_supervised_frame,
    enrich_panel_for_week5,
    load_cleaned,
)
from week5_ml.lstm_vol import run_lstm_vol
from week5_ml.metrics import mae_rmse
from week5_ml.splits import split_by_date


def _load_panel(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, parse_dates=["Date", "expiration"])
    p["Date"] = p["Date"].dt.normalize()
    return p


def _panel_only_market_supervised(panel: pd.DataFrame) -> dict:
    """23-row panel: chronological split + tiny supervised baseline (documentation)."""
    tr_m, va_m, te_m, split_meta_m = split_by_date(panel, "Date")
    if len(te_m) < 2:
        return {"skipped": True, "reason": "test fold too small"}
    X_tr = tr_m[PRICE_FEATURE_COLS].astype(float).values
    y_tr = tr_m["market_price_close"].astype(float).values
    X_va = va_m[PRICE_FEATURE_COLS].astype(float).values
    y_va = va_m["market_price_close"].astype(float).values
    X_te = te_m[PRICE_FEATURE_COLS].astype(float).values
    y_te = te_m["market_price_close"].astype(float).values
    best_mae = float("inf")
    best_spec = None
    val_curves: dict[str, dict[str, float]] = {}
    for spec in build_price_estimators():
        if "linear" in spec.name and (
            pd.isna(tr_m[PRICE_FEATURE_COLS]).any().any()
            or pd.isna(va_m[PRICE_FEATURE_COLS]).any().any()
        ):
            continue
        est = spec.estimator
        est.fit(X_tr, y_tr)
        pred_va = est.predict(X_va)
        mm = mae_rmse(y_va, pred_va)
        val_curves[spec.name] = mm
        if mm["MAE"] < best_mae:
            best_mae = mm["MAE"]
            best_spec = spec
    if best_spec is None:
        return {"skipped": True, "reason": "no model could fit panel features (check enrich_panel_for_week5)"}
    tv_X = pd.concat([tr_m, va_m], axis=0).sort_values("Date")[PRICE_FEATURE_COLS].astype(float).values
    tv_y = pd.concat([tr_m, va_m], axis=0).sort_values("Date")["market_price_close"].astype(float).values
    from sklearn.base import clone

    final = clone(best_spec.estimator)
    final.fit(tv_X, tv_y)
    pred_te = final.predict(X_te)
    return {
        "note": "Very small sample; illustrative only.",
        "split": split_meta_m,
        "selected": best_spec.name,
        "val_metrics_by_model": val_curves,
        "test_metrics_vs_market": mae_rmse(y_te, pred_te),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 5 ML pipelines")
    ap.add_argument("--cleaned", type=Path, default=ROOT / "data" / "cleaned_dataset.csv")
    ap.add_argument("--panel", type=Path, default=ROOT / "data" / "week4_evaluation_panel.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "week5_metrics.json")
    ap.add_argument("--lstm-epochs", type=int, default=45)
    ap.add_argument("--no-lstm", action="store_true", help="Skip optional TensorFlow LSTM vol run.")
    args = ap.parse_args()

    cleaned = load_cleaned(args.cleaned)
    panel = enrich_panel_for_week5(_load_panel(args.panel), args.cleaned)
    strike = float(panel["strike"].iloc[0])
    expiration = pd.Timestamp(panel["expiration"].iloc[0]).normalize()

    # ----- Approach 1: volatility -----
    vol_frame, vol_target, _vol_cols = build_volatility_supervised_frame(cleaned)
    tr_v, va_v, te_v, split_meta_v = split_by_date(vol_frame, "Date")
    spec_v, sel_v = fit_and_select_vol_model(tr_v, va_v, vol_target)
    est_v = refit_on_trainval(tr_v, va_v, spec_v, vol_target)
    vol_test_by_model = evaluate_all_vol_models_on_test(tr_v, va_v, te_v, vol_target)
    vol_test_metrics = vol_test_by_model[sel_v["selected"]]
    bs_series, panel_vol_metrics = panel_bsm_from_predicted_vol(panel, est_v, strike)

    approach1 = {
        "volatility_target": "next_row_rolling_vol_21",
        "split": split_meta_v,
        "model_selection": sel_v,
        "test_metrics_by_model": vol_test_by_model,
        "rolling_vol_test_metrics": vol_test_metrics,
        "panel_bs_vs_market_using_ml_vol": panel_vol_metrics,
        "models_tried_vol": [s.name for s in build_vol_estimators()],
        "xgboost_available": any(s.name == "xgboost" for s in build_vol_estimators()),
    }

    if not args.no_lstm:
        lstm_block = run_lstm_vol(
            vol_frame,
            vol_target,
            split_meta_v["train_date_end"],
            split_meta_v["val_date_end"],
            epochs=args.lstm_epochs,
        )
    else:
        lstm_block = {"skipped": True, "reason": "disabled via --no-lstm"}

    if lstm_block.get("skipped"):
        approach1["lstm_vol"] = lstm_block
    else:
        approach1["lstm_vol_test"] = lstm_block
        vol_test_by_model["lstm"] = lstm_block["test_metrics_vol_scale"]
        if lstm_block.get("persistence_baseline_same_test_dates"):
            vol_test_by_model["persistence_baseline"] = lstm_block["persistence_baseline_same_test_dates"]
        approach1["test_metrics_by_model"] = vol_test_by_model
        approach1["models_tried_vol"].append("lstm")

    # ----- Approach 2: end-to-end price (teacher BSM on daily grid) -----
    px_frame, px_target, _px_cols = build_daily_teacher_bsm_frame(cleaned, strike, expiration)
    tr_p, va_p, te_p, split_meta_p = split_by_date(px_frame, "Date")
    spec_p, sel_p = fit_and_select_price_model(tr_p, va_p, px_target)
    est_p = refit_price_on_trainval(tr_p, va_p, spec_p, px_target)
    price_test_by_model = evaluate_all_price_models_on_test(tr_p, va_p, te_p, px_target)
    price_test_metrics = price_test_by_model[sel_p["selected"]]
    yhat_panel, panel_price_metrics = panel_price_predictions(panel, est_p)

    approach2 = {
        "teacher_target": "bsm_european_call_same_day_realized_vol",
        "strike": strike,
        "expiration": str(expiration.date()),
        "split": split_meta_p,
        "model_selection": sel_p,
        "test_metrics_by_model": price_test_by_model,
        "daily_teacher_test_metrics": price_test_metrics,
        "panel_direct_price_vs_market": panel_price_metrics,
        "panel_only_market_supervised": _panel_only_market_supervised(panel),
        "models_tried_price": [s.name for s in build_price_estimators()],
    }

    out = {
        "approach1_vol_then_bsm": approach1,
        "approach2_end_to_end_price": approach2,
        "notes": [
            "Panel dates are after cleaned_dataset; panel metrics are out-of-sample relative to training calendar.",
            "Approach 2 trains on BSM teacher prices on a dense calendar without same-day realized vol as an input feature.",
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    panel_out = args.out.parent / "week5_panel_predictions.csv"
    panel2 = panel.copy()
    panel2["approach1_bs_ml_vol"] = bs_series.values
    panel2["approach2_price_hat"] = yhat_panel
    panel2.to_csv(panel_out, index=False)

    print(json.dumps({"wrote_metrics": str(args.out), "wrote_panel": str(panel_out)}, indent=2))
    print("Vol test:", vol_test_metrics)
    print("Panel ML vol→BSM:", panel_vol_metrics)
    print("Price teacher test:", price_test_metrics)
    print("Panel end-to-end price:", panel_price_metrics)


if __name__ == "__main__":
    main()
