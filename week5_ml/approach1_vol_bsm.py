"""Approach 1: supervised next-step rolling vol, then BSM call on the Week 4 panel."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from chooser_pricing import black_scholes_call

from week5_ml.features import VOL_FEATURE_COLS, normalize_div_yield, panel_feature_matrix
from week5_ml.metrics import mae_rmse

try:
    from xgboost import XGBRegressor

    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


@dataclass
class VolModelSpec:
    name: str
    estimator: Any


def _annualize_daily_vol(daily_vol: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(daily_vol, dtype=float) * np.sqrt(252.0), 0.01, 2.0)


def build_vol_estimators(random_state: int = 42) -> list[VolModelSpec]:
    specs: list[VolModelSpec] = [
        VolModelSpec(
            "random_forest",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=3,
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        VolModelSpec(
            "sklearn_gbdt",
            GradientBoostingRegressor(
                random_state=random_state,
                max_depth=4,
                learning_rate=0.05,
                n_estimators=200,
            ),
        ),
    ]
    if _HAS_XGB:
        specs.append(
            VolModelSpec(
                "xgboost",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        )
    return specs


def fit_and_select_vol_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
) -> tuple[VolModelSpec, dict[str, Any]]:
    X_tr = train[VOL_FEATURE_COLS].astype(float).values
    y_tr = train[target_col].astype(float).values
    X_va = val[VOL_FEATURE_COLS].astype(float).values
    y_va = val[target_col].astype(float).values

    best: tuple[float, VolModelSpec | None, Any] = (float("inf"), None, None)
    curves: dict[str, dict[str, float]] = {}
    for spec in build_vol_estimators():
        est = spec.estimator
        est.fit(X_tr, y_tr)
        pred = est.predict(X_va)
        m = mae_rmse(y_va, pred)
        curves[spec.name] = m
        if m["MAE"] < best[0]:
            best = (m["MAE"], spec, est)

    assert best[1] is not None and best[2] is not None
    return best[1], {"val_metrics_by_model": curves, "selected": best[1].name}


def refit_on_trainval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    spec: VolModelSpec,
    target_col: str,
) -> Any:
    tv = pd.concat([train, val], axis=0).sort_values("Date").reset_index(drop=True)
    X = tv[VOL_FEATURE_COLS].astype(float).values
    y = tv[target_col].astype(float).values
    est = clone(spec.estimator)
    est.fit(X, y)
    return est


def evaluate_vol_test(estimator: Any, test: pd.DataFrame, target_col: str) -> dict[str, float]:
    X_te = test[VOL_FEATURE_COLS].astype(float).values
    y_te = test[target_col].astype(float).values
    pred = estimator.predict(X_te)
    return mae_rmse(y_te, pred)


def persistence_vol_baseline_metrics(test: pd.DataFrame, target_col: str) -> dict[str, float]:
    """Predict y_vol_next[t] with rolling_vol_21[t] (highly persistent next-day vol)."""
    y = test[target_col].astype(float).values
    pred = test["rolling_vol_21"].astype(float).values
    return mae_rmse(y, pred)


def evaluate_all_vol_models_on_test(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> dict[str, dict[str, float]]:
    """Fit each tabular vol model on train+val; report MAE/RMSE on the held-out test segment."""
    tv = pd.concat([train, val], axis=0).sort_values("Date").reset_index(drop=True)
    X_tv = tv[VOL_FEATURE_COLS].astype(float).values
    y_tv = tv[target_col].astype(float).values
    X_te = test[VOL_FEATURE_COLS].astype(float).values
    y_te = test[target_col].astype(float).values

    out: dict[str, dict[str, float]] = {
        "persistence_baseline": persistence_vol_baseline_metrics(test, target_col),
    }
    for spec in build_vol_estimators():
        est = clone(spec.estimator)
        est.fit(X_tv, y_tv)
        out[spec.name] = mae_rmse(y_te, est.predict(X_te))
    return out


def panel_bsm_from_predicted_vol(
    panel: pd.DataFrame,
    estimator: Any,
    strike: float,
) -> tuple[pd.Series, dict[str, float]]:
    """Predict daily vol (next-step model applied to same-day features as proxy), annualize, BSM."""
    Xp = panel_feature_matrix(panel, VOL_FEATURE_COLS)
    sigma_daily_hat = estimator.predict(Xp)
    sigma_hat = _annualize_daily_vol(sigma_daily_hat)
    S = panel["JPM_Close"].astype(float).values
    r = panel["r_decimal"].astype(float).values
    q = np.array([normalize_div_yield(v) for v in panel["div_yield"].astype(float).values], dtype=float)
    T = panel["T_years"].astype(float).values
    yhat = black_scholes_call(S, strike, T, r, q, sigma_hat)
    series = pd.Series(yhat, index=panel.index, name="bs_ml_vol")
    y = panel["market_price_close"].astype(float).values
    metrics = mae_rmse(y, yhat)
    return series, metrics
