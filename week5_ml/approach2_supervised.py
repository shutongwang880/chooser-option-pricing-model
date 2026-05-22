"""Approach 2: end-to-end supervised option price (linear / GBDT / NN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from week5_ml.features import PRICE_FEATURE_COLS, panel_feature_matrix
from week5_ml.metrics import mae_rmse


@dataclass
class PriceModelSpec:
    name: str
    estimator: Any


def build_price_estimators(random_state: int = 42) -> list[PriceModelSpec]:
    return [
        PriceModelSpec(
            "linear_regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LinearRegression()),
                ]
            ),
        ),
        PriceModelSpec(
            "hist_gbdt",
            HistGradientBoostingRegressor(
                max_depth=6,
                learning_rate=0.06,
                max_iter=250,
                random_state=random_state,
                early_stopping=True,
                validation_fraction=0.12,
                n_iter_no_change=15,
            ),
        ),
        PriceModelSpec(
            "mlp_nn",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "mlp",
                        MLPRegressor(
                            hidden_layer_sizes=(48, 24),
                            activation="relu",
                            max_iter=600,
                            random_state=random_state,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=30,
                        ),
                    ),
                ]
            ),
        ),
    ]


def fit_and_select_price_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    target_col: str,
) -> tuple[PriceModelSpec, dict[str, Any]]:
    X_tr = train[PRICE_FEATURE_COLS].astype(float).values
    y_tr = train[target_col].astype(float).values
    X_va = val[PRICE_FEATURE_COLS].astype(float).values
    y_va = val[target_col].astype(float).values

    best = (float("inf"), None, None)
    curves: dict[str, dict[str, float]] = {}
    for spec in build_price_estimators():
        est = spec.estimator
        est.fit(X_tr, y_tr)
        pred = est.predict(X_va)
        m = mae_rmse(y_va, pred)
        curves[spec.name] = m
        if m["MAE"] < best[0]:
            best = (m["MAE"], spec, est)

    assert best[1] is not None and best[2] is not None
    return best[1], {"val_metrics_by_model": curves, "selected": best[1].name}


def refit_price_on_trainval(
    train: pd.DataFrame,
    val: pd.DataFrame,
    spec: PriceModelSpec,
    target_col: str,
) -> Any:
    tv = pd.concat([train, val], axis=0).sort_values("Date").reset_index(drop=True)
    X = tv[PRICE_FEATURE_COLS].astype(float).values
    y = tv[target_col].astype(float).values
    est = clone(spec.estimator)
    est.fit(X, y)
    return est


def evaluate_price_test(estimator: Any, test: pd.DataFrame, target_col: str) -> dict[str, float]:
    X_te = test[PRICE_FEATURE_COLS].astype(float).values
    y_te = test[target_col].astype(float).values
    pred = estimator.predict(X_te)
    return mae_rmse(y_te, pred)


def evaluate_all_price_models_on_test(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_col: str,
) -> dict[str, dict[str, float]]:
    """Fit each price model on train+val; report MAE/RMSE vs BSM teacher on test."""
    tv = pd.concat([train, val], axis=0).sort_values("Date").reset_index(drop=True)
    X_tv = tv[PRICE_FEATURE_COLS].astype(float).values
    y_tv = tv[target_col].astype(float).values
    X_te = test[PRICE_FEATURE_COLS].astype(float).values
    y_te = test[target_col].astype(float).values

    out: dict[str, dict[str, float]] = {}
    for spec in build_price_estimators():
        est = clone(spec.estimator)
        est.fit(X_tv, y_tv)
        out[spec.name] = mae_rmse(y_te, est.predict(X_te))
    return out


def panel_price_predictions(panel: pd.DataFrame, estimator: Any) -> tuple[np.ndarray, dict[str, float]]:
    Xp = panel_feature_matrix(panel, PRICE_FEATURE_COLS)
    yhat = estimator.predict(Xp)
    y = panel["market_price_close"].astype(float).values
    return yhat, mae_rmse(y, yhat)
