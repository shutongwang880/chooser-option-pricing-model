"""Optional LSTM volatility forecaster (TensorFlow)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from week5_ml.features import VOL_FEATURE_COLS
from week5_ml.metrics import mae_rmse


def _build_sequences(
    Xs: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """
    One sample per index i: features on days [i - seq_len + 1, …, i] (inclusive),
    label y[i] = next-day rolling vol — aligned with tabular models at row i.
    """
    Xseq: list[np.ndarray] = []
    yseq: list[float] = []
    ends: list = []
    start_i = seq_len - 1
    for i in range(start_i, len(Xs)):
        Xseq.append(Xs[i - seq_len + 1 : i + 1])
        yseq.append(float(y[i]))
        ends.append(dates[i])
    return (
        np.asarray(Xseq, dtype=np.float32),
        np.asarray(yseq, dtype=np.float32),
        pd.to_datetime(ends),
    )


def run_lstm_vol(
    full: pd.DataFrame,
    target_col: str,
    train_date_end: pd.Timestamp,
    val_date_end: pd.Timestamp,
    seq_len: int = 20,
    epochs: int = 80,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Sequence LSTM over scaled features; target is next-day rolling_vol_21.

    Fixes vs earlier version:
    - Feature window includes day t (same information timing as RF/GBDT at row t).
    - StandardScaler fit on training rows only (no test leakage).
    - Target scaled on train only; metrics computed after inverse_transform.
    - Positive volatility via Softplus output head.
    - Early stopping on validation loss.
    """
    try:
        from tensorflow import keras
    except ImportError:
        return {"skipped": True, "reason": "tensorflow not installed"}

    df = full.sort_values("Date").reset_index(drop=True)
    X = df[VOL_FEATURE_COLS].astype(float).values
    y = df[target_col].astype(float).values
    dates = pd.to_datetime(df["Date"]).values

    t_end = pd.Timestamp(train_date_end).normalize()
    v_end = pd.Timestamp(val_date_end).normalize()
    train_rows = pd.to_datetime(df["Date"]).dt.normalize() <= t_end

    x_scaler = StandardScaler()
    x_scaler.fit(X[train_rows.to_numpy()])
    Xs = x_scaler.transform(X)

    y_scaler = StandardScaler()
    y_scaler.fit(y[train_rows.to_numpy()].reshape(-1, 1))

    Xseq_arr, yseq_arr, dend_ser = _build_sequences(Xs, y, dates, seq_len)
    yseq_scaled = y_scaler.transform(yseq_arr.reshape(-1, 1)).ravel().astype(np.float32)

    tr_m = dend_ser <= t_end
    va_m = (dend_ser > t_end) & (dend_ser <= v_end)
    te_m = dend_ser > v_end

    if int(tr_m.sum()) < 64 or int(va_m.sum()) < 16 or int(te_m.sum()) < 8:
        return {"skipped": True, "reason": "insufficient LSTM sequence rows in a split"}

    keras.utils.set_random_seed(random_state)
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(seq_len, X.shape[1])),
            keras.layers.LSTM(32),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="softplus"),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            verbose=0,
        )
    ]
    history = model.fit(
        Xseq_arr[tr_m],
        yseq_scaled[tr_m],
        validation_data=(Xseq_arr[va_m], yseq_scaled[va_m]),
        epochs=epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )

    pred_scaled = model.predict(Xseq_arr[te_m], verbose=0).ravel()
    pred_te = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    y_te = yseq_arr[te_m]
    metrics = mae_rmse(y_te, pred_te)

    # Sanity: persistence using last day rolling_vol in the window (row i)
    rv = df["rolling_vol_21"].astype(float).values
    te_mask = np.asarray(te_m, dtype=bool)
    persist = np.array([rv[i] for i in range(seq_len - 1, len(df))], dtype=float)[te_mask]
    persist_metrics = mae_rmse(y_te, persist)

    return {
        "model": "lstm_vol",
        "seq_len": seq_len,
        "epochs_requested": epochs,
        "epochs_fitted": int(len(history.history.get("loss", []))),
        "test_metrics_vol_scale": metrics,
        "persistence_baseline_same_test_dates": persist_metrics,
        "n_train_seq": int(tr_m.sum()),
        "n_val_seq": int(va_m.sum()),
        "n_test_seq": int(te_m.sum()),
        "notes": [
            "Feature window ends at day t (inclusive), label is y_vol_next at t.",
            "X/y scalers fit on training calendar rows only; Softplus keeps vol predictions non-negative.",
        ],
    }
