from __future__ import annotations

import numpy as np


def mae_rmse(y: np.ndarray, yhat: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float).ravel()
    yhat = np.asarray(yhat, dtype=float).ravel()
    e = yhat - y
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "n": int(len(y)),
    }
