"""
Chooser option (two-stage GBM Monte Carlo) and plain-vanilla Black–Scholes
European options. Used by notebooks and Week 4 baseline evaluation.

Time convention: T in years (365 calendar days ≈ 1y unless noted).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def black_scholes_call(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float | np.ndarray,
    r: float | np.ndarray,
    q: float | np.ndarray,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """Black–Scholes–Merton European call under continuous yield q."""
    T = np.asarray(T, dtype=float)
    eps = 1e-12
    T_eff = np.maximum(T, eps)
    S, K, r, q, sigma = (
        np.asarray(x, dtype=float) for x in (S, K, r, q, sigma)
    )
    sqrtT = np.sqrt(T_eff)
    d1 = (np.log(np.maximum(S, eps) / np.maximum(K, eps)) + (r - q + 0.5 * sigma**2) * T_eff) / (
        np.maximum(sigma, eps) * sqrtT
    )
    d2 = d1 - sigma * sqrtT
    return S * np.exp(-q * T_eff) * norm.cdf(d1) - K * np.exp(-r * T_eff) * norm.cdf(d2)


def chooser_option_mc(
    S: float,
    K: float,
    T1: float,
    T2: float,
    r: float,
    q: float,
    sigma: float,
    n_sims: int = 100_000,
    seed: int = 42,
) -> float:
    """
    Chooser (simple): at T1 choose European call or put with strike K, maturity T2.
    GBM under risk-neutral measure; discounted payoff at T2.
    Decision: call if S_{T1} > K else put (matches project notebooks / paper Table 3).
    """
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n_sims)
    st1 = S * np.exp((r - q - 0.5 * sigma**2) * T1 + sigma * np.sqrt(T1) * z1)
    choose_call = st1 > K
    z2 = rng.standard_normal(n_sims)
    st2 = st1 * np.exp((r - q - 0.5 * sigma**2) * (T2 - T1) + sigma * np.sqrt(T2 - T1) * z2)
    intrinsic_call = np.maximum(st2 - K, 0.0)
    intrinsic_put = np.maximum(K - st2, 0.0)
    payoff = np.where(choose_call, intrinsic_call, intrinsic_put)
    return float(np.exp(-r * T2) * payoff.mean())


def chooser_option_mc_batch(
    S: np.ndarray,
    K: float,
    T1: np.ndarray,
    T2: np.ndarray,
    r: np.ndarray,
    q: np.ndarray,
    sigma: np.ndarray,
    n_sims: int = 25_000,
    base_seed: int = 42,
) -> np.ndarray:
    """
    One independent MC per row (different seed per row for i.i.d. noise reduction in mean).
    Slower than vectorized single-path; OK for small panels (e.g. <100 rows).
    """
    out = np.empty(len(S), dtype=float)
    for i in range(len(S)):
        out[i] = chooser_option_mc(
            float(S[i]),
            float(K),
            float(T1[i]),
            float(T2[i]),
            float(r[i]),
            float(q[i]),
            float(sigma[i]),
            n_sims=n_sims,
            seed=base_seed + i,
        )
    return out
