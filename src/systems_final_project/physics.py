"""Fitted routing model of the Point's water level (physics-informed component).

Adapted from the RoutingModel in the sibling project's model/physics.py.

Two pieces of river physics are built into the model's structure:

1. Routing (flood-wave travel). A rise at an upstream gauge reaches the Point after a
   travel delay and spread out in time. The Point's change over a horizon H is a sum of
   past changes at each upstream gauge g weighted by a response kernel k[g, u]
   (u = hours of delay, 1..MAX_LAG_H):

       h_P(t+H) - h_P(t) = sum_g sum_u k[g, u] * (h_g(min(t, t+H-u)) - h_g(t-u))

   Upstream readings after the issue time t are unknown, so they are held at their
   value at t. The kernels are shared by every horizon, constrained to be non-negative
   (more water upstream can never lower the Point) and smooth (a diffusing wave spreads
   out; it does not arrive in jagged pulses).

2. Storage / recession. Between rises the Point drains back toward the Emsworth Dam
   pool: a linear-reservoir term -r_H * (h_P(t) - centre) + offset_H, with r_H >= 0.

The fit is a bound-constrained least-squares problem solved on accumulated normal
equations over the training hours only, so it takes seconds per fold.
"""

import numpy as np
import pandas as pd
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import lsq_linear

ROUTED_GAUGES = ("natrona", "kittanning", "sutersville", "braddock")
MAX_LAG_H = 36      # longest delay a kernel can represent
SMOOTHNESS = 0.5    # weight of the second-difference penalty, relative to the data term


def _earlier(a: np.ndarray, n: int) -> np.ndarray:
    """Value n hours earlier (n >= 0) on a complete hourly grid."""
    if n == 0:
        return a
    out = np.full_like(a, np.nan)
    out[n:] = a[:-n]
    return out


def routing_design(levels: pd.DataFrame, h: int) -> np.ndarray:
    """Columns (gauge, u): h_g(min(t, t+h-u)) - h_g(t-u) for u = 1..MAX_LAG_H.
    Uses only readings at or before the issue time t."""
    cols = []
    for g in ROUTED_GAUGES:
        s = levels[g].to_numpy(dtype=float)
        for u in range(1, MAX_LAG_H + 1):
            recent = s if u <= h else _earlier(s, u - h)
            cols.append(recent - _earlier(s, u))
    return np.column_stack(cols)


class RoutingModel:
    """Non-negative, smooth upstream response kernels + recession toward the pool."""

    def fit(self, designs: dict[int, np.ndarray], point: pd.Series, targets: dict[int, pd.Series],
            train: np.ndarray) -> "RoutingModel":
        self.horizons = list(designs)
        nk, nh = len(ROUTED_GAUGES) * MAX_LAG_H, len(self.horizons)
        p = nk + 2 * nh
        now = point.to_numpy(dtype=float)
        self.centre = float(np.nanmedian(now[train]))
        AtA, Atb = np.zeros((p, p)), np.zeros(p)
        for j, h in enumerate(self.horizons):
            K = designs[h]
            y = targets[h].to_numpy(dtype=float) - now
            ok = train & np.isfinite(y) & np.isfinite(now) & np.isfinite(K).all(axis=1)
            A = np.zeros((int(ok.sum()), p))
            A[:, :nk] = K[ok]
            A[:, nk + j] = -(now[ok] - self.centre)   # recession toward the pool
            A[:, nk + nh + j] = 1.0                    # pool offset for this horizon
            AtA += A.T @ A
            Atb += A.T @ y[ok]

        # smoothness: second differences along each gauge's kernel
        D = np.diff(np.eye(MAX_LAG_H), n=2, axis=0)
        lam = SMOOTHNESS * np.mean(np.diag(AtA)[:nk])
        P = np.zeros((p, p))
        for i in range(len(ROUTED_GAUGES)):
            sl = slice(i * MAX_LAG_H, (i + 1) * MAX_LAG_H)
            P[sl, sl] = lam * D.T @ D
        M = AtA + P + 1e-9 * np.trace(AtA) / p * np.eye(p)

        L = cholesky(M, lower=True)
        rhs = solve_triangular(L, Atb, lower=True)
        lb = np.r_[np.zeros(nk), np.zeros(nh), np.full(nh, -np.inf)]   # kernels, recession >= 0
        x = lsq_linear(L.T, rhs, bounds=(lb, np.full(p, np.inf)), method="bvls", lsmr_tol="auto").x
        self.k = x[:nk].reshape(len(ROUTED_GAUGES), MAX_LAG_H)
        self.recession = dict(zip(self.horizons, x[nk:nk + nh]))
        self.offset = dict(zip(self.horizons, x[nk + nh:]))
        return self

    def predict_change(self, designs: dict[int, np.ndarray], point: pd.Series) -> dict[int, pd.Series]:
        """Predicted h_P(t+H) - h_P(t). A missing upstream reading contributes no change;
        a missing Point reading gives NaN."""
        now = point.to_numpy(dtype=float)
        return {h: pd.Series(np.nan_to_num(designs[h], nan=0.0) @ self.k.ravel()
                             - self.recession[h] * (now - self.centre) + self.offset[h], index=point.index)
                for h in self.horizons}

    def step_gain(self, h: int) -> float:
        """Rise at the Point within h hours per 1 ft rise now at every upstream gauge."""
        return float(self.k[:, :min(h, MAX_LAG_H)].sum())

    def summary(self) -> pd.DataFrame:
        """Per gauge: hour of peak response, response-weighted mean travel time, total gain."""
        lags = np.arange(1, MAX_LAG_H + 1)
        rows = []
        for g, k in zip(ROUTED_GAUGES, self.k):
            rows.append({"gauge": g, "peak_lag_h": int(lags[k.argmax()]) if k.max() > 0 else np.nan,
                         "mean_travel_h": float((lags * k).sum() / k.sum()) if k.sum() > 0 else np.nan,
                         "total_gain": float(k.sum())})
        return pd.DataFrame(rows)

    def pool_level(self, h: int) -> float:
        """Level the recession drains toward: centre + offset / recession."""
        r = self.recession[h]
        return self.centre + self.offset[h] / r if r > 0 else np.nan
