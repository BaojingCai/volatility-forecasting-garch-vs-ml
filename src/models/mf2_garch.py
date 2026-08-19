"""
MF2-GARCH Model

Purpose
-------
This module implements the Multiplicative Factor Two-Component
GARCH (MF2-GARCH) model proposed by Conrad and Engle that
decompose total volatility into h_t and τ_t.

where

    h_t : short-run GJR-GARCH component
    τ_t : long-run volatility cycle component

Model equations
--------------
    r_t = sqrt(σ²_t) * Z_t,
    σ²_t = h_t × τ_t

Short-run (unit-variance GJR-GARCH):
    h_t = (1-φ) + (α + γ*1_{r_{t-1}<0}) * r²_{t-1}/τ_{t-1} + β*h_{t-1}
    φ = α + γ/2 + β

Long-run (MEM for squared deGARCHed returns):
    τ_t = λ0 + λ1 * V^(m)_{t-1} + λ2 * τ_{t-1}
    V^(m)_{t-1} = (1/m) * sum_{j=1}^m  r²_{t-j} / h_{t-j}

The MF2-GARCH model extends traditional GARCH models by
allowing volatility persistence to arise from multiple
time scales simultaneously.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# =============================================================================
# JIT-COMPILED VARIANCE LOOP
# =============================================================================
# The inner recursion is a sequential loop (each step depends on the previous)
# so it cannot be vectorised with NumPy. Numba compiles it to native code,
# giving ~50-100x speedup over the pure-Python version and making the rolling
# estimation feasible within minutes rather than hours.

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _var_loop(
        returns: np.ndarray,
        alpha: float, beta: float, gamma: float,
        l0: float, l1: float, l2: float,
        phi: float, m: int,
    ):
        T = len(returns)
        h      = np.ones(T)
        tau    = np.ones(T)
        sigma2 = np.ones(T)

        for t in range(1, T):
            # V^(m)_{t-1}: mean squared de-GARCHed return over the past m obs
            start = t - m if t >= m else 0
            V_m   = 0.0

            for j in range(start, t):
                V_m += returns[j] * returns[j] / h[j]

            V_m /= (t - start)

            tau_t  = l0 + l1 * V_m + l2 * tau[t - 1]
            tau[t] = tau_t if tau_t > 1e-12 else 1e-12
            I_neg = 1.0 if returns[t - 1] < 0.0 else 0.0

            h_t   = ((1.0 - phi)
                     + (alpha + gamma * I_neg) * returns[t - 1] ** 2 / tau[t - 1]
                     + beta * h[t - 1])
            
            h[t]      = h_t if h_t > 1e-12 else 1e-12
            sigma2[t] = h[t] * tau[t]

        return h, tau, sigma2

    _NUMBA = True

except ImportError:
    # Pure-Python fallback — correct but slow for large rolling windows.
    _NUMBA = False

    def _var_loop(returns, alpha, beta, gamma, l0, l1, l2, phi, m):
        T = len(returns)
        h      = np.ones(T)
        tau    = np.ones(T)
        sigma2 = np.ones(T)

        for t in range(1, T):
            start  = max(0, t - m)
            V_m    = np.mean(returns[start:t] ** 2 / h[start:t])
            tau[t] = max(l0 + l1 * V_m + l2 * tau[t - 1], 1e-12)

            I_neg  = float(returns[t - 1] < 0)

            h[t]   = max(
                (1 - phi)
                + (alpha + gamma * I_neg) * returns[t - 1] ** 2 / tau[t - 1]
                + beta * h[t - 1],
                1e-12,
            )
            sigma2[t] = h[t] * tau[t]

        return h, tau, sigma2


# =============================================================================
# MF2-GARCH CLASS
# =============================================================================
class MF2GARCH:

    # Initialization
    # ------------------------------------------------------------------------
    def __init__(
        self,
        alpha: float,   # ARCH parameter
        beta: float,    # GARCH persistence parameter
        gamma: float,   # Asymmetry (leverage) parameter
        lambda0: float, # Long-run intercept
        lambda1: float, # Long-run memory parameter
        lambda2: float, # Long-run persistence parameter
        m: int = 63,
        kappa: float = 3.0,
    ):

        # Short-run volatility parameters
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

        # Long-run volatility cycle parameters
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.m     = m
        self.kappa = kappa

        self.phi       = alpha + gamma / 2 + beta
        self.phi_kappa = (alpha + gamma / 2) * kappa + beta

        self._returns:    np.ndarray | None = None
        self._nobs:       int | None        = None
        self._loglik_val: float | None      = None

    # Estimation
    # ------------------------------------------------------------------------

    @classmethod
    def fit(
        cls,
        returns: pd.Series | np.ndarray,
        m: int = 63,
        kappa: float = 3.0,
        warm_start: np.ndarray | None = None,
    ) -> "MF2GARCH":
        """
        Estimate parameters via QMLE (Gaussian log-likelihood).

        Parameters
        ----------
        returns    : open-to-close log-returns in percentage form,
                     100 * log(close_t / open_t) - same trading session as rv5
        m          : rolling-window length for V^(m)  (paper optimal: 63)
        kappa      : E[Z^4] for forecast recursion; = 3 under QMLE
        warm_start : length-6 array [alpha, beta, gamma, l0, l1, l2] from a
                     previous fit.  When provided the optimizer starts from
                     the prior estimate (+ one cold fallback) instead of the
                     full cold-start grid, cutting computation by ~70%.

        Returns
        -------
        Fitted MF2GARCH instance.
        """

        if isinstance(returns, pd.Series):
            returns = returns.values.astype(float)
        else:
            returns = np.asarray(returns, dtype=float)

        # Hard box constraints
        bounds = [
            (1e-6, 0.5),    # alpha
            (0.5,  0.999),  # beta
            (0.0,  0.5),    # gamma
            (1e-6, 1.0),    # lambda0
            (1e-6, 0.5),    # lambda1
            (0.5,  0.999),  # lambda2
        ]

        # Theoretical stationarity restrictions
        constraints = [
            {"type": "ineq", "fun": lambda p: 1.0 - (p[0] + p[1] + p[2] / 2) - 1e-6},
            {"type": "ineq", "fun": lambda p: 1.0 - (p[4] + p[5]) - 1e-6},
            {"type": "ineq", "fun": lambda p: p[0]},
            {"type": "ineq", "fun": lambda p: p[0] + p[2]},
            {"type": "ineq", "fun": lambda p: p[1]},
        ]

        # Two best cold starts (chosen by proximity to typical equity estimates).
        # [0.05, 0.85, 0.05, 0.20, 0.05, 0.70] — best lambda0 initialisation.
        # [0.08, 0.80, 0.08, 0.10, 0.08, 0.80] — best beta / lambda2 match.
        cold_starts = [
            [0.05, 0.85, 0.05, 0.20, 0.05, 0.70],
            [0.08, 0.80, 0.08, 0.10, 0.08, 0.80],
        ]

        if warm_start is not None:
            # Clip to bounds so the warm point is always feasible
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            ws = np.clip(warm_start, lo, hi)
            p0_grid = [ws.tolist(), cold_starts[0]]
        else:
            p0_grid = cold_starts

        # Objective: Gaussian QMLE, evaluated via the JIT-compiled loop
        def neg_loglik(params):
            a, b, g, l0, l1, l2 = params
            phi    = a + g / 2 + b
            _, _, sigma2 = _var_loop(returns, a, b, g, l0, l1, l2, phi, m)
            sigma2 = np.maximum(sigma2, 1e-12)
            return 0.5 * np.sum(np.log(sigma2) + returns ** 2 / sigma2)

        best_result = None
        best_fval   = np.inf

        for p0 in p0_grid:
            res = minimize(
                neg_loglik,
                np.array(p0),
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"ftol": 1e-10, "maxiter": 500, "disp": False},
            )
            if res.fun < best_fval:
                best_fval   = res.fun
                best_result = res

        alpha, beta, gamma, l0, l1, l2 = best_result.x

        inst              = cls(alpha, beta, gamma, l0, l1, l2, m=m, kappa=kappa)
        inst._returns     = returns
        inst._nobs        = len(returns)
        inst._loglik_val  = -best_result.fun

        return inst

    # In-sample variance components
    # ------------------------------------------------------------------------

    def _compute_variance(
        self,
        returns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute h_t, τ_t, σ²_t via the JIT-compiled loop."""
        return _var_loop(
            returns,
            self.alpha, self.beta, self.gamma,
            self.lambda0, self.lambda1, self.lambda2,
            self.phi, self.m,
        )

    def compute_volatility(
        self,
        returns: pd.Series | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Public wrapper: returns (h, tau, sigma²)."""
        if isinstance(returns, pd.Series):
            returns = returns.values.astype(float)
        return self._compute_variance(returns)

    # Multi-step forecasts
    # ------------------------------------------------------------------------

    def forecast(self, horizon: int = 22) -> np.ndarray:
        """
        Compute E[sigma²_{T+s} | F_T] for s = 1, ..., horizon.
        Uses Theorem 2 (equations 11–12) from the paper.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Daily variance forecasts at steps 1 to horizon.
        """

        if self._returns is None:
            raise RuntimeError("Call fit() before forecast().")

        returns = self._returns
        h, tau, sigma2 = self._compute_variance(returns)
        alpha, beta, gamma = self.alpha, self.beta, self.gamma
        l0, l1, l2 = self.lambda0, self.lambda1, self.lambda2
        phi   = self.phi
        phi_k = self.phi_kappa
        m     = self.m

        past_V = (returns[-m:] ** 2 / h[-m:])[::-1]  # length m

        r_T, h_T, tau_T = returns[-1], h[-1], tau[-1]
        V_m_T = np.mean(returns[-m:] ** 2 / h[-m:])

        tau1 = l0 + l1 * V_m_T + l2 * tau_T
        h1   = max(
            (1 - phi)
            + (alpha + gamma * float(r_T < 0)) * r_T ** 2 / tau_T
            + beta * h_T,
            1e-12,
        )

        if horizon == 1:
            return np.array([h1 * tau1])

        E_h   = np.zeros(horizon + 1)
        E_tau = np.zeros(horizon + 1)
        E_sig = np.zeros(horizon + 1)

        E_h[1]   = h1
        E_tau[1] = tau1
        E_sig[1] = h1 * tau1

        def geo(n: int) -> float:
            if n <= 0:
                return 0.0
            if phi == 1.0:
                return float(n)
            return (1.0 - phi ** n) / (1.0 - phi)

        for s in range(2, horizon + 1):

            E_h[s] = 1.0 + phi ** (s - 1) * (h1 - 1.0)

            E_V_m = 0.0
            for j in range(1, m + 1):
                if j < s:
                    E_V_m += E_tau[s - j]
                else:
                    k = j - s
                    if k < len(past_V):
                        E_V_m += past_V[k]
            E_V_m /= m
            E_tau[s] = l0 + l1 * E_V_m + l2 * E_tau[s - 1]

            if s <= m:
                known_past = sum(
                    past_V[j - 1] for j in range(s, m + 1) if j - 1 < len(past_V)
                )
                sum_tau = 0.0
                sum_sig = 0.0

                for j in range(2, s):
                    g        = geo(j - 1)
                    sum_tau += E_tau[s - j] * g
                    sum_sig += phi ** (j - 2) * E_sig[s - j]

                E_sig[s] = (
                    (1 - phi) * E_tau[s]
                    + l0 * phi * E_h[s - 1]
                    + (l1 / m * phi_k + l2 * phi) * E_sig[s - 1]
                    + l1 * phi * E_h[s - 1] * known_past / m
                    + (1 - phi) * l1 / m * sum_tau
                    + l1 * phi_k / m * sum_sig
                )
            else:
                sum_tau = 0.0
                sum_sig = 0.0
                
                for j in range(2, m + 1):
                    g        = geo(j - 1)
                    sum_tau += E_tau[s - j] * g
                    sum_sig += phi ** (j - 2) * E_sig[s - j]

                E_sig[s] = (
                    (1 - phi) * E_tau[s]
                    + l0 * phi * E_h[s - 1]
                    + (l1 / m * phi_k + l2 * phi) * E_sig[s - 1]
                    + (1 - phi) * l1 / m * sum_tau
                    + l1 * phi_k / m * sum_sig
                )

        return E_sig[1 : horizon + 1]

    # Diagnostics
    # ------------------------------------------------------------------------

    def parameters(self) -> dict:
        return {
            "alpha":   self.alpha,
            "beta":    self.beta,
            "gamma":   self.gamma,
            "lambda0": self.lambda0,
            "lambda1": self.lambda1,
            "lambda2": self.lambda2,
            "phi":     self.phi,
            "m":       self.m,
        }

    def log_likelihood(self) -> float:

        if self._loglik_val is None:
            raise RuntimeError("Call fit() first.")
        
        return self._loglik_val

    def information_criteria(self) -> dict:

        if self._loglik_val is None:
            raise RuntimeError("Call fit() first.")
        
        k = 6
        n = self._nobs

        return {
            "AIC": -2 * self._loglik_val + 2 * k,
            "BIC": -2 * self._loglik_val + k * np.log(n),
        }
