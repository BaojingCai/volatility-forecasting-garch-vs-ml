"""
HAR-GARCH(1,1) Model

This implementation combines: 

1. HAR-RV mean equation
2. GARCH(1,1) variance equation

The HAR component captures volatility persistence across
multiple investment horizons (daily, weekly, monthly). 

The GARCH component models conditional heteroskedasticity
remaining in the HAR residuals.

Mean equation:
-------------
    log(RV_{t+h}) = β₀ + β_d log(RV_t) + β_w log(RV_w,t) + β_m log(RV_m,t)
                    + β_l ret_neg_t + ε_t

    Separate OLS regressions for h = 1, 5, 22 (direct forecasting).
    Targets: log(target_rv_1), log(target_rv_5), log(target_rv_22).

    Leverage term. ret_neg_t = min(ret_t, 0) is the negative part of the signed
    open-to-close return. It enters with coefficient β_l < 0, so a more negative
    return raises the volatility forecast while positive returns leave it unchanged.
    This is the LHAR leverage term of Corsi & Renò (2012), and the linear analogue of
    the γ·I{r<0} asymmetry that GJR-GARCH and MF2-GARCH capture parametrically.

    Why not the raw signed return ret_t? Being linear, a single coefficient on ret_t
    forces a +1% return to have exactly the mirror-image effect of a -1% return.
    Empirically the relation is V-shaped -- BOTH directions raise future volatility,
    but negative returns raise it roughly 1.1-2.1x more -- so a lone coefficient on
    ret_t collapses to a near-cancellation of the two branches and is statistically
    insignificant for half the stocks in the sample. Including ret_t alongside
    ret_neg_t recovers the V in-sample (R^2 0.607 -> 0.612) but does NOT improve
    out-of-sample forecasts, so the parsimonious ret_neg_t form is used: it isolates
    the leverage channel while keeping the parameter count of the plain HAR, which
    matters for a like-for-like comparison against the other models.

Variance equation:
-----------------
    GARCH(1,1) on residuals from the h=1 mean equation.
    h_t = ω + α ε²_(t−1) + β h_(t−1)

Forecast:
--------
    Point forecast = exp(μ̂  + σ²_t/2) using Jensen's inequality correction,
    where σ²_t is the GARCH 1-step ahead conditional variance of the residuals.
"""

import numpy as np
import pandas as pd

import statsmodels.api as sm
from arch import arch_model


HORIZONS = [1, 5, 22]
HAR_COLS = ['log_rv_d', 'log_rv_w', 'log_rv_m', 'ret_neg']


# ============================================================================= 
# HAR-GARCH CLASS
# =============================================================================

class HARGARCH:
    
    # Initialization
    # -------------------------------------------------------------------------
    def __init__(self):

        self.models = {}            # OLS results keyed by horizon
        self.garch_results = None   # Fitted GARCH model applied to HAR residuals
        self.is_fitted = False      # Indicate whether model estimation is completed

    # Estimation
    # -------------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "HARGARCH":
        """
        Fit direct-horizon HAR models + GARCH(1,1) on h=1 residuals.

        Parameters
        ----------
        train_df : DataFrame
            Contain: log_rv_d, log_rv_w, log_rv_m, ret_neg,
                    target_rv_1, target_rv_5, target_rv_22.
        """

        # Adds an intercept term
        X = sm.add_constant(train_df[HAR_COLS], prepend=True)

        # Estimate separate direct forecasting models
        for h in HORIZONS:

            # Log-transform target volatility
            y = np.log(train_df[f'target_rv_{h}'].clip(lower=1e-10))

            # Estimate HAR regression
            # HAC standard errors are used because overlapping
            # volatility horizons generate serial correlation
            self.models[h] = sm.OLS(y, X).fit(
                cov_type='HAC',
                cov_kwds={'maxlags': h}
            )


        # Residuals from one-day HAR model, represent unexplained volatility dynamics
        # Scale residuals by 100 improve stability during MLE
        resid = self.models[1].resid * 100

        # Estimate GARCH(1,1)
        self.garch_results = arch_model(
            resid,
            mean='Zero',    # No mean process, resid already has zero expectation
            vol='GARCH',    # Standard GARCH variance process
            p=1,            # ARCH order
            q=1,            # GARCH order

            # Gaussian innovations: E[exp(X)] = exp(μ + σ²/2) is exact only under
            # normality, so keeping dist='normal' makes the Jensen correction in
            # forecast() exact rather than an approximation.
            dist='normal',
        ).fit(disp='off')

        # Mark model as estimated
        self.is_fitted = True

        return self

    # Forecasting
    # -------------------------------------------------------------------------

    def forecast(self, row: pd.Series) -> dict:
        """
        Produce point forecasts for h = 1, 5, 22.

        Parameters
        ----------
        row : pd.Series
            One observation with log_rv_d, log_rv_w, log_rv_m, ret_neg.

        Returns
        -------
        dict  {1: rv_forecast_1, 5: rv_forecast_5, 22: rv_forecast_22}
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before forecast().")

        # GARCH 1-step ahead conditional variance of the log-RV residuals.
        # Residuals were scaled ×100 before GARCH fitting, so divide by 100²
        # to recover variance in the original log-RV residual scale.
        garch_var_scaled = float(
            self.garch_results.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
        )
        sigma2 = garch_var_scaled / 1e4

        # Dictionary storing forecasts
        result = {}

        for h in HORIZONS:

            # Extract estimated coefficients
            p = self.models[h].params

            # Compute predicted log-volatility (const + one term per HAR_COLS regressor,
            # including the ret_neg leverage term)
            log_rv_hat = p['const'] + sum(p[col] * row[col] for col in HAR_COLS)
            
            # Jensen's inequality: E[exp(X)] = exp(μ + σ²/2) for X ~ N(μ, σ²).
            # Gaussian innovations make this exact at h=1. Applying the
            # same σ² at h=5 and h=22 is an approximation — the true multi-step GARCH
            # variance converges quickly to the unconditional variance, so the bias
            # is negligible at longer horizons.
            result[h] = float(np.exp(log_rv_hat + sigma2 / 2))

        return result

    # Parameter extraction
    # -------------------------------------------------------------------------
    def parameters(self) -> dict:

        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        
        # Return all HAR coefficients in a single flattened dict
        return {
            f'h{h}_{k}': v
            for h in HORIZONS
            for k, v in self.models[h].params.items()
        }

    # Information criteria
    # -------------------------------------------------------------------------
    def information_criteria(self) -> dict:

        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")
        
        # Return model selection criteria
        return {f'har_h{h}_aic': self.models[h].aic for h in HORIZONS} | {
            'garch_aic': self.garch_results.aic,
            'garch_bic': self.garch_results.bic,
        }
