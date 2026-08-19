"""
GJR-GARCH(1,1) MODEL

Purpose
-------
Implementation of the Glosten-Jagannathan-Runkle GARCH (GJR-GARCH)
model used as a benchmark econometric volatility forecasting model.

The model extends the standard GARCH framework by allowing negative
return shocks to have a different impact on future volatility than
positive shocks of equal magnitude. This asymmetry captures the
leverage effect commonly observed in financial markets.

Model Specification
-------------------
Return Equation:
    r_t = ε_t 
    ε_t = σ_t z_t

where
    z_t ~ t(0,1)

Conditional Variance Equation:
    σ²_t = ω
          + α ε²_(t−1)
          + γ I_(ε_(t−1)<0) ε²_(t−1)
          + β σ²_(t−1)

where
    I_(ε_(t−1)<0) = 1  if ε_(t−1) < 0
                    0  otherwise

Return Series
-------------
r_t is the OPEN-TO-CLOSE log return (x100), i.e. 100 * log(close_t / open_t). It
covers the same regular trading session as the 5-minute realized variance (rv5)
used as the forecast target, so h_t is the conditional variance of the intraday
move that rv5 measures. Fitting on close-to-close returns would instead make h_t
the variance of the overnight-plus-intraday move, leaving the forecasts on a
systematically larger scale than rv5.

Interpretation
--------------
ω : Long-run variance level
α : Symmetric response to shocks
γ : Asymmetric (leverage) effect
β : Volatility persistence
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

# =============================================================================
# GJR-GARCH CLASS
# =============================================================================

class GJRGARCH:
    """
    Wrapper around the ARCH package implementation of a GJR-GARCH(1,1) volatility model.
    
    Attributes
    ----------
    model :
        Model specification created by arch_model().
    
    results :
        Fitted model results after estimation.
    """

    # Initialize an empty GJR-GARCH model object
    # -------------------------------------------------------------------------
    def __init__(self):

        # Attributes to store the model and its results
        self.model = None
        self.results = None

    # Estimate a GJR-GARCH(1,1) model
    # -------------------------------------------------------------------------
    def fit(self, returns: pd.Series):
   
        # Create the model specification
        self.model = arch_model(
            returns,            # Model input
            vol='GARCH',        # GARCH model to use
            p=1,                # ARCH lag order
            o=1,                # Asymmetric shock lag order
            q=1,                # GARCH lag order
            mean='Zero',        # Zero-mean return process
            dist='t'            # Student-t innovations
        )

        # Maximum likelihood estimation
        self.results = self.model.fit(disp='off', show_warning=False)

        return self

    # In-sample conditional volatility
    # -------------------------------------------------------------------------
    def conditional_volatility(self) -> pd.Series | np.ndarray:
        """
        Return estimated conditional volatility.
        
        Returns
        -------
        pd.Series
            σ_t = √h_t 
        
        where h_t denotes the conditional variance.
        """

        if self.results is None:
            raise RuntimeError("Model must be fitted before accessing volatility.")
        
        return self.results.conditional_volatility

    # In-sample conditional variance
    # -------------------------------------------------------------------------
    def conditional_variance(self) -> pd.Series | np.ndarray:
        """ 
        Return estimated conditional variance.
        
        Returns
        -------
        pd.Series
            h_t = σ²_t    
        """
        if self.results is None:
            raise RuntimeError(
                "Model must be fitted before accessing variance.")

        return self.results.conditional_volatility ** 2
        
    # Volatility forecasts
    # -------------------------------------------------------------------------
    def forecast(self, horizon: int=1) -> pd.Series:
        """
        Generate multi-step-ahead variance forecasts.

        Parameters
        ----------
        horizon : int, default=1
            Forecast horizon measured in trading days.

        Returns
        -------
        pd.Series
            Forecasted conditional variances
                [h_(t+1),
                h_(t+2)
                ...
                h_(t+h)]
        
        Notes
        -----
        The ARCH package returns a forecast matrix. The final row corresponds to the
        most recent forecast origin and is therefore extracted.
        """

        if self.results is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        forecasts = self.results.forecast(
            horizon=horizon,         
            reindex=False     
        )

        variance = forecasts.variance.iloc[-1]

        return variance

    # Fixed-parameter forecast (origin advanced without re-estimation)
    # -------------------------------------------------------------------------
    def forecast_fixed(self, returns: pd.Series, horizon: int = 1) -> pd.Series:
        """
        Forecast conditional variance from the end of ``returns`` while holding
        the estimated parameters fixed (no re-estimation).

        Rationale
        ---------
        Calling ``forecast()`` on the stored results object always forecasts
        from the endpoint of the estimation sample. Under periodic re-estimation 
        that endpoint is stale between refits, so the same forecast would be re-issued 
        for several days. This method instead re-runs the GJR-GARCH filter — with the
        parameters pinned at their estimated values via ``arch``'s ``fix()`` —
        over ``returns``, advancing the forecast origin one day at a time. This
        matches the daily-updated forecast origin of the HAR and ML models and
        keeps the cross-model comparison apples-to-apples.

        Parameters
        ----------
        returns : pd.Series
            Return series ending at the desired forecast origin (the current
            rolling window). Same scale/type as passed to ``fit()``.
        horizon : int, default=1
            Forecast horizon in trading days.

        Returns
        -------
        pd.Series
            Forecasted conditional variances [h_(t+1), ..., h_(t+horizon)].
        """
        if self.results is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        # Rebuild the identical GJR-GARCH(1,1) specification and pin the
        # parameters at their fitted values (no MLE), then filter over `returns`.
        fixed_res = arch_model(
            returns, vol='GARCH', p=1, o=1, q=1, mean='Zero', dist='t',
        ).fix(self.results.params)

        forecasts = fixed_res.forecast(horizon=horizon, reindex=False)

        return forecasts.variance.iloc[-1]

    # Parameter estimates
    # -------------------------------------------------------------------------
    def parameters(self) -> pd.Series:
        """
        Return estimated model parameters.

        Returns
        -------
        pd.Series

        Contains estimates for: 

        ω : constant 
        α : ARCH effect
        γ : leverage effect
        β : GARCH persistence
        ν : Student-t degrees of freedom
        """

        if self.results is None:
            raise RuntimeError("Model must be fitted before getting parameters.")

        return self.results.params
    
    # Information criteria
    # -------------------------------------------------------------------------
    def information_criteria(self) -> dict:
        """
        Return model selection criteria.
        
        Returns
        -------
        dict
        
        AIC : Akaike Information Criterion
        BIC : Bayesian Information Criterion
        """

        if self.results is None:
            raise RuntimeError("Model must be fitted before getting information criteria.")
        
        return {'AIC': self.results.aic,
                'BIC': self.results.bic}
    
    # Standardized residuals
    # -------------------------------------------------------------------------
    def standardized_residuals(self) -> pd.Series | np.ndarray:
        """
        Return standardized residuals.
        
        Returns
        -------
        pd.Series
            z_t = ε_t / σ_t
        
        Notes
        -----
        If the volatility model is correctly specified, the standardized residuals
        should behave approximately as an independent and identically distributed process.
        """

        if self.results is None:
            raise RuntimeError( "Model must be fitted before accessing residuals.")

        return self.results.std_resid   

    # Model summary
    # -------------------------------------------------------------------------

    def summary(self):
        """
        Return complete estimation summary.
        
        Includes --------
        • Parameter estimates
        • Standard errors
        • t-statistics
        • p-values
        • Log-likelihood
        • AIC
        • BIC
        """

        if self.results is None:
            raise RuntimeError("Model must be fitted before accessing summary.")

        return self.results.summary()