"""
Hybrid Model: XGBoost augmented with MF2-GARCH volatility regime features.

Purpose 
------- 

This module implements a hybrid forecasting model that combines the 
MF2-GARCH volatility decomposition framework with the XGBoost
machine learning algorithm. 

The objective is to determine whether econometrically estimated 
volatility-state information can improve the forecasting 
performance of a nonlinear machine learning model.

Motivation:
----------
    Traditional ML models learn volatility dynamics from realized-volatility features.
    However, MF2-GARCH model decomposes conditional variance into a short-run (h_t) and
    long-run (tau_t) component. These components summarize the current market volatility 
    and may contain info that is not fully captured by standard realized-volatility predictors.  
    
    The hybrid model therefore augments the original feature set with MF2-GARCH regime variables.

Feature set:
-----------
    base_features:
        log_rv_d, log_rv_w, log_rv_m  HAR components in log-scale
        rv_lag_5, rv_lag_22           point RV lags (distinct from log versions)
        rv_ratio                      short/long volatility ratio
        ret                           open-to-close return; leverage-effect signal
    + mf2_h    short-run GJR-GARCH component h_t at the forecast origin
    + mf2_tau  long-run MEM cycle component τ_t at the forecast origin

    X_hybrid = [log_rv_d, log_rv_w, log_rv_m, rv_lag_5, rv_lag_22,
                rv_ratio, ret, mf2_h, mf2_tau]

h_t and τ_t are MF2-GARCH state variables, not the combined forecast.
Separating them lets XGBoost learn differential weights for the
short-run (transient shock) and long-run (structural regime) components.

Estimation:
-----------
    1. Fit MF2-GARCH on the training return series.
    2. Compute in-sample h_t and tau_t for each training obs.
    3. Fit three XGBRegressor models (one per h) on the enriched
       feature matrix: 

Forecasting:
-----------
    The last in-sample h_t and tau_t serve as regime indicators and are held
    fixed between re-estimations (every 5 days). Re-filtering them daily over the
    rolling window with the parameters held fixed was tested and shifts QLIKE only
    marginally (a few thousandths, too small to move any ranking): the tree draws
    most of its signal from the log-RV features (see Figure 12), so the MF2-GARCH
    state adds little incremental information. The frozen state is kept for
    simplicity. (Note: the standalone GJR- and MF2-GARCH forecasts DO re-filter
    daily; only this regime-feature input to the tree is held fixed between refits.)
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Import MF2-GARCH
try:
    from .mf2_garch import MF2GARCH

except ImportError:

    # Standalone execution fallback
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.models.mf2_garch import MF2GARCH


HORIZONS = [1, 5, 22]
MF2_COLS = ['mf2_h', 'mf2_tau']

# ============================================================================= 
# HYBRID MF2-GARCH + XGBOOST MODEL CLASS 
# =============================================================================
class HybridModel:

    # Initialize hybrid forecasting model
    # -------------------------------------------------------------------------
    def __init__(
        self,
        base_features: list,    
        m: int = 63,    
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        
        self.base_features = base_features              # Original ML predictors
        self.all_features = base_features + MF2_COLS    # Enriched feature set
        self.m = m                                      # MF2-GARCH long-run window length

        # XGBoost hyperparameters
        self._xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective='reg:squarederror',
            random_state=random_state,
            n_jobs=-1,
        )

        # One XGBoost model per horizon
        self.models = {}

        # Fitted MF2-GARCH model
        self.mf2 = None

        # Last in-sample MF2-GARCH state (short-run h_t, long-run tau_t); held
        # fixed between re-estimations and used as regime features at forecast time
        self._last_h = 1.0
        self._last_tau = 1.0

    # Model estimation
    # -------------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "HybridModel":
        """
        Estimate MF2-GARCH and XGBoost models

        Parameters
        ----------
        train_df : DataFrame
            Must contain: 'ret' (returns), base_features columns,
                          target_rv_1, target_rv_5, target_rv_22.
        """

        # Extract training returns
        returns = train_df['ret'].values.astype(float)

        # Estimate MF2-GARCH
        self.mf2 = MF2GARCH.fit(returns, m=self.m)

        # Compute volatility decomposition
        h_t, tau_t, _ = self.mf2._compute_variance(returns)

        # Store most recent volatility regime
        self._last_h = float(h_t[-1])
        self._last_tau = float(tau_t[-1])

        # Build enriched feature matrix
        X = train_df[self.base_features].copy()
        X = X.reset_index(drop=True)
        X['mf2_h'] = h_t
        X['mf2_tau'] = tau_t

        # Estimate one XGBoost model per forecast horizon
        for h in HORIZONS:
            y = train_df[f'target_rv_{h}'].reset_index(drop=True)
            model = XGBRegressor(**self._xgb_params)
            model.fit(X[self.all_features], y, verbose=False)
            self.models[h] = model

        return self

    # Forecasting
    # -------------------------------------------------------------------------

    def forecast(self, row: pd.Series) -> dict:
        """
        Generate realized volatility forecasts

        Parameters
        ----------
        row : one test obs (Series) with base_features

        Returns
        -------
        dict  {1: forecast_1, 5: forecast_5, 22: forecast_22}
        """
        if not self.models:
            raise RuntimeError("Call fit() before forecast().")

        # Construct hybrid feature vector
        x = pd.DataFrame([{
            **{f: row[f] for f in self.base_features},
            'mf2_h': self._last_h,
            'mf2_tau': self._last_tau,
        }])

        # Generate horizon-specific forecasts
        return {
            h: float(max(self.models[h].predict(x[self.all_features])[0], 0.0))
            for h in HORIZONS
        }
