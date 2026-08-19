"""
XGBoost Model

Purpose 
------- 
This module implements an Extreme Gradient Boosting (XGBoost)
regression model for realized volatility forecasting. 

Unlike traditional econometric models, XGBoost does not impose
a parametric volatility structure. Instead, it learns nonlinear 
relationships and interactions directly from the data through an 
ensemble of decision trees.

Forecasting Strategy 
-------------------- 
Direct multi-horizon forecasting: 

Three separate XGBRegressor models, one per forecast horizon (h = 1, 5, 22). 
Each model directly predicts the corresponding future realized variance:

    RV_(t+h)

rather than recursively generating forecasts.

Expected Feature Set (xgb_features from config)
------------------------------------------------
    log_rv_d    log RV_t                          daily HAR component
    log_rv_w    log mean(RV_{t-4..t})             weekly HAR component
    log_rv_m    log mean(RV_{t-21..t})            monthly HAR component
    rv_lag_5    RV_{t-5}   point lag — distinct from log_rv_w (rolling mean)
    rv_lag_22   RV_{t-22}  point lag — distinct from log_rv_m (rolling mean)
    rv_ratio    rv_roll_5 / rv_roll_22             short/long volatility ratio
    ret         100 × log(C_t/O_t)               open-to-close return; leverage signal

rv_roll_5, rv_roll_22 (the raw rolling means) are excluded: monotonically
related to log_rv_w, log_rv_m — no new decision boundaries for tree splits.
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor


HORIZONS = [1, 5, 22]

# =============================================================================
# XGBOOST CLASS
# =============================================================================
class XGBoostModel:

    # Initialize XGBoost forecasting model
    # -------------------------------------------------------------------------
    def __init__(
        self,
        n_estimators: int = 300,    # Number of boosting iterations (trees)
        max_depth: int = 3,         # Maximum depth of each decision tree
        learning_rate: float = 0.05,    # Shrinkage parameter controlling the contribution of each tree
        subsample: float = 0.8,     # Fraction of obs randomly sampled for each boosting iteration
        colsample_bytree: float = 0.8,  # Fraction of features randomly sampled for each tree
        random_state: int = 42,     # Seed for reproducibility
    ):
        
        # Store model hyperparameters
        self._params = dict(
            n_estimators=n_estimators,      # Number of trees in the ensemble
            max_depth=max_depth,            # Controls tree complexity
            learning_rate=learning_rate,    # Step size used during boosting
            subsample=subsample,            # Obs subsampling
            colsample_bytree=colsample_bytree,  # Feature subsampling
            objective='reg:squarederror',   # Squared-error loss function
            random_state=random_state,
            n_jobs=-1,                      # Use all available CPU cores
        )

        # Dictionary storing one model per forecast horizon
        self.models = {}

        # Feature names used during training
        self.feature_cols = None

    # Model estimation
    # -------------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, train_df: pd.DataFrame) -> "XGBoostModel":
        """
        Fit one XGBRegressor per horizon.

        Parameters
        ----------
        X_train  : feature DataFrame (rows = training obs, cols = features)
        train_df : full training DataFrame — must contain target_rv_1/5/22

        Returns
        -------
        XGBoostModel Fitted model instance.
        """

        # Store feature ordering, ensure forecasting uses the
        # exact same variable sequence as training
        self.feature_cols = X_train.columns.tolist()

        # Estimate separate model for each horizon
        for h in HORIZONS:

            # Future realized variance target
            y = train_df[f'target_rv_{h}']

            # Create XGBoost model
            m = XGBRegressor(**self._params)

            # Train gradient boosting ensemble
            # Sequential trees are fitted to residual errors from previous trees
            m.fit(X_train, y, verbose=False)

            # Store fitted model
            self.models[h] = m

        return self

    # Forecasting
    # -------------------------------------------------------------------------

    def forecast(self, x_test: pd.DataFrame) -> dict:
        """
        Predict realized variance for each horizon.

        Parameters
        ----------
        x_test : single-row DataFrame with the same feature columns as X_train

        Returns
        -------
        dict  {1: forecast_1, 5: forecast_5, 22: forecast_22}
        """

        if not self.models:
            raise RuntimeError("Call fit() before forecast().")

        return {
            # Volatility forecasts must be non-negative. Although XGBoost can 
            # theoretically produce negative predictions, rv cannot be negative
            h: float(max(self.models[h].predict(x_test[self.feature_cols])[0], 0.0))
            for h in HORIZONS
        }

    # Diagnostics
    # -------------------------------------------------------------------------

    def feature_importance(self) -> pd.DataFrame:
        '''
        Return feature importance rankings. 
        
        Returns 
        ------- 
        pd.DataFrame 
        
            Columns: 
            horizon 
            feature 
            importance 
        
        Notes 
        ----- 
        XGBoost computes importance based on each feature's contribution 
        to reducing the objective function during tree construction. 
        '''
        if not self.models:
            raise RuntimeError("Call fit() first.")

        rows = []

        # Extract feature importance values separately for each forecast h
        for h in HORIZONS:

            imp = self.models[h].feature_importances_
            for feat, val in zip(self.feature_cols, imp):
                rows.append({'horizon': h, 'feature': feat, 'importance': val})

        # Convert to DataFrame and rank features from most important to the least
        return (
            pd.DataFrame(rows)
            .sort_values(['horizon', 'importance'], ascending=[True, False])
            .reset_index(drop=True)
        )
