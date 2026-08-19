"""
Multi-Layer Perceptron (MLP) Model

Purpose 
------- 
This module implements a Multi-Layer Perceptron (MLP)
neural network for realized volatility forecasting. 

Unlike econometric models, the MLP does not assume a predefined
functional relationship between predictors and future volatility. 
Instead, it learns nonlinear mappings directly from the data through 
multiple hidden layers of interconnected neurons.

Forecasting Strategy 
--------------------
Direct multi-step approach: three separate MLPRegressors,
one per forecast horizon (h = 1, 5, 22). 

Each network directly predicts: RV_(t+h)
thereby avoiding recursive forecasting errors.

Neural Network Architecture 
--------------------------- 
Input Layer 
↓ 
Hidden Layer 1 (64 neurons) 
↓
Hidden Layer 2 (32 neurons)
↓ 
Hidden Layer 3 (16 neurons) 
↓ 
Output Layer (1 neuron)

Activation Function 
------------------- 
ReLU: f(x) = max(0, x)

Optimization Algorithm 
---------------------- 
Adam Optimizer 

The Adam algorithm combines momentum and adaptive learning rates,
making it suitable for nonlinear forecasting problems.

Feature Scaling
---------------
Explanatory variables are standardized directly.
Target variables are log-transformed before standardization,
since log(RV) is approximately Gaussian:

x_scaled = (x - μ_x) / σ_x
y_scaled = (log(RV) - μ_log) / σ_log

Back-transformation
-------------------
    Point forecast = exp(μ̂ + σ²_h/2) using Jensen's inequality correction,
    where σ²_h is the variance of the training-sample residuals of the
    horizon-h network on the log-RV scale.

    A plain exp(μ̂) would return the conditional MEDIAN of RV, not its mean.
    Because MSE and QLIKE are both evaluated against RV in levels, and every
    other model in the comparison targets the conditional mean -- XGBoost and
    the Hybrid fit RV levels under squared-error loss, GJR- and MF2-GARCH
    forecast conditional variances, and HAR-GARCH applies the same correction
    to its log-RV regression -- omitting it would bias the MLP forecasts
    downward by a factor of exp(σ²_h/2) and handicap the model relative to the
    rest of the comparison.

Expected Feature Set (mlp_features from config)
------------------------------------------------
    log_rv_d    log RV_t                      daily HAR component
    log_rv_w    log mean(RV_{t-4..t})         weekly HAR component
    log_rv_m    log mean(RV_{t-21..t})        monthly HAR component
    ret         100 × log(C_t/O_t)            open-to-close return; leverage signal

Log-scale inputs only: consistent with the target transformation (log-RV)
and reduces input magnitude variance for gradient-based training. Mirrors
the HAR structure used by HAR-GARCH without raw/log redundancy.
"""

import numpy as np
import pandas as pd

# Feed-forward neural network regression
from sklearn.neural_network import MLPRegressor

# Feature standardization
from sklearn.preprocessing import StandardScaler


HORIZONS = [1, 5, 22]

# ============================================================================= 
# MLP MODEL Class
# =============================================================================
class MLPModel:

    # Initialize Multi-Layer Perceptron model
    # -------------------------------------------------------------------------
    def __init__(
        self,
        hidden_layer_sizes: tuple = (64, 32, 16), # Number of neurons in each hidden layer
        alpha: float = 1e-4,    # L2 reg param to control model complexity & reduces overfitting risk
        learning_rate_init: float = 0.001,  # Initial learning rate used by the Adam optimizer
        max_iter: int = 1000,   # Maximum number of training epochs
        random_state: int = 42, # Seed for reproducibility
    ):
        # Neural network hyperparameters
        self._params = dict(
            hidden_layer_sizes=hidden_layer_sizes,  # Hidden layer architecture
            activation='relu',                      # Rectified Linear Unit activation
            solver='adam',                          # Adaptive Moment Estimation optimizer
            alpha=alpha,
            learning_rate='adaptive',               # Adaptive learning-rate schedule
            learning_rate_init=learning_rate_init,  # Initial learning rate
            max_iter=max_iter,

            # Automatic validation split: Training stops if validation performance doesn't improve
            early_stopping=True,   

            
            validation_fraction=0.10,   # Fraction of observations used for validation
            n_iter_no_change=10,    # Number of epochs without improvement before stopping
            random_state=random_state,
            verbose=False,
        )

        # Dictionary storing one neural network per forecast horizon
        self.models = {}

        # Feature standardization object: shared across h
        # because explanatory variables are identical
        self.scaler_x = StandardScaler()

        # Target standardization objects: separate scaler across h
        # because target distributions differ
        self.scaler_y = {}

        # Variance of the training residuals on the log-RV scale, per horizon.
        # Supplies the Jensen correction term applied in forecast().
        self.sigma2 = {}

        # Feature names used during training
        self.feature_cols = None

    # Model estimation
    # -------------------------------------------------------------------------

    def fit(self, train_df: pd.DataFrame, feature_cols: list) -> "MLPModel":
        """
        Fit one MLPRegressor per horizon.

        Parameters
        ----------
        train_df     : DataFrame with feature_cols + target_rv_1/5/22
        feature_cols : feature column names

        Returns 
        ------- 
        MLPModel Fitted model instance
        """

        # Store feature ordering
        self.feature_cols = feature_cols

        X = train_df[feature_cols].values.astype(float)

        # Standardize explanatory variables 
        # Neural networks are highly sensitive to var scaling
        X_scaled = self.scaler_x.fit_transform(X)

        # Estimate separate model for each horizon
        for h in HORIZONS:

            # Log-transform target: log(RV) is approximately Gaussian,
            # so StandardScaler works correctly and exp() back-transform
            # is always positive — no zero-clipping issues.
            y_raw = train_df[f'target_rv_{h}'].clip(lower=1e-10).values.astype(float)
            y = np.log(y_raw).reshape(-1, 1)

            # Create horizon-specific target scaler
            sy = StandardScaler()

            # Standardize log-target
            y_scaled = sy.fit_transform(y).ravel()
            self.scaler_y[h] = sy

            # Create neural network
            m = MLPRegressor(**self._params)

            # Train neural network using backpropagation and Adam optimization.
            m.fit(X_scaled, y_scaled)

            # Residual variance on the log-RV scale, needed for the Jensen
            # correction in forecast(). Predictions are inverse-standardized
            # first so the residuals live on the same scale as log(RV).
            log_rv_hat = sy.inverse_transform(
                m.predict(X_scaled).reshape(-1, 1)
            ).ravel()
            self.sigma2[h] = float(np.var(y.ravel() - log_rv_hat, ddof=1))

            # Store fitted network
            self.models[h] = m

        return self

    # Forecasting
    # -------------------------------------------------------------------------

    def forecast(self, x_test: pd.DataFrame) -> dict:
        """
        Predict realized variance for each horizon.

        Parameters
        ----------
        x_test : single-row DataFrame with feature_cols

        Returns
        -------
        dict  {1: forecast_1, 5: forecast_5, 22: forecast_22}
        """
        if not self.models:
            raise RuntimeError("Call fit() before forecast().")

        # Standardize test observation using training-sample statistics
        X_scaled = self.scaler_x.transform(x_test[self.feature_cols].values.astype(float))

        result = {}

        for h in HORIZONS:
            # Neural network prediction on standardized log-RV scale
            y_scaled = self.models[h].predict(X_scaled)[0]

            # Inverse-standardize to recover log(RV)
            y_log = float(self.scaler_y[h].inverse_transform([[y_scaled]])[0][0])

            # Jensen's inequality: E[exp(X)] = exp(μ + σ²/2) for X ~ N(μ, σ²).
            # Without the σ²/2 term this returns the conditional median of RV
            # rather than its mean, which is what MSE and QLIKE score against.
            # Always positive, so no clipping is needed.
            result[h] = float(np.exp(y_log + self.sigma2[h] / 2))

        return result
