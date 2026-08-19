"""
FORECAST EVALUATION METRICS

Purpose
-------
This module implements forecast evaluation metrics used to compare
the predictive performance of econometric and machine-learning
volatility forecasting models.

Standard loss functions used in the RV literature:
  - MSE   : sensitive to large errors; scale-dependent
  - QLIKE : robust to noise in the volatility proxy; scale-free ratio
"""

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Loss functions
# ------------------------------------------------------------------

def mse(y_true, y_pred) -> float:
    """
    Compute Mean Squared Error.

    Parameters
    ----------
    y_true : array-like
        Observed realized variance.

    y_pred : array-like
        Forecasted realized variance.

    Returns
    -------
    float
        Mean Squared Error.
    """
    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return float(np.mean((y_true - y_pred) ** 2))


def qlike(y_true, y_pred) -> float:
    """
    QLIKE loss: E[h/sigma² - log(h/sigma²) - 1]
    where h = realized variance proxy, sigma² = model forecast.

    Lower values indicate superior forecasting performance.

    Patton (2011) shows this loss is robust to noise in the RV proxy.
    """
    # Ensure strictly positive realized variance
    y_true = np.maximum(np.asarray(y_true, dtype=float), 1e-10)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-10)

    return float(np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1.0))


# ------------------------------------------------------------------
# Pointwise loss series (needed for DM test and MCS)
# ------------------------------------------------------------------

def mse_series(y_true, y_pred) -> np.ndarray:
    """
    Observation-level MSE losses

    Returns
    -------
    ndarray
        Loss series: (y_t − ŷ_t)^2
    """
    return (np.asarray(y_true) - np.asarray(y_pred)) ** 2


def qlike_series(y_true, y_pred) -> np.ndarray:
    """
    Returns observation-level QLIKE losses.

    Needed by the Model Confidence Set procedure, which operates on the
    observation-level loss series rather than aggregate losses.
    """
    y_true = np.maximum(np.asarray(y_true, dtype=float), 1e-10)
    y_pred = np.maximum(np.asarray(y_pred, dtype=float), 1e-10)

    return y_true / y_pred - np.log(y_true / y_pred) - 1.0

# ------------------------------------------------------------------
# Summary evaluation
# ------------------------------------------------------------------
def evaluate_model(
        y_true, 
        y_pred, 
        model: str, 
        horizon: int
) -> dict:
    '''
    Single model evaluation:

    Computes summary forecasting performance for one model
    and one forecast horizon.
    '''
    return {
        'Model': model,
        'Horizon': horizon,
        'MSE': mse(y_true, y_pred),
        'QLIKE': qlike(y_true, y_pred),
    }


def evaluate_all_horizons(
    eval_df: pd.DataFrame,
    model_name: str,
    horizons=(1, 5, 22),
) -> pd.DataFrame:
    """
    Compute MSE and QLIKE for a model across all forecast horizons.

    Parameters
    ----------
    eval_df     : DataFrame with target_rv_{h} and {model_name}_h{h} columns
    model_name  : short name, e.g. 'gjr', 'har', 'mf2', 'xgb', 'mlp', 'hybrid'
    """
    rows = []   # Store evaluation results per h

    # Evaluate each forecast horizon separately
    for h in horizons:

        y_true = eval_df[f'target_rv_{h}']
        y_pred = eval_df[f'{model_name}_h{h}']

        # Compute performance metrics for this horizon
        rows.append(evaluate_model(y_true, y_pred, model_name, h))

    # Return summary table
    return pd.DataFrame(rows)
