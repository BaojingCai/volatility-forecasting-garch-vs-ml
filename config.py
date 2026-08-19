# =============================================================================
# CONFIGURATION FILE
# =============================================================================
# Master's Thesis:
# Machine Learning versus GARCH-type Models in Volatility
# Forecasting: Evidence from the U.S. Equity Market
#
# Program:
# M.Sc. Economics (Macro & Financial Econometrics)
# Heidelberg University, Germany
#
# Purpose:
# This file stores all global parameters used throughout my thesis,
# including asset selection, forecasting horizons, rolling-window
# specifications, feature definitions, and output directories.
# Centralizing these settings ensures consistency, reproducibility,
# and ease of modification across all forecasting experiments.
# =============================================================================

# Interact with the operating system
import os 

# Assets
stocks = ["MSFT", "JNJ", "PG", "XOM", "JPM", "NVDA"]

# Reproducibility
random_seed = 42

# Forecast horizons in trading days (1-day, 1-week, 1-month)
forecast_horizons = [1, 5, 22]
max_horizon = max(forecast_horizons)

# Rolling estimation settings
rolling_window = 1000
refit_frequency = 5

# Train/test split date
split_date = '2022-01-01'

# MF2-GARCH rolling window (quarterly; paper-optimal value)
mf2_m = 63

# NOTE ON RETURNS: `ret` is the OPEN-TO-CLOSE log return, 100 * log(close_t / open_t).
# It spans the same regular trading session (09:30-16:05) over which VOLARE computes
# rv5, so GJR-GARCH / MF2-GARCH (and the MF2 block inside the Hybrid) model the
# conditional variance of the same intraday quantity that rv5 measures. Every model
# therefore shares one common return definition for the leverage term.

# Features for HAR-GARCH: daily/weekly/monthly log-RV components + ONE leverage term,
# `ret_neg` = min(ret, 0), the negative part of the open-to-close return. This is the
# LHAR leverage term of Corsi & Reno (2012) and the linear analogue of the GJR/MF2
# gamma*I{r<0} asymmetry. The raw signed `ret` is NOT used here: being linear, a single
# coefficient on it forces a +1% return to mirror a -1% return, and it comes out
# insignificant for half the stocks. Including both `ret` and `ret_neg` recovers the
# V-shape in-sample but does not improve out-of-sample forecasts, so the parsimonious
# form is kept -- it also leaves HAR with the same parameter count as the plain HAR,
# which matters for a like-for-like model comparison. Mirrors HAR_COLS in
# src/models/har_garch.py, the single source of truth actually used by the model.
har_features = ['log_rv_d', 'log_rv_w', 'log_rv_m', 'ret_neg']

# Features for XGBoost and base features for Hybrid
# rv_roll_5, rv_roll_22 (raw rolling means) excluded: monotonically related to
# log_rv_w/m — no new split information for tree models.
# rv_lag_5/22 kept as point lags (distinct from the rolling-mean log versions).
# ret added: leverage-effect direction signal that GJR/MF2-GARCH capture
# parametrically but XGBoost cannot infer from RV lags alone.
# ret_neg is deliberately NOT added here (nor to mlp_features): it is a deterministic
# function of ret and adds no information. A tree splits on ret < 0, and a ReLU unit
# builds min(ret, 0), so both models can learn the asymmetric kink themselves. Only
# the linear HAR needs it hand-coded.
xgb_features = [
    'log_rv_d', 'log_rv_w', 'log_rv_m',
    'rv_lag_5', 'rv_lag_22',
    'rv_ratio',
    'ret',
]

# Features for MLP (parsimonious neural network benchmark)
# Log-scale inputs match the target transformation (log-RV) and reduce input
# magnitude variance for gradient-based training. No raw/log redundancy.
mlp_features = [
    'log_rv_d', 'log_rv_w', 'log_rv_m',
    'ret',
]

# Create directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "results")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
FORECAST_DIR = os.path.join(OUTPUT_DIR, "forecasts")

for path in [OUTPUT_DIR, METRICS_DIR, FIGURE_DIR, TABLE_DIR, FORECAST_DIR]:
    os.makedirs(path, exist_ok=True)