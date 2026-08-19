"""
MODEL CONFIDENCE SET (MCS) PROCEDURE

Purpose
-------
This module implements the Hansen-Lunde-Nason (2011) MCS procedure.

The MCS identifies the subset of forecasting models whose predictive
performance cannot be statistically distinguished from the best-performing
model. Unlike pairwise Diebold-Mariano tests, the MCS evaluates all candidate
models jointly and accounts for multiple model comparisons.

Hypotheses
----------

Null Hypothesis

    H0: All models have equal predictive ability.

Alternative Hypothesis

    H1: At least one model performs significantly worse.

Procedure
---------

1. Begin with all candidate models.
2. Test Equal Predictive Ability (EPA).
3. If EPA is rejected: Remove the worst-performing model.
4. Repeat until EPA can no longer be rejected.
5. Remaining models form the Model Confidence Set (MCS).

Interpretation
--------------

Models remaining in the final MCS are statistically
indistinguishable from the best model at significance level α.

Bootstrap
---------

The null distribution of T_max is approximated via a circular block
bootstrap (Politis & Romano, 1992) with block length l. Each bootstrap
t-statistic is centered by subtracting the observed mean difference,
simulating the null hypothesis of equal predictive ability.

This follows the bootstrap construction in Hansen, Lunde & Nason (2011)
Section 3.3. The block structure preserves the serial dependence present
in financial loss series.

Block Length
------------

The block length l must exceed the serial dependence in the loss series,
otherwise the bootstrap breaks dependence that is present in the data, the
null distribution of T_max comes out too narrow, and the procedure
over-rejects -- yielding confidence sets that are too small.

At horizon h the forecast target is the average realized variance over
[t+1, t+h], so consecutive targets share h-1 of their h daily terms. The
loss series therefore inherits an MA(h-1) dependence structure purely from
the overlap, on top of the volatility clustering already present at h = 1.
A single block must be long enough to contain that overlap, so l has to
grow with the horizon; see block_length_for_horizon() below.
"""

import numpy as np
import pandas as pd


# =============================================================================
# BLOCK LENGTH SELECTION
# =============================================================================

def block_length_for_horizon(horizon: int, min_length: int = 10) -> int:
    """
    Circular block length for an h-step overlapping loss series.

    Parameters
    ----------
    horizon : int
        Forecast horizon h. The target averages RV over [t+1, t+h], which
        induces MA(h-1) dependence in the loss series.
    min_length : int, default=10
        Floor on the block length. At h = 1 the overlap vanishes but the loss
        series still inherits the volatility clustering of realized variance,
        so a block of 10 trading days (~T^(1/3) for the 1,000-observation
        evaluation samples used here) is retained.

    Returns
    -------
    int
        l = max(min_length, 2h).

    Notes
    -----
    Blocks of length exactly h-1 would only just span the overlap, and the
    dependence is still severed at every block boundary. The usual remedy is
    a block a small multiple of the dependence order; 2h is used here. The
    resulting choice leaves h = 1 and h = 5 at the default l = 10 and raises
    only the monthly horizon, l = 44, where the 22-day overlap makes the
    original fixed l = 10 untenable.
    """
    return max(min_length, 2 * horizon)


# =============================================================================
# MODEL CONFIDENCE SET
# =============================================================================
class ModelConfidenceSet:

    def __init__(
            self,
            alpha: float = 0.10,
            B: int = 1000,
            block_length: int = 10,
            random_state: int = 42
    ):
        self.alpha = alpha          # Significance level
        self.B = B                  # Bootstrap replications

        # Circular block length (l). For overlapping multi-step targets this
        # must scale with the horizon -- pass block_length_for_horizon(h)
        # rather than relying on the default, which is valid only at h = 1.
        self.block_length = block_length

        self.rng = np.random.default_rng(random_state)
        self.mcs_models = None
        self.history = None

    # Pairwise t-statistic
    # -----------------------------------------------------------------------
    def _t_stat(self, d_ij: np.ndarray) -> float:
        """
        t_ij = mean(d_ij) / std(d_ij)

        where d_ij,t = L_i,t - L_j,t is the loss differential.
        """
        return float(np.mean(d_ij) / (np.std(d_ij, ddof=1) + 1e-12))

    # Maximum test statistic
    # -----------------------------------------------------------------------
    def _t_max(self, losses: pd.DataFrame) -> float:
        """
        T_max = max_{i≠j} |t_ij|

        The maximum absolute pairwise t-statistic over all model pairs
        in the current candidate set.
        """
        models = losses.columns.tolist()
        t_vals = [
            abs(self._t_stat(losses[i].values - losses[j].values))
            for i in models for j in models if i != j
        ]
        return max(t_vals)

    # Circular block bootstrap indices
    # -----------------------------------------------------------------------
    def _block_indices(self, T: int) -> np.ndarray:
        """
        Draw T indices via circular block bootstrap with block length l.

        Blocks wrap around the end of the series, preserving serial
        dependence in the loss series.
        """
        l = self.block_length
        n_blocks = int(np.ceil(T / l))
        starts = self.rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([(s + np.arange(l)) % T for s in starts])
        return idx[:T]

    # Bootstrap p-value (centered under H0)
    # -----------------------------------------------------------------------
    def _bootstrap_p_value(self, losses: pd.DataFrame) -> float:
        """
        Approximate the null distribution of T_max via circular block
        bootstrap with centering under H0.

        For each bootstrap replication b:

            d*_ij,t  = L*_i,t - L*_j,t           (bootstrap loss diff)
            t*_ij    = mean(d*_ij - d̄_ij) / std(d*_ij)

        The centering term d̄_ij = mean(d_ij) shifts the bootstrap
        distribution to have mean zero under H0, as required by
        Hansen et al. (2011) Section 3.3.

        p-value = Pr(T*_max >= T_obs)
        """
        T = len(losses)
        models = losses.columns.tolist()
        loss_arr = losses.values                            # (T, M) array
        col_map = {m: k for k, m in enumerate(models)}

        t_obs = self._t_max(losses)

        # Pre-compute observed mean loss differences (used for centering)
        pairs = [(i, j) for i in models for j in models if i != j]
        obs_means = {
            (i, j): np.mean(loss_arr[:, col_map[i]] - loss_arr[:, col_map[j]])
            for i, j in pairs
        }

        t_boot = np.empty(self.B)

        for b in range(self.B):
            idx = self._block_indices(T)
            t_vals = []

            for i, j in pairs:
                d_boot = loss_arr[idx, col_map[i]] - loss_arr[idx, col_map[j]]

                # Center under H0: subtract observed mean difference
                centered = d_boot - obs_means[(i, j)]

                t_vals.append(
                    abs(np.mean(centered) / (np.std(centered, ddof=1) + 1e-12))
                )

            t_boot[b] = max(t_vals)

        return float(np.mean(t_boot >= t_obs))

    # MCS elimination procedure
    # -----------------------------------------------------------------------
    def fit(self, losses: pd.DataFrame) -> list:
        """
        Iterative elimination algorithm.

        Begin with all models in the candidate set.

        While EPA is rejected at level alpha:
            1. Remove the worst model (highest mean loss).
            2. Re-test EPA on the remaining set.

        Stop when EPA can no longer be rejected.

        Returns
        -------
        list of model names surviving in the MCS at level alpha.
        """
        survivors = losses.columns.tolist()
        self.history = []

        while len(survivors) > 1:
            subset = losses[survivors]
            p = self._bootstrap_p_value(subset)

            self.history.append({'models': survivors.copy(), 'p_value': p})

            # Fail to reject EPA — remaining models form the MCS
            if p > self.alpha:
                break

            # Remove the worst-performing model
            worst = subset.mean().idxmax()
            survivors.remove(worst)

        self.mcs_models = survivors
        return survivors
