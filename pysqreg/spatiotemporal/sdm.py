"""DynQuantSDM -- Dynamic Spatial Durbin Quantile Regression for panel data."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse

from ..areal._base import (
    ArrayLike,
    WeightMatrix,
    prepare_w,
    prepare_x,
    prepare_y,
    solve_spatial_multiplier,
    validate_inputs,
)
from ._base import BaseSTQuantReg
from ._panel import (
    PanelStructure,
    build_spatial_lag_panel,
    build_spatiotemporal_lag,
    build_temporal_lag,
    subset_to_valid,
)
from .sar import DynQuantSAR


class DynQuantSDM(BaseSTQuantReg):
    """Dynamic Spatial Durbin Quantile Regression for panel data.

    Estimates the full model::

        y_{it} = rho W y_{it} + X_{it} beta + W X_{it} theta
                 + gamma y_{i,t-1} + delta W y_{i,t-1}
                 + alpha_i + u_{it},   Q_tau(u | X) = 0

    This nests DynQuantSAR (theta=0, delta=0) and DynQuantSLX
    (rho=0, delta=0) as special cases.

    Internally delegates to DynQuantSAR with an augmented design
    matrix ``[X, WX, Wy_{t-1}]``, analogous to how the cross-
    sectional QuantSDM delegates to QuantSAR.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    method : {'two_stage', 'grid_search'}
        Estimation method.
    fixed_effects : {'penalized', 'dummies', 'none'}
        Strategy for unit fixed effects.
    lam : float or None
        Penalty for penalized FE (None = auto-select via BIC).
    inference : {'bootstrap', 'analytical'} or None
        Inference strategy.
    nboot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    rhomat : array-like or None
        Grid of candidate rho values (grid_search only).
    max_temporal_lag : int
        Deepest temporal lag used as instrument (default 3).
    verbose : int
        Verbosity level.
    random_state : int or None
        Random seed.

    Attributes
    ----------
    coef_ : ndarray
        Coefficients for X variables (beta).
    theta_ : ndarray
        Coefficients for WX variables (theta).
    intercept_ : float
        Intercept.
    rho_ : float
        Spatial autoregressive parameter.
    gamma_ : float
        Temporal autoregressive parameter.
    delta_ : float
        Spatio-temporal lag parameter (coef on Wy_{t-1}).
    alpha_i_ : ndarray or None
        Unit fixed effects.
    se_ : ndarray
        Standard errors.
    pvalues_ : ndarray
        Two-sided p-values.
    results_ : DataFrame
        Full results table.
    impacts_ : DataFrame
        Direct, indirect, and total spatial impacts.
    panel_ : PanelStructure
        Panel structure metadata.

    Examples
    --------
    >>> model = DynQuantSDM(tau=0.5, method='two_stage')
    >>> model.fit(X, y, W, n_units=50, n_periods=10)
    >>> model.summary()
    """

    _param_names = (
        "tau", "method", "fixed_effects", "lam", "inference", "nboot",
        "alpha", "rhomat", "max_temporal_lag", "verbose", "random_state",
    )

    def __init__(
        self,
        tau: float = 0.5,
        method: str = "two_stage",
        fixed_effects: str = "penalized",
        lam: float | None = None,
        inference: str | None = None,
        nboot: int = 100,
        alpha: float = 0.05,
        rhomat: npt.ArrayLike | None = None,
        max_temporal_lag: int = 3,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1 (exclusive)")
        if method not in ("two_stage", "grid_search"):
            raise ValueError(
                f"method must be 'two_stage' or 'grid_search', "
                f"got '{method}'."
            )
        if fixed_effects not in ("penalized", "dummies", "none"):
            raise ValueError(
                f"fixed_effects must be 'penalized', 'dummies', or 'none', "
                f"got '{fixed_effects}'."
            )
        if inference is not None and inference not in ("bootstrap", "analytical"):
            raise ValueError(
                f"inference must be 'bootstrap', 'analytical', or None, "
                f"got '{inference}'."
            )
        self.tau = tau
        self.method = method
        self.fixed_effects = fixed_effects
        self.lam = lam
        self.inference = inference
        self.nboot = nboot
        self.alpha = alpha
        self.rhomat = rhomat
        self.max_temporal_lag = max_temporal_lag
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        W: WeightMatrix,
        n_units: int | None = None,
        n_periods: int | None = None,
        unit_ids: ArrayLike | None = None,
        time_ids: ArrayLike | None = None,
    ) -> DynQuantSDM:
        """Fit the dynamic Spatial Durbin quantile regression model.

        Augments X with ``[WX, Wy_{t-1}]`` and delegates to
        :class:`DynQuantSAR` for estimation.

        Parameters
        ----------
        X : array-like of shape (N*T, k)
            Explanatory variables **without** intercept.
        y : array-like of shape (N*T,)
            Dependent variable.
        W : array-like or sparse of shape (N, N)
            Row-standardized spatial weight matrix.
        n_units, n_periods : int, optional
            Panel dimensions.
        unit_ids, time_ids : array-like, optional
            Alternative panel identifiers.

        Returns
        -------
        self
        """
        # --- Prepare inputs ---
        y_full = prepare_y(y)
        X_full, feature_names = prepare_x(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        k = X_full.shape[1]
        self.n_features_in_ = k
        W = prepare_w(W)

        # --- Panel structure ---
        if n_units is not None and n_periods is not None:
            self.n_units = n_units
            self.n_periods = n_periods
        panel = self._build_panel(
            len(y_full), unit_ids=unit_ids, time_ids=time_ids,
        )
        self.panel_ = panel
        N, T = panel.n_units, panel.n_periods

        y_full = panel.sort_data(y_full)
        X_full = panel.sort_data(X_full)

        validate_inputs(X_full[:N], y_full[:N], W, N)

        # --- Build WX (per period) ---
        X_panel = panel.reshape_to_panel(X_full)  # (N, T, k)
        WX_panel = np.zeros_like(X_panel)
        for t in range(T):
            WX_panel[:, t, :] = W @ X_panel[:, t, :]
        WX_full = panel.flatten_panel(WX_panel)  # (N*T, k)

        # --- Build Wy_{t-1} ---
        wy_lag, valid_mask_st = build_spatiotemporal_lag(
            y_full, W, panel, lag=1,
        )

        # --- Temporal lag for valid mask (same as lag=1) ---
        _, valid_mask = build_temporal_lag(y_full, panel, lag=1)

        # Subset to valid (t >= 1)
        X_v = X_full[valid_mask]
        WX_v = WX_full[valid_mask]

        # --- Augmented X: [X, WX, Wy_{t-1}] ---
        X_aug = np.column_stack([X_v, WX_v, wy_lag])

        # Build augmented y (stacked, valid periods only)
        y_v = y_full[valid_mask]

        # Rebuild the stacked augmented dataset as (N*T_eff)
        T_eff = T - 1
        n_aug_total = N * T
        X_aug_full = np.zeros((n_aug_total, X_aug.shape[1]))
        X_aug_full[valid_mask] = X_aug
        y_aug_full = np.zeros(n_aug_total)
        y_aug_full[valid_mask] = y_v

        # But for DynQuantSAR.fit, we need full (N*T_eff) stacked data
        # Re-stack as a clean (N * T_eff) panel
        y_panel_eff = y_full.reshape(N, T)[:, 1:]  # (N, T-1)
        X_aug_panel_eff = X_aug.reshape(N, T_eff, -1)

        y_eff = y_panel_eff.ravel()
        X_eff = X_aug_panel_eff.reshape(N * T_eff, -1)

        # --- Delegate to DynQuantSAR ---
        self._sar = DynQuantSAR(
            tau=self.tau,
            method=self.method,
            fixed_effects=self.fixed_effects,
            lam=self.lam,
            inference=self.inference,
            nboot=self.nboot,
            alpha=self.alpha,
            rhomat=self.rhomat,
            max_temporal_lag=self.max_temporal_lag,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self._sar.fit(X_eff, y_eff, W, n_units=N, n_periods=T_eff)

        # --- Unpack results ---
        # _sar.coef_ has [x1..xk, Wx1..Wxk, Wy_{t-1}]
        all_coefs = self._sar.coef_
        self.intercept_ = self._sar.intercept_
        self.coef_ = all_coefs[:k]
        self.theta_ = all_coefs[k:2 * k]
        self.delta_ = all_coefs[2 * k] if len(all_coefs) > 2 * k else 0.0
        self.rho_ = self._sar.rho_
        self.gamma_ = self._sar.gamma_
        self.se_ = self._sar.se_
        self.zvalues_ = self._sar.zvalues_
        self.pvalues_ = self._sar.pvalues_

        # Rebuild results with proper names
        var_names = self._build_st_var_names(
            k,
            include_temporal_lag=True,
            include_spatial_lag=True,
            include_st_lag=True,
            include_wx=True,
        )
        self.results_ = self._sar.results_.copy()
        self.results_.index = var_names[:len(self.results_)]

        # --- SDM-specific impacts ---
        if hasattr(self._sar, "impacts_"):
            self._build_sdm_impacts(W, N, k)

        if hasattr(self._sar, "rho_path_"):
            self.rho_path_ = self._sar.rho_path_
            self.rho_grid_ = self._sar.rho_grid_

        return self

    # ------------------------------------------------------------------
    # SDM impacts
    # ------------------------------------------------------------------

    def _build_sdm_impacts(
        self,
        W: WeightMatrix,
        N: int,
        k: int,
    ) -> None:
        """Compute SDM impacts including theta and delta terms.

        For the SDM, the contemporaneous impact of variable j is:

            S_j(W) = (I - rho W)^{-1} (beta_j I + theta_j W)

        Direct_j  = trace(S_j) / N
        Total_j   = 1' S_j 1 / N
        Indirect_j = Total_j - Direct_j
        """
        S_inv = solve_spatial_multiplier(self.rho_, W, N)
        if issparse(W):
            W_dense = W.toarray()
        else:
            W_dense = np.asarray(W)

        if hasattr(self, "feature_names_in_"):
            feat_names = list(self.feature_names_in_)
        else:
            feat_names = [f"x{i + 1}" for i in range(k)]

        direct = np.zeros(k)
        indirect = np.zeros(k)
        total = np.zeros(k)

        ones = np.ones(N)
        for j in range(k):
            S_j = S_inv @ (
                self.coef_[j] * np.eye(N) + self.theta_[j] * W_dense
            )
            direct[j] = np.trace(S_j) / N
            total[j] = (ones @ S_j @ ones) / N
            indirect[j] = total[j] - direct[j]

        self.impacts_ = pd.DataFrame(
            {
                "Direct": direct,
                "Indirect (Spillover)": indirect,
                "Total": total,
            },
            index=feat_names,
        )

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: ArrayLike,
        W: WeightMatrix | None = None,
        y: ArrayLike | None = None,
        y_prev: ArrayLike | None = None,
    ) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Current-period explanatory variables.
        W : array-like or sparse of shape (N, N), optional
            Spatial weight matrix.
        y : array-like of shape (n,), optional
            Current observed values for rho * Wy.
        y_prev : array-like of shape (n,), optional
            Previous-period values for gamma * y_{t-1} and
            delta * Wy_{t-1}.

        Returns
        -------
        y_pred : ndarray
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = X @ self.coef_ + self.intercept_

        if W is not None:
            WX = W @ X
            y_pred = y_pred + WX @ self.theta_

            if y is not None:
                y_obs = np.asarray(y, dtype=float).ravel()
                y_pred = y_pred + self.rho_ * (W @ y_obs)

            if y_prev is not None:
                y_prev_arr = np.asarray(y_prev, dtype=float).ravel()
                y_pred = y_pred + self.gamma_ * y_prev_arr
                y_pred = y_pred + self.delta_ * (W @ y_prev_arr)
        else:
            if y_prev is not None:
                y_prev_arr = np.asarray(y_prev, dtype=float).ravel()
                y_pred = y_pred + self.gamma_ * y_prev_arr

        return y_pred

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Print formatted results and impacts.

        Returns
        -------
        results : DataFrame
        """
        self._check_is_fitted()
        method_name = {
            "two_stage": "Kim & Muller Two-Stage (Panel)",
            "grid_search": "Chernozhukov & Hansen Grid Search (Panel)",
        }
        print("=" * 75)
        print(
            f"Dynamic Spatial Durbin Model -- {method_name[self.method]}"
        )
        print(
            f"  y_{{it}} = rho Wy + X beta + WX theta "
            f"+ gamma y_{{t-1}} + delta Wy_{{t-1}} + alpha_i + u"
        )
        print(f"  Quantile: tau = {self.tau}")
        print(f"  Fixed effects: {self.fixed_effects}")
        if hasattr(self, "panel_"):
            print(
                f"  Panel: N = {self.panel_.n_units}, "
                f"T = {self.panel_.n_periods}"
            )
        print("=" * 75)

        print("1. MODEL COEFFICIENTS")
        print("-" * 75)
        print(self.results_.to_string(float_format=lambda x: f"{x:.5f}"))

        if hasattr(self, "impacts_"):
            print("\n2. SPATIAL IMPACTS (LeSage & Pace)")
            print("-" * 75)
            print(self.impacts_.to_string(float_format=lambda x: f"{x:.5f}"))

        print("=" * 75)
        return self.results_
