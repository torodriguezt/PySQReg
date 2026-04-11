"""DynQuantSLX -- Dynamic SLX Quantile Regression for panel data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from ..areal._base import (
    ArrayLike,
    WeightMatrix,
    bofinger_bandwidth,
    prepare_w,
    prepare_x,
    prepare_y,
    validate_inputs,
)
from ._base import BaseSTQuantReg, penalized_qreg, select_lambda_bic
from ._panel import (
    PanelStructure,
    build_fixed_effects_dummies,
    build_temporal_lag,
    subset_to_valid,
)


class DynQuantSLX(BaseSTQuantReg):
    """Dynamic Spatial Lag of X quantile regression for panel data.

    Estimates the model::

        y_{it} = X_{it} beta + W X_{it} theta + gamma y_{i,t-1}
                 + alpha_i + u_{it},   Q_tau(u | X) = 0

    The temporal lag ``y_{i,t-1}`` makes this a dynamic panel model.
    Since there is no endogenous spatial lag (Wy), the spatial
    component enters only through the exogenous WX terms.

    Fixed effects ``alpha_i`` are handled via Koenker (2004) penalized
    quantile regression with an L1 penalty, avoiding the incidental
    parameters problem in short panels.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    fixed_effects : {'penalized', 'dummies', 'none'}
        Strategy for unit fixed effects.

        * ``'penalized'`` -- Koenker (2004) L1-penalized FE (default).
        * ``'dummies'`` -- include raw FE dummies (biased for small T).
        * ``'none'`` -- pooled estimation, no fixed effects.
    lam : float or None
        Penalty parameter for penalized FE.  When *None*, selected
        automatically via BIC.
    inference : {'bootstrap', 'analytical'} or None
        ``'bootstrap'`` (default) or ``'analytical'`` (sandwich SE).
    nboot : int
        Bootstrap replications.
    alpha : float
        Significance level for confidence intervals.
    verbose : int
        Verbosity level.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : ndarray
        Coefficients for X variables (beta).
    theta_ : ndarray
        Coefficients for WX variables (theta, spatial spillovers).
    gamma_ : float
        Temporal autoregressive parameter.
    intercept_ : float
        Intercept (or mean of penalized fixed effects).
    alpha_i_ : ndarray or None
        Estimated unit fixed effects (when applicable).
    se_ : ndarray
        Standard errors for slope parameters.
    pvalues_ : ndarray
        Two-sided p-values.
    results_ : DataFrame
        Full results table.
    panel_ : PanelStructure
        Panel structure metadata.

    Examples
    --------
    >>> model = DynQuantSLX(tau=0.5, fixed_effects='penalized')
    >>> model.fit(X, y, W, n_units=50, n_periods=10)
    >>> model.summary()
    """

    _param_names = (
        "tau", "fixed_effects", "lam", "inference", "nboot",
        "alpha", "verbose", "random_state",
    )

    def __init__(
        self,
        tau: float = 0.5,
        fixed_effects: str = "penalized",
        lam: float | None = None,
        inference: str | None = None,
        nboot: int = 100,
        alpha: float = 0.05,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1 (exclusive)")
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
        self.fixed_effects = fixed_effects
        self.lam = lam
        self.inference = inference
        self.nboot = nboot
        self.alpha = alpha
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
    ) -> DynQuantSLX:
        """Fit the dynamic SLX quantile regression model.

        Data must be stacked in long format (N*T rows), ordered by
        unit then time within each unit.  Provide either
        ``(n_units, n_periods)`` or ``(unit_ids, time_ids)``.

        Parameters
        ----------
        X : array-like of shape (N*T, k)
            Explanatory variables **without** intercept.
        y : array-like of shape (N*T,)
            Dependent variable.
        W : array-like or sparse of shape (N, N)
            Row-standardized spatial weight matrix.
        n_units : int, optional
            Number of cross-sectional units.
        n_periods : int, optional
            Number of time periods.
        unit_ids : array-like, optional
            Unit identifiers (alternative to n_units).
        time_ids : array-like, optional
            Time identifiers (alternative to n_periods).

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

        # --- Build panel structure ---
        if n_units is not None and n_periods is not None:
            self.n_units = n_units
            self.n_periods = n_periods
        panel = self._build_panel(
            len(y_full), unit_ids=unit_ids, time_ids=time_ids,
        )
        self.panel_ = panel

        # Sort if IDs provided
        y_full = panel.sort_data(y_full)
        X_full = panel.sort_data(X_full)

        # Validate spatial dimensions
        validate_inputs(
            X_full[:panel.n_units],
            y_full[:panel.n_units],
            W,
            panel.n_units,
        )

        # --- Build WX ---
        X_panel = panel.reshape_to_panel(X_full)  # (N, T, k)
        WX_panel = np.zeros_like(X_panel)
        for t in range(panel.n_periods):
            WX_panel[:, t, :] = W @ X_panel[:, t, :]
        WX_full = panel.flatten_panel(WX_panel)

        # --- Build temporal lag ---
        y_lag, valid_mask = build_temporal_lag(y_full, panel, lag=1)

        # Subset all arrays to valid observations (t >= 1)
        y_v, X_v, WX_v = subset_to_valid(valid_mask, y_full, X_full, WX_full)
        n_valid = len(y_v)

        # --- Assemble design matrix ---
        # [intercept, X, WX, y_{t-1}]
        X_design = sm.add_constant(np.column_stack([X_v, WX_v, y_lag]))

        # --- Fixed effects ---
        D = None
        if self.fixed_effects != "none":
            D_full = build_fixed_effects_dummies(panel)
            D_v = D_full[valid_mask]
            D = D_v

        # --- Estimate ---
        if self.fixed_effects == "penalized" and D is not None:
            bmat, alpha_hat = self._fit_penalized(y_v, X_design, D)
            self.alpha_i_ = alpha_hat
        elif self.fixed_effects == "dummies" and D is not None:
            X_with_fe = np.column_stack([X_design, D])
            bmat = self._fit_qreg(y_v, X_with_fe)
            self.alpha_i_ = bmat[X_design.shape[1]:]
            bmat = bmat[:X_design.shape[1]]
        else:
            bmat = self._fit_qreg(y_v, X_design)
            self.alpha_i_ = None

        # --- Inference ---
        inference = self.inference or "bootstrap"
        if inference == "bootstrap":
            se = self._bootstrap_se(
                y_full, X_full, WX_full, W, panel, valid_mask, y_lag,
                D, bmat,
            )
            se_label = "Bootstrap SE"
        else:
            se = self._analytical_se(y_v, X_design, bmat, D)
            se_label = "Std. Err."

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # --- Store attributes ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:k + 1]
        self.theta_ = bmat[k + 1:2 * k + 1]
        self.gamma_ = bmat[-1]
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._build_st_var_names(
            k, include_temporal_lag=True, include_wx=True,
        )
        self.results_ = pd.DataFrame(
            {
                "Coef.": bmat,
                se_label: se,
                "Z-values": z_values,
                "Pr(>|z|)": p_values,
            },
            index=var_names[:len(bmat)],
        )

        # Cache for predict
        self._valid_mask = valid_mask

        return self

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------

    def _fit_qreg(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Standard quantile regression (no penalty)."""
        model = sm.QuantReg(y, X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(q=self.tau, max_iter=5000)
        return fit.params

    def _fit_penalized(
        self,
        y: np.ndarray,
        X: np.ndarray,
        D: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit with Koenker (2004) penalized fixed effects."""
        lam = self.lam
        if lam is None:
            if self.verbose:
                print("  Selecting lambda via BIC...")
            lam = select_lambda_bic(y, X, D, self.tau)
            if self.verbose:
                print(f"  Selected lambda = {lam:.4f}")
            self.lam_ = lam

        beta, alpha = penalized_qreg(y, X, D, self.tau, lam)
        return beta, alpha

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _bootstrap_se(
        self,
        y_full: np.ndarray,
        X_full: np.ndarray,
        WX_full: np.ndarray,
        W: WeightMatrix,
        panel: PanelStructure,
        valid_mask: np.ndarray,
        y_lag: np.ndarray,
        D: np.ndarray | None,
        bmat: np.ndarray,
    ) -> np.ndarray:
        """Paired bootstrap standard errors.

        Resamples entire units (block bootstrap) to preserve the
        within-unit temporal structure.
        """
        n_params = len(bmat)
        bootmat = np.zeros((self.nboot, n_params))
        rng = np.random.default_rng(self.random_state)

        N = panel.n_units
        T = panel.n_periods

        for iboot in range(self.nboot):
            # Block bootstrap: resample whole units
            units = rng.choice(N, size=N, replace=True)

            # Reconstruct stacked data for resampled units
            y_panel = panel.reshape_to_panel(y_full)
            X_panel = panel.reshape_to_panel(X_full)
            WX_panel = panel.reshape_to_panel(WX_full)

            y_b = y_panel[units].reshape(-1)
            X_b = X_panel[units].reshape(-1, X_full.shape[1])
            WX_b = WX_panel[units].reshape(-1, WX_full.shape[1])

            # Temporal lag for resampled data
            y_b_panel = y_b.reshape(N, T)
            y_lag_b = y_b_panel[:, :-1].ravel()

            # Valid obs (drop t=0 per unit)
            y_bv = y_b_panel[:, 1:].ravel()
            X_bv = X_b.reshape(N, T, -1)[:, 1:, :].reshape(-1, X_full.shape[1])
            WX_bv = WX_b.reshape(N, T, -1)[:, 1:, :].reshape(
                -1, WX_full.shape[1],
            )

            X_design_b = sm.add_constant(
                np.column_stack([X_bv, WX_bv, y_lag_b]),
            )

            try:
                if self.fixed_effects == "penalized" and D is not None:
                    # Rebuild FE dummies for resampled panel
                    panel_b = PanelStructure(N, T)
                    D_b = build_fixed_effects_dummies(panel_b)
                    D_bv = D_b.reshape(N, T, -1)[:, 1:, :].reshape(
                        -1, N,
                    )
                    lam = getattr(self, "lam_", self.lam) or 1.0
                    coefs, _ = penalized_qreg(
                        y_bv, X_design_b, D_bv, self.tau, lam,
                    )
                elif self.fixed_effects == "dummies" and D is not None:
                    panel_b = PanelStructure(N, T)
                    D_b = build_fixed_effects_dummies(panel_b)
                    D_bv = D_b.reshape(N, T, -1)[:, 1:, :].reshape(
                        -1, N,
                    )
                    X_with_fe = np.column_stack([X_design_b, D_bv])
                    all_params = self._fit_qreg(y_bv, X_with_fe)
                    coefs = all_params[:X_design_b.shape[1]]
                else:
                    coefs = self._fit_qreg(y_bv, X_design_b)

                bootmat[iboot, :len(coefs)] = coefs
            except Exception:
                bootmat[iboot, :] = np.nan

            if self.verbose and (iboot + 1) % max(1, self.nboot // 10) == 0:
                print(f"  Bootstrap: {iboot + 1}/{self.nboot}")

        valid = ~np.any(np.isnan(bootmat), axis=1)
        if valid.sum() < self.nboot * 0.5:
            warnings.warn(
                f"Only {valid.sum()}/{self.nboot} bootstrap "
                "iterations converged."
            )
        return np.std(bootmat[valid], axis=0, ddof=0)

    def _analytical_se(
        self,
        y: np.ndarray,
        X: np.ndarray,
        bmat: np.ndarray,
        D: np.ndarray | None,
    ) -> np.ndarray:
        """Sandwich standard errors for the slope parameters."""
        if D is not None and self.fixed_effects == "penalized":
            # Use the design matrix without FE for SE computation
            pass

        e = y - X @ bmat
        n = len(y)
        h = bofinger_bandwidth(n, self.tau)
        tau_upper = min(self.tau + h, 0.9999)
        tau_lower = max(self.tau - h, 0.0001)
        dq = np.quantile(e, tau_upper) - np.quantile(e, tau_lower)
        fe = (tau_upper - tau_lower) / dq if dq > 1e-8 else 0.0

        J = fe * (X.T @ X) / n
        V = self.tau * (1 - self.tau) * (X.T @ X) / n

        try:
            J_inv = np.linalg.solve(J, np.eye(J.shape[0]))
            var_mat = (J_inv @ V @ J_inv.T) / n
            se = np.sqrt(np.maximum(np.diag(var_mat), 0))
        except np.linalg.LinAlgError:
            warnings.warn("Singular J matrix; falling back to bootstrap.")
            se = np.full(len(bmat), np.nan)

        return se

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: ArrayLike,
        W: WeightMatrix | None = None,
        y_prev: ArrayLike | None = None,
    ) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Current-period explanatory variables.
        W : array-like or sparse of shape (N, N), optional
            Spatial weight matrix for WX spillovers.
        y_prev : array-like of shape (n,), optional
            Previous-period values y_{t-1} for the temporal lag.

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

        if y_prev is not None:
            y_prev = np.asarray(y_prev, dtype=float).ravel()
            y_pred = y_pred + self.gamma_ * y_prev

        return y_pred

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Print formatted results.

        Returns
        -------
        results : DataFrame
        """
        self._check_is_fitted()
        print("=" * 75)
        print("Dynamic SLX Quantile Regression  (Panel)")
        print(f"  y_{{it}} = X beta + WX theta + gamma y_{{t-1}} + alpha_i + u")
        print(f"  Quantile: tau = {self.tau}")
        print(f"  Fixed effects: {self.fixed_effects}")
        if hasattr(self, "panel_"):
            print(
                f"  Panel: N = {self.panel_.n_units}, "
                f"T = {self.panel_.n_periods}"
            )
        print("=" * 75)
        print(self.results_.to_string(float_format=lambda x: f"{x:.5f}"))
        print("=" * 75)
        return self.results_
