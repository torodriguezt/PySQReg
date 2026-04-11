"""DynQuantSAR -- Dynamic Spatial Autoregressive Quantile Regression."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import issparse
from scipy.stats import norm

from ..areal._base import (
    ArrayLike,
    WeightMatrix,
    bofinger_bandwidth,
    compute_impacts,
    impact_scalars,
    prepare_w,
    prepare_x,
    prepare_y,
    solve_spatial_multiplier,
    validate_inputs,
)
from ._base import (
    BaseSTQuantReg,
    build_panel_instruments,
    panel_qriv_two_stage,
    penalized_qreg,
    select_lambda_bic,
)
from ._panel import (
    PanelStructure,
    build_fixed_effects_dummies,
    build_spatial_lag_panel,
    build_temporal_lag,
    subset_to_valid,
)


class DynQuantSAR(BaseSTQuantReg):
    """Dynamic Spatial Autoregressive Quantile Regression for panel data.

    Estimates the model::

        y_{it} = rho W y_{it} + X_{it} beta + gamma y_{i,t-1}
                 + alpha_i + u_{it},   Q_tau(u | X) = 0

    Both the contemporaneous spatial lag ``W y_{it}`` and the temporal
    lag ``y_{i,t-1}`` are endogenous.  The spatial lag is instrumented
    with ``WX, W^2 X`` (as in the cross-sectional QuantSAR), and the
    temporal lag is instrumented with deeper lags ``y_{i,t-2}, ...``
    (Arellano-Bond style).

    Two estimation strategies are available:

    * ``'two_stage'`` -- Kim & Muller (2004) extended to panels.
    * ``'grid_search'`` -- Chernozhukov & Hansen (2006) extended.

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
    intercept_ : float
        Intercept.
    rho_ : float
        Spatial autoregressive parameter.
    gamma_ : float
        Temporal autoregressive parameter.
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
    >>> model = DynQuantSAR(tau=0.5, method='two_stage')
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
    ) -> DynQuantSAR:
        """Fit the dynamic spatial quantile regression model.

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

        validate_inputs(
            X_full[:N], y_full[:N], W, N,
        )

        # --- Spatial lag (contemporaneous) ---
        wy_full = build_spatial_lag_panel(y_full, W, panel)

        # --- Temporal lag ---
        y_lag, valid_mask = build_temporal_lag(y_full, panel, lag=1)

        # Subset to valid (t >= 1)
        y_v, X_v, wy_v = subset_to_valid(valid_mask, y_full, X_full, wy_full)
        n_valid = len(y_v)

        # --- Design matrix: [const, X, y_{t-1}] ---
        X_with_const = sm.add_constant(np.column_stack([X_v, y_lag]))

        # --- Instruments ---
        Z = build_panel_instruments(
            sm.add_constant(X_v),
            W, panel, valid_mask, y_full,
            include_temporal=True,
            max_temporal_lag=self.max_temporal_lag,
        )

        # --- Fixed effects ---
        D = None
        if self.fixed_effects != "none":
            D_full = build_fixed_effects_dummies(panel)
            D_v = D_full[valid_mask]
            D = D_v

        # Number of slope params: const + k + gamma + rho
        n_params = k + 3

        # --- Resolve inference ---
        inference = self.inference
        if inference is None:
            inference = (
                "bootstrap" if self.method == "two_stage" else "analytical"
            )

        if self.method == "two_stage":
            self._fit_two_stage(
                y_v, X_v, X_with_const, wy_v, y_lag, Z, W, D,
                panel, valid_mask, y_full, N, n_valid, n_params, k,
                inference=inference,
            )
        elif self.method == "grid_search":
            self._fit_grid_search(
                y_v, X_v, X_with_const, wy_v, y_lag, Z, W, D,
                panel, valid_mask, y_full, N, n_valid, n_params, k,
                inference=inference,
            )

        return self

    # ------------------------------------------------------------------
    # Two-stage estimation
    # ------------------------------------------------------------------

    def _fit_two_stage(
        self,
        y_v: np.ndarray,
        X_v: np.ndarray,
        X_with_const: np.ndarray,
        wy_v: np.ndarray,
        y_lag: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        D: np.ndarray | None,
        panel: PanelStructure,
        valid_mask: np.ndarray,
        y_full: np.ndarray,
        N: int,
        n_valid: int,
        n_params: int,
        k: int,
        *,
        inference: str = "bootstrap",
    ) -> None:
        """Two-stage IV estimation for the dynamic SAR panel model."""
        # Stage 1-2 IV estimation
        bmat = panel_qriv_two_stage(
            wy_v, Z, y_v, X_with_const, self.tau,
        )
        # bmat = [const, x1..xk, y_{t-1}, wy_hat_coef]
        # The last element is the coefficient on wy_hat => rho

        if inference == "bootstrap":
            se = self._bootstrap_two_stage(
                y_full, X_v, W, panel, valid_mask, y_lag, Z,
                X_with_const, wy_v, D, N, n_params, k, bmat,
            )
            se_label = "Bootstrap SE"
        else:
            se = self._analytical_se(
                y_v, X_with_const, wy_v, Z, bmat, n_valid,
            )
            se_label = "Std. Err."

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # --- Unpack coefficients ---
        # bmat layout: [const, x1..xk, y_{t-1}, rho]
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:k + 1]
        self.gamma_ = bmat[k + 1]
        self.rho_ = bmat[-1]
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._build_st_var_names(
            k, include_temporal_lag=True, include_spatial_lag=True,
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

        # --- Spatial impacts ---
        self._compute_and_store_impacts(k, W, N, var_names)

    def _bootstrap_two_stage(
        self,
        y_full: np.ndarray,
        X_v: np.ndarray,
        W: WeightMatrix,
        panel: PanelStructure,
        valid_mask: np.ndarray,
        y_lag: np.ndarray,
        Z: np.ndarray,
        X_with_const: np.ndarray,
        wy_v: np.ndarray,
        D: np.ndarray | None,
        N: int,
        n_params: int,
        k: int,
        bmat: np.ndarray,
    ) -> np.ndarray:
        """Block bootstrap for two-stage panel SAR."""
        bootmat = np.zeros((self.nboot, len(bmat)))
        rng = np.random.default_rng(self.random_state)
        T = panel.n_periods

        for iboot in range(self.nboot):
            units = rng.choice(N, size=N, replace=True)
            try:
                # Reconstruct resampled panel
                y_panel = panel.reshape_to_panel(y_full)
                y_b = y_panel[units].reshape(-1)

                # Rebuild spatial lag for resampled data
                wy_b = build_spatial_lag_panel(
                    y_b, W, PanelStructure(N, T),
                )

                # Temporal lag
                y_b_panel = y_b.reshape(N, T)
                y_lag_b = y_b_panel[:, :-1].ravel()
                y_bv = y_b_panel[:, 1:].ravel()
                wy_bv = wy_b.reshape(N, T)[:, 1:].ravel()

                # X for resampled units
                X_full_r = panel.reshape_to_panel(
                    np.zeros((panel.n_total, X_v.shape[1])),
                )
                # We need the original X_v mapped back... use subset
                X_full_orig = np.zeros((panel.n_total, X_v.shape[1]))
                X_full_orig[valid_mask] = X_v
                X_panel_orig = panel.reshape_to_panel(X_full_orig)
                X_b = X_panel_orig[units].reshape(N, T, -1)
                X_bv = X_b[:, 1:, :].reshape(-1, X_v.shape[1])

                X_wc_b = sm.add_constant(
                    np.column_stack([X_bv, y_lag_b]),
                )

                # Rebuild instruments
                valid_b = np.zeros(N * T, dtype=bool)
                valid_b_panel = valid_b.reshape(N, T)
                valid_b_panel[:, 1:] = True
                valid_b = valid_b_panel.ravel()

                Z_b = build_panel_instruments(
                    sm.add_constant(X_bv), W,
                    PanelStructure(N, T), valid_b, y_b,
                    include_temporal=True,
                    max_temporal_lag=self.max_temporal_lag,
                )

                coefs = panel_qriv_two_stage(
                    wy_bv, Z_b, y_bv, X_wc_b, self.tau,
                )
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

    # ------------------------------------------------------------------
    # Grid search estimation
    # ------------------------------------------------------------------

    def _fit_grid_search(
        self,
        y_v: np.ndarray,
        X_v: np.ndarray,
        X_with_const: np.ndarray,
        wy_v: np.ndarray,
        y_lag: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        D: np.ndarray | None,
        panel: PanelStructure,
        valid_mask: np.ndarray,
        y_full: np.ndarray,
        N: int,
        n_valid: int,
        n_params: int,
        k: int,
        *,
        inference: str = "analytical",
    ) -> None:
        """Chernozhukov & Hansen grid search for panel SAR."""
        if self.rhomat is None:
            rhomat = np.arange(-1, 1.01, 0.01)
        else:
            rhomat = np.asarray(self.rhomat)

        # OLS first stage for wy
        Z_with_const = sm.add_constant(Z)
        ols_fit = sm.OLS(wy_v, Z_with_const).fit()
        wyhat = ols_fit.predict(Z_with_const)

        # Grid search: for each rho, regress (y - rho*wy) on [X, y_lag, wyhat]
        rhohat = np.zeros(len(rhomat))
        for i, rho_candidate in enumerate(rhomat):
            newy = y_v - rho_candidate * wy_v
            X_ch = np.column_stack([X_with_const, wyhat])
            model = sm.QuantReg(newy, X_ch)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = model.fit(q=self.tau, max_iter=5000)
            rhohat[i] = fit.params[-1]

        j = np.argmin(np.abs(rhohat))
        if j == 0 or j == len(rhomat) - 1:
            warnings.warn(
                "rho is at an endpoint of rhomat. "
                "Consider expanding the range."
            )
        minrho = rhomat[j]

        self.rho_path_ = rhohat
        self.rho_grid_ = rhomat

        # Final estimation at optimal rho
        newy = y_v - minrho * wy_v
        model_final = sm.QuantReg(newy, X_with_const)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_final = model_final.fit(q=self.tau, max_iter=5000)

        bmat = np.append(fit_final.params, minrho)

        # --- Inference ---
        if inference == "analytical":
            se = self._analytical_se(
                y_v, X_with_const, wy_v, Z, bmat, n_valid,
            )
            se_label = "Std. Err."
        else:
            # Bootstrap for grid search is expensive; warn user
            if self.verbose:
                print("  Bootstrap for grid_search is computationally heavy.")
            se = np.full(len(bmat), np.nan)
            se_label = "Std. Err."
            warnings.warn(
                "Bootstrap for grid_search in panel SAR is not yet "
                "implemented. Using NaN for standard errors."
            )

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # --- Unpack ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:k + 1]
        self.gamma_ = bmat[k + 1]
        self.rho_ = minrho
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._build_st_var_names(
            k, include_temporal_lag=True, include_spatial_lag=True,
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

        self._compute_and_store_impacts(k, W, N, var_names)

    # ------------------------------------------------------------------
    # Analytical SE
    # ------------------------------------------------------------------

    def _analytical_se(
        self,
        y_v: np.ndarray,
        X_with_const: np.ndarray,
        wy_v: np.ndarray,
        Z: np.ndarray,
        bmat: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """Sandwich variance estimator for the panel SAR model."""
        Z_with_const = sm.add_constant(Z)
        wyhat = sm.OLS(wy_v, Z_with_const).fit().predict(Z_with_const)

        S = np.column_stack([X_with_const, wyhat])
        e = y_v - S @ bmat

        h = bofinger_bandwidth(n, self.tau)
        tau_upper = min(self.tau + h, 0.9999)
        tau_lower = max(self.tau - h, 0.0001)
        dq = np.quantile(e, tau_upper) - np.quantile(e, tau_lower)
        fe = (tau_upper - tau_lower) / dq if dq > 1e-8 else 0.0

        B = np.column_stack([X_with_const, wy_v])
        A = fe * S
        try:
            jmat = np.linalg.solve(A.T @ B, np.eye(A.shape[1]))
        except np.linalg.LinAlgError:
            warnings.warn("Singular matrix in SE computation.")
            return np.full(len(bmat), np.nan)

        smat = S.T @ S
        var_mat = self.tau * (1 - self.tau) * (jmat @ smat @ jmat.T)
        return np.sqrt(np.maximum(np.diag(var_mat), 0))

    # ------------------------------------------------------------------
    # Impacts
    # ------------------------------------------------------------------

    def _compute_and_store_impacts(
        self,
        k: int,
        W: WeightMatrix,
        N: int,
        var_names: list[str],
    ) -> None:
        """Compute and store LeSage & Pace spatial impacts."""
        tr_n, sum_n = impact_scalars(
            self.rho_, W, N, random_state=self.random_state,
        )
        d_pt, ind_pt, t_pt = compute_impacts(
            self.coef_, self.rho_, W, N, tr_n=tr_n, sum_n=sum_n,
        )

        feat_names = var_names[1:k + 1]
        self.impacts_ = pd.DataFrame(
            {
                "Direct": d_pt,
                "Indirect (Spillover)": ind_pt,
                "Total": t_pt,
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
            Current observed values for spatial lag rho * W @ y.
        y_prev : array-like of shape (n,), optional
            Previous-period values for temporal lag gamma * y_{t-1}.

        Returns
        -------
        y_pred : ndarray
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = X @ self.coef_ + self.intercept_

        if W is not None and y is not None:
            y_obs = np.asarray(y, dtype=float).ravel()
            wy = W @ y_obs
            y_pred = y_pred + self.rho_ * wy

        if y_prev is not None:
            y_prev = np.asarray(y_prev, dtype=float).ravel()
            y_pred = y_pred + self.gamma_ * y_prev

        return y_pred

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Print formatted results and spatial impacts.

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
        print(f"Dynamic SAR Quantile Regression -- {method_name[self.method]}")
        print(f"  y_{{it}} = rho Wy + X beta + gamma y_{{t-1}} + alpha_i + u")
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
