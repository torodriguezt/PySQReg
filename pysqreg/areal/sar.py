"""QuantSAR -- Spatial Autoregressive Quantile Regression for areal data."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import issparse
from scipy.stats import norm

from ._base import (
    ArrayLike,
    BaseSpatialQuantReg,
    WeightMatrix,
    bofinger_bandwidth,
    build_instruments,
    compute_impacts,
    impact_scalars,
    prepare_w,
    prepare_x,
    prepare_y,
    qriv_two_stage,
    solve_spatial_multiplier,
    validate_inputs,
)


class QuantSAR(BaseSpatialQuantReg):
    """Spatial Quantile Regression for areal (lattice) data.

    Estimates the SAR model  y = rho * W @ y + X @ beta + u  at a given
    quantile tau using instrumental-variable approaches designed for
    areal / lattice spatial data with a row-standardized contiguity
    weight matrix W.

    Two estimation strategies are available:

    * ``'two_stage'`` -- Kim & Muller (2004) two-stage quantile regression
      with bootstrap inference.
    * ``'grid_search'`` -- Chernozhukov & Hansen (2006) IV quantile
      regression that finds the optimal rho via grid search.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    method : {'two_stage', 'grid_search'}
        Estimation method.
    inference : {'bootstrap', 'analytical'} or None
        Inference strategy.  ``'bootstrap'`` uses nonparametric paired
        bootstrap; ``'analytical'`` uses a sandwich / delta-method
        variance estimator.  When *None* (default), ``'two_stage'``
        defaults to ``'bootstrap'`` and ``'grid_search'`` defaults to
        ``'analytical'``.
    nboot : int
        Number of bootstrap replications (used when
        ``inference='bootstrap'``).
    alpha : float
        Significance level for confidence intervals.
    rhomat : array-like or None
        Grid of candidate rho values (only used by ``'grid_search'``).
        Defaults to ``np.arange(-1, 1.01, 0.01)`` when *None*.
    verbose : int
        Verbosity level.  0 = silent, 1 = progress messages.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    coef_ : ndarray of shape (k,)
        Estimated coefficients for explanatory variables (excluding
        intercept and spatial lag).
    intercept_ : float
        Estimated intercept.
    rho_ : float
        Estimated spatial autoregressive parameter.
    se_ : ndarray of shape (k + 2,)
        Standard errors for all parameters (intercept, covariates, rho).
    pvalues_ : ndarray of shape (k + 2,)
        Two-sided p-values.
    zvalues_ : ndarray of shape (k + 2,)
        Z-statistics.
    results_ : DataFrame
        Full results table.
    impacts_ : DataFrame
        Direct, indirect, and total spatial impacts.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.
    feature_names_in_ : ndarray of shape (k,)
        Feature names (only when *X* is a DataFrame).

    Examples
    --------
    >>> model = QuantSAR(tau=0.5, method='two_stage', nboot=200)
    >>> model.fit(X, y, W)
    >>> print(model.coef_)
    >>> print(model.rho_)
    >>> model.summary()
    """

    _param_names = (
        "tau", "method", "inference", "nboot", "alpha",
        "rhomat", "verbose", "random_state",
    )

    def __init__(
        self,
        tau: float = 0.5,
        method: str = "two_stage",
        inference: str | None = None,
        nboot: int = 100,
        alpha: float = 0.05,
        rhomat: npt.ArrayLike | None = None,
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
        if inference is not None and inference not in ("bootstrap", "analytical"):
            raise ValueError(
                f"inference must be 'bootstrap', 'analytical', or None, "
                f"got '{inference}'."
            )
        self.tau = tau
        self.method = method
        self.inference = inference
        self.nboot = nboot
        self.alpha = alpha
        self.rhomat = rhomat
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
        wy: ArrayLike | None = None,
        inst: ArrayLike | None = None,
        winst: ArrayLike | None = None,
    ) -> QuantSAR:
        """Fit the spatial quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Explanatory variables **without** intercept (added automatically).
        y : array-like of shape (n,)
            Dependent variable.
        W : array-like or sparse of shape (n, n)
            Row-standardized spatial weight matrix for areal data.
        wy : array-like of shape (n,) or None
            Pre-computed spatial lag ``W @ y``.  Computed internally when
            *None*.
        inst : array-like of shape (n, p) or None
            Additional instruments not pre-multiplied by W.
        winst : array-like of shape (n, q) or None
            Variables to pre-multiply by W to create instruments.

        Returns
        -------
        self
        """
        y = prepare_y(y)
        X, feature_names = prepare_x(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        n = len(y)
        self.n_features_in_ = X.shape[1]

        W = prepare_w(W)
        validate_inputs(X, y, W, n)

        if wy is None:
            wy = W @ y
        else:
            wy = np.asarray(wy, dtype=float).ravel()

        X_with_const = sm.add_constant(X)
        n_params = X_with_const.shape[1] + 1  # +1 for rho

        Z = build_instruments(X_with_const, W, X, inst=inst, winst=winst)

        # Resolve inference strategy
        inference = self.inference
        if inference is None:
            inference = (
                "bootstrap" if self.method == "two_stage" else "analytical"
            )

        if self.method == "two_stage":
            self._fit_two_stage(
                y, X, X_with_const, wy, Z, W, n, n_params,
                inference=inference,
            )
        elif self.method == "grid_search":
            self._fit_grid_search(
                y, X, X_with_const, wy, Z, W, n, n_params,
                inference=inference,
            )

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: ArrayLike,
        W: WeightMatrix | None = None,
        y: ArrayLike | None = None,
    ) -> np.ndarray:
        """Predict target values.

        When both *W* and *y* are provided the full SAR prediction is
        returned: ``X @ beta + intercept + rho * W @ y``.
        Otherwise only the non-spatial component is returned:
        ``X @ beta + intercept``.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Explanatory variables **without** intercept.
        W : array-like or sparse of shape (n, n), optional
            Spatial weight matrix.
        y : array-like of shape (n,), optional
            Observed values used to compute the spatial lag ``W @ y``.

        Returns
        -------
        y_pred : ndarray of shape (n,)
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        y_pred = X @ self.coef_ + self.intercept_

        if W is not None and y is not None:
            y_obs = np.asarray(y, dtype=float).ravel()
            if issparse(W):
                wy = W @ y_obs
            else:
                wy = np.asarray(W, dtype=float) @ y_obs
            y_pred = y_pred + self.rho_ * wy

        return y_pred

    # ------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------

    def summary(self) -> pd.DataFrame:
        """Print coefficient results and spatial impacts (if available).

        Returns
        -------
        results : DataFrame
        """
        self._check_is_fitted()

        method_name = {
            "two_stage": "Kim & Muller Two-Stage Quantile Regression",
            "grid_search": "Chernozhukov & Hansen IV Quantile Regression",
        }
        print("=" * 75)
        print(f"{method_name[self.method]}")
        print(f"Quantile: tau = {self.tau}")
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

    # ------------------------------------------------------------------
    # Two-stage (Kim & Muller 2004)
    # ------------------------------------------------------------------

    def _fit_two_stage(
        self,
        y: np.ndarray,
        X: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        n: int,
        n_params: int,
        *,
        inference: str = "bootstrap",
    ) -> None:
        """Estimate the model via Kim & Muller (2004) two-stage approach."""
        bmat = qriv_two_stage(wy, Z, y, X_with_const, self.tau)
        k_vars = X.shape[1]

        if inference == "bootstrap":
            se, se_dir, se_ind, se_tot = self._bootstrap_two_stage(
                y, X, X_with_const, wy, Z, W, n, n_params, k_vars, bmat,
            )
            se_label = "Bootstrap SE"
        else:
            se, se_dir, se_ind, se_tot = self._analytical_two_stage(
                y, X_with_const, wy, Z, W, n, bmat,
            )
            se_label = "Std. Err."

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # Impact point estimates
        tr_n, sum_n = impact_scalars(
            bmat[-1], W, n, random_state=self.random_state,
        )
        d_pt, ind_pt, t_pt = compute_impacts(
            bmat[1:-1], bmat[-1], W, n, tr_n=tr_n, sum_n=sum_n,
        )

        # --- Store fitted attributes ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:-1]
        self.rho_ = bmat[-1]
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._build_var_names(k_vars, suffix_names=["WY"])
        feat_names = var_names[1:-1]

        self.results_ = pd.DataFrame(
            {
                "Coef.": bmat,
                se_label: se,
                "Z-values": z_values,
                "Pr(>|z|)": p_values,
            },
            index=var_names[:n_params],
        )

        self.impacts_ = pd.DataFrame(
            {
                "Direct": d_pt,
                "SE Direct": se_dir,
                "Indirect (Spillover)": ind_pt,
                "SE Indirect": se_ind,
                "p-val Indirect": 2 * (1 - norm.cdf(np.abs(ind_pt / se_ind))),
                "Total": t_pt,
                "SE Total": se_tot,
            },
            index=feat_names,
        )

    # ------------------------------------------------------------------
    # Two-stage inference helpers
    # ------------------------------------------------------------------

    def _bootstrap_two_stage(
        self,
        y: np.ndarray,
        X: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        n: int,
        n_params: int,
        k_vars: int,
        bmat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Bootstrap inference for the two-stage estimator."""
        bootmat = np.zeros((self.nboot, n_params))
        boot_direct = np.zeros((self.nboot, k_vars))
        boot_indirect = np.zeros((self.nboot, k_vars))
        boot_total = np.zeros((self.nboot, k_vars))

        rng = np.random.default_rng(self.random_state)
        for iboot in range(self.nboot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                coefs = qriv_two_stage(
                    wy[idx], Z[idx], y[idx],
                    X_with_const[idx], self.tau,
                )
                bootmat[iboot, :len(coefs)] = coefs

                beta_boot = coefs[1:-1]
                rho_boot = coefs[-1]
                tr_n, sum_n = impact_scalars(
                    rho_boot, W, n, random_state=self.random_state,
                )
                d, ind, t = compute_impacts(
                    beta_boot, rho_boot, W, n, tr_n=tr_n, sum_n=sum_n,
                )
                boot_direct[iboot, :] = d
                boot_indirect[iboot, :] = ind
                boot_total[iboot, :] = t
            except Exception:
                bootmat[iboot, :] = np.nan
                boot_direct[iboot, :] = np.nan
                boot_indirect[iboot, :] = np.nan
                boot_total[iboot, :] = np.nan

            if self.verbose and (iboot + 1) % max(1, self.nboot // 10) == 0:
                print(f"  Bootstrap: {iboot + 1}/{self.nboot}")

        valid = ~np.any(np.isnan(bootmat), axis=1)
        bootmat_v = bootmat[valid]
        if len(bootmat_v) < self.nboot * 0.5:
            warnings.warn(
                f"Only {len(bootmat_v)}/{self.nboot} bootstrap "
                "iterations converged."
            )

        se = np.std(bootmat_v, axis=0, ddof=0)
        se_dir = np.std(boot_direct[valid], axis=0, ddof=0)
        se_ind = np.std(boot_indirect[valid], axis=0, ddof=0)
        se_tot = np.std(boot_total[valid], axis=0, ddof=0)
        return se, se_dir, se_ind, se_tot

    def _analytical_two_stage(
        self,
        y: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        n: int,
        bmat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Analytical sandwich SE for the two-stage estimator."""
        # Stage 1: OLS projection of wy onto Z
        Z_with_const = sm.add_constant(Z)
        wyhat = sm.OLS(wy, Z_with_const).fit().predict(Z_with_const)

        # Stage 2 design matrix
        S = np.column_stack([X_with_const, wyhat])
        e = y - S @ bmat

        h = bofinger_bandwidth(n, self.tau)
        tau_upper = min(self.tau + h, 0.9999)
        tau_lower = max(self.tau - h, 0.0001)
        dq = np.quantile(e, tau_upper) - np.quantile(e, tau_lower)
        fe = (tau_upper - tau_lower) / dq if dq > 1e-8 else 0.0

        # Sandwich: J^{-1} V J^{-T}
        B = np.column_stack([X_with_const, wy])
        A = fe * S
        jmat = np.linalg.solve(A.T @ B, np.eye(A.shape[1]))
        smat = S.T @ S
        var_mat = self.tau * (1 - self.tau) * (jmat @ smat @ jmat.T)

        se_all = np.sqrt(np.maximum(np.diag(var_mat), 0))

        # Impact SEs via delta method
        k_vars = len(bmat) - 2
        beta = bmat[1:-1]
        rho = bmat[-1]
        S_inv = solve_spatial_multiplier(rho, W, n)
        if issparse(W):
            W_dm = W.toarray()
        else:
            W_dm = np.asarray(W)
        SWS = S_inv @ W_dm @ S_inv
        tr_S_n = np.trace(S_inv) / n
        sum_S_n = S_inv.sum() / n
        tr_SWS_n = np.trace(SWS) / n
        sum_SWS_n = SWS.sum() / n

        se_dir = np.zeros(k_vars)
        se_ind = np.zeros(k_vars)
        se_tot = np.zeros(k_vars)
        for j in range(k_vars):
            g_d = np.zeros(var_mat.shape[0])
            g_d[-1] = beta[j] * tr_SWS_n
            g_d[j + 1] = tr_S_n
            se_dir[j] = np.sqrt(max(g_d @ var_mat @ g_d, 0))

            g_t = np.zeros(var_mat.shape[0])
            g_t[-1] = beta[j] * sum_SWS_n
            g_t[j + 1] = sum_S_n
            se_tot[j] = np.sqrt(max(g_t @ var_mat @ g_t, 0))

            g_i = g_t - g_d
            se_ind[j] = np.sqrt(max(g_i @ var_mat @ g_i, 0))

        return se_all, se_dir, se_ind, se_tot

    # ------------------------------------------------------------------
    # Grid search (Chernozhukov & Hansen 2006)
    # ------------------------------------------------------------------

    def _fit_grid_search(
        self,
        y: np.ndarray,
        X: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        n: int,
        n_params: int,
        *,
        inference: str = "analytical",
    ) -> None:
        """Estimate the model via Chernozhukov & Hansen (2006) grid search."""
        if self.rhomat is None:
            rhomat = np.arange(-1, 1.01, 0.01)
        else:
            rhomat = np.asarray(self.rhomat)

        minrho, rhohat, bmat, fit_final, wyhat = self._grid_search_core(
            y, X_with_const, wy, Z, rhomat,
        )

        self.rho_path_ = rhohat
        self.rho_grid_ = rhomat

        k_vars = len(bmat) - 2
        var_names = self._build_var_names(
            X.shape[1], suffix_names=["WY"],
        )
        feat_names = var_names[1:-1]

        if inference == "analytical":
            se, var_mat = self._analytical_grid_search(
                y, X_with_const, wy, wyhat, fit_final, bmat, n,
            )
            se_label = "Std. Err."
        else:
            se = self._bootstrap_grid_search(
                y, X, X_with_const, wy, Z, W, n, n_params, rhomat,
            )
            var_mat = None
            se_label = "Bootstrap SE"

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # --- Store fitted attributes ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:-1]
        self.rho_ = minrho
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        self.results_ = pd.DataFrame(
            {
                "Coef.": bmat,
                se_label: se,
                "Z-Values": z_values,
                "Pr(>|z|)": p_values,
            },
            index=var_names[:n_params],
        )

        # --- Spatial impacts (LeSage & Pace) ---
        beta_gs = bmat[1:-1]
        S_inv = solve_spatial_multiplier(minrho, W, n)
        d_pt, ind_pt, t_pt = compute_impacts(
            beta_gs, minrho, W, n, S_inv=S_inv,
        )

        if var_mat is not None and not issparse(S_inv):
            self._build_grid_search_impacts_with_se(
                var_mat, S_inv, W, n, k_vars, beta_gs, d_pt, ind_pt, t_pt,
                feat_names,
            )
        else:
            self.impacts_ = pd.DataFrame(
                {
                    "Direct": d_pt,
                    "Indirect (Spillover)": ind_pt,
                    "Total": t_pt,
                },
                index=feat_names,
            )

    def _build_grid_search_impacts_with_se(
        self,
        var_mat: np.ndarray,
        S_inv: np.ndarray,
        W: WeightMatrix,
        n: int,
        k_vars: int,
        beta_gs: np.ndarray,
        d_pt: np.ndarray,
        ind_pt: np.ndarray,
        t_pt: np.ndarray,
        feat_names: list[str],
    ) -> None:
        """Build the impacts DataFrame with delta-method SEs."""
        if issparse(W):
            W_dm = W.toarray()
        else:
            W_dm = np.asarray(W)
        SWS = S_inv @ W_dm @ S_inv
        tr_S_n = np.trace(S_inv) / n
        sum_S_n = S_inv.sum() / n
        tr_SWS_n = np.trace(SWS) / n
        sum_SWS_n = SWS.sum() / n

        se_dir = np.zeros(k_vars)
        se_ind = np.zeros(k_vars)
        se_tot = np.zeros(k_vars)
        for j in range(k_vars):
            g_d = np.zeros(var_mat.shape[0])
            g_d[0] = beta_gs[j] * tr_SWS_n
            g_d[j + 2] = tr_S_n
            se_dir[j] = np.sqrt(max(g_d @ var_mat @ g_d, 0))

            g_t = np.zeros(var_mat.shape[0])
            g_t[0] = beta_gs[j] * sum_SWS_n
            g_t[j + 2] = sum_S_n
            se_tot[j] = np.sqrt(max(g_t @ var_mat @ g_t, 0))

            g_i = g_t - g_d
            se_ind[j] = np.sqrt(max(g_i @ var_mat @ g_i, 0))

        self.impacts_ = pd.DataFrame(
            {
                "Direct": d_pt,
                "SE Direct": se_dir,
                "Indirect (Spillover)": ind_pt,
                "SE Indirect": se_ind,
                "p-val Indirect": 2 * (
                    1 - norm.cdf(np.abs(ind_pt / se_ind))
                ),
                "Total": t_pt,
                "SE Total": se_tot,
            },
            index=feat_names,
        )

    # ------------------------------------------------------------------
    # Grid search helpers
    # ------------------------------------------------------------------

    def _grid_search_core(
        self,
        y: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        rhomat: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray, object, np.ndarray]:
        """Run the CH grid search and return optimal rho + coefficients."""
        Z_with_const = sm.add_constant(Z)
        ols_fit = sm.OLS(wy, Z_with_const).fit()
        wyhat = ols_fit.predict(Z_with_const)

        rhohat = np.zeros(len(rhomat))
        for i, rho in enumerate(rhomat):
            newy = y - rho * wy
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

        newy = y - minrho * wy
        model_final = sm.QuantReg(newy, X_with_const)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_final = model_final.fit(q=self.tau, max_iter=5000)

        bmat = np.append(fit_final.params, minrho)
        return minrho, rhohat, bmat, fit_final, wyhat

    def _analytical_grid_search(
        self,
        y: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        wyhat: np.ndarray,
        fit_final: object,
        bmat: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sandwich variance estimator for grid_search."""
        S = np.column_stack([wyhat, X_with_const])
        smat = S.T @ S

        e = fit_final.resid
        h = bofinger_bandwidth(n, self.tau)
        tau_upper = min(self.tau + h, 0.9999)
        tau_lower = max(self.tau - h, 0.0001)
        dq = np.quantile(e, tau_upper) - np.quantile(e, tau_lower)
        fe_scalar = (tau_upper - tau_lower) / dq if dq > 1e-8 else 0.0

        phistar = (fe_scalar * wyhat).reshape(-1, 1)
        xstar = X_with_const * fe_scalar

        A = np.column_stack([phistar, xstar])
        B = np.column_stack([wy, X_with_const])
        jmat = np.linalg.solve(A.T @ B, np.eye(A.shape[1]))

        var_mat = self.tau * (1 - self.tau) * (jmat @ smat @ jmat.T)
        se_all = np.sqrt(np.maximum(np.diag(var_mat), 0))
        se_reordered = np.append(se_all[1:], se_all[0])
        return se_reordered, var_mat

    def _bootstrap_grid_search(
        self,
        y: np.ndarray,
        X: np.ndarray,
        X_with_const: np.ndarray,
        wy: np.ndarray,
        Z: np.ndarray,
        W: WeightMatrix,
        n: int,
        n_params: int,
        rhomat: np.ndarray,
    ) -> np.ndarray:
        """Bootstrap inference for the grid_search estimator."""
        bootmat = np.zeros((self.nboot, n_params))
        rng = np.random.default_rng(self.random_state)

        for iboot in range(self.nboot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                _, _, coefs, _, _ = self._grid_search_core(
                    y[idx], X_with_const[idx], wy[idx],
                    Z[idx], rhomat,
                )
                bootmat[iboot, :len(coefs)] = coefs
            except Exception:
                bootmat[iboot, :] = np.nan

            if self.verbose and (iboot + 1) % max(1, self.nboot // 10) == 0:
                print(f"  Bootstrap: {iboot + 1}/{self.nboot}")

        valid = ~np.any(np.isnan(bootmat), axis=1)
        bootmat_v = bootmat[valid]
        if len(bootmat_v) < self.nboot * 0.5:
            warnings.warn(
                f"Only {len(bootmat_v)}/{self.nboot} bootstrap "
                "iterations converged."
            )
        return np.std(bootmat_v, axis=0, ddof=0)
