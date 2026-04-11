"""QuantSLX -- Spatial Lag of X Quantile Regression for areal data."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import issparse
from scipy.stats import norm

from ._base import (
    ArrayLike,
    BaseSpatialQuantReg,
    WeightMatrix,
    prepare_w,
    prepare_x,
    prepare_y,
    validate_inputs,
)


class QuantSLX(BaseSpatialQuantReg):
    """Quantile regression with Spatial Lag of X (SLX) specification.

    Estimates the model

        y = X @ beta + W @ X @ theta + u,    Q_tau(u | X) = 0

    This is a purely exogenous model -- there is no endogenous spatial
    lag (W @ y), so standard quantile regression applies directly.
    The spatial spillover is captured by the WX terms, where theta
    measures the indirect effect of neighbours' covariates.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    inference : {'bootstrap', 'analytical'} or None
        ``'bootstrap'`` uses nonparametric paired bootstrap;
        ``'analytical'`` uses the standard quantile regression
        covariance (Powell kernel).  Default is ``'analytical'``.
    nboot : int
        Bootstrap replications when ``inference='bootstrap'``.
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
    intercept_ : float
        Intercept.
    se_ : ndarray
        Standard errors for all parameters.
    pvalues_ : ndarray
        Two-sided p-values.
    results_ : DataFrame
        Full results table.

    Examples
    --------
    >>> model = QuantSLX(tau=0.5)
    >>> model.fit(X, y, W)
    >>> model.summary()
    """

    _param_names = (
        "tau", "inference", "nboot", "alpha", "verbose", "random_state",
    )

    def __init__(
        self,
        tau: float = 0.5,
        inference: str | None = None,
        nboot: int = 100,
        alpha: float = 0.05,
        verbose: int = 0,
        random_state: int | None = None,
    ) -> None:
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1 (exclusive)")
        if inference is not None and inference not in ("bootstrap", "analytical"):
            raise ValueError(
                f"inference must be 'bootstrap', 'analytical', or None, "
                f"got '{inference}'."
            )
        self.tau = tau
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
    ) -> QuantSLX:
        """Fit the SLX quantile regression model.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Explanatory variables **without** intercept.
        y : array-like of shape (n,)
            Dependent variable.
        W : array-like or sparse of shape (n, n)
            Row-standardized spatial weight matrix.

        Returns
        -------
        self
        """
        y = prepare_y(y)
        X, feature_names = prepare_x(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        n = len(y)
        k = X.shape[1]
        self.n_features_in_ = k

        W = prepare_w(W)
        validate_inputs(X, y, W, n)

        WX = W @ X
        X_full = sm.add_constant(np.column_stack([X, WX]))

        inference = self.inference if self.inference is not None else "analytical"

        if inference == "analytical":
            bmat, se = self._fit_analytical(y, X_full)
        else:
            bmat, se = self._fit_bootstrap(y, X_full, n)

        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # Store attributes
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:k + 1]
        self.theta_ = bmat[k + 1:]
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._slx_var_names(k)
        self.results_ = pd.DataFrame(
            {
                "Coef.": bmat,
                "Std. Err.": se,
                "Z-values": z_values,
                "Pr(>|z|)": p_values,
            },
            index=var_names,
        )

        return self

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _fit_analytical(
        self,
        y: np.ndarray,
        X_full: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit with analytical (Powell kernel) standard errors."""
        model = sm.QuantReg(y, X_full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(q=self.tau, max_iter=5000)
        return fit.params, fit.bse

    def _fit_bootstrap(
        self,
        y: np.ndarray,
        X_full: np.ndarray,
        n: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit with bootstrap standard errors."""
        # Point estimates
        model = sm.QuantReg(y, X_full)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = model.fit(q=self.tau, max_iter=5000)
        bmat = fit.params

        # Bootstrap SEs
        bootmat = np.zeros((self.nboot, len(bmat)))
        rng = np.random.default_rng(self.random_state)
        for iboot in range(self.nboot):
            idx = rng.choice(n, size=n, replace=True)
            try:
                m = sm.QuantReg(y[idx], X_full[idx])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = m.fit(q=self.tau, max_iter=5000)
                bootmat[iboot] = f.params
            except Exception:
                bootmat[iboot] = np.nan
            if self.verbose and (iboot + 1) % max(1, self.nboot // 10) == 0:
                print(f"  Bootstrap: {iboot + 1}/{self.nboot}")

        valid = ~np.any(np.isnan(bootmat), axis=1)
        if valid.sum() < self.nboot * 0.5:
            warnings.warn(
                f"Only {valid.sum()}/{self.nboot} bootstrap "
                "iterations converged."
            )
        se = np.std(bootmat[valid], axis=0, ddof=0)
        return bmat, se

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(
        self,
        X: ArrayLike,
        W: WeightMatrix | None = None,
    ) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n, k)
            Explanatory variables **without** intercept.
        W : array-like or sparse of shape (n, n), optional
            Spatial weight matrix. If provided, WX spillovers are
            included in the prediction.

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
        print("SLX Quantile Regression  (y = Xbeta + WXtheta + u)")
        print(f"Quantile: tau = {self.tau}")
        print("=" * 75)
        print(self.results_.to_string(float_format=lambda x: f"{x:.5f}"))
        print("=" * 75)
        return self.results_

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _slx_var_names(self, k: int) -> list[str]:
        """Build variable names for the SLX results table."""
        if hasattr(self, "feature_names_in_"):
            x_names = list(self.feature_names_in_)
        else:
            x_names = [f"x{i + 1}" for i in range(k)]
        wx_names = [f"W*{name}" for name in x_names]
        return ["(Intercept)"] + x_names + wx_names
