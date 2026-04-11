"""QuantSDM -- Spatial Durbin Quantile Regression for areal data."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse

from ._base import (
    ArrayLike,
    BaseSpatialQuantReg,
    WeightMatrix,
    prepare_w,
    prepare_x,
    prepare_y,
    solve_spatial_multiplier,
    validate_inputs,
)
from .sar import QuantSAR


class QuantSDM(BaseSpatialQuantReg):
    """Spatial Durbin Quantile Regression for areal (lattice) data.

    Estimates the model

        y = rho * W @ y + X @ beta + W @ X @ theta + u,
        Q_tau(u | X) = 0

    This nests both QuantSAR (theta=0) and QuantSLX (rho=0) as special
    cases.  The endogenous spatial lag (W @ y) is handled via IV
    approaches identical to QuantSAR, while the exogenous WX terms
    enter directly.

    Two estimation strategies are available:

    * ``'two_stage'`` -- Kim & Muller (2004) two-stage quantile
      regression with bootstrap inference.
    * ``'grid_search'`` -- Chernozhukov & Hansen (2006) IV quantile
      regression.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    method : {'two_stage', 'grid_search'}
        Estimation method.
    inference : {'bootstrap', 'analytical'} or None
        Inference strategy. Default follows method (bootstrap for
        two_stage, analytical for grid_search).
    nboot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    rhomat : array-like or None
        Grid of candidate rho values (grid_search only).
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
    se_ : ndarray
        Standard errors for all parameters.
    pvalues_ : ndarray
        Two-sided p-values.
    results_ : DataFrame
        Full results table.
    impacts_ : DataFrame
        Direct, indirect, and total spatial impacts.

    Examples
    --------
    >>> model = QuantSDM(tau=0.5, method='two_stage', nboot=200)
    >>> model.fit(X, y, W)
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
    ) -> QuantSDM:
        """Fit the Spatial Durbin quantile regression model.

        The model augments QuantSAR by including WX as additional
        regressors.  Internally, it delegates to a QuantSAR instance
        with ``X_augmented = [X, WX]``.

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
        y_arr = prepare_y(y)
        X_arr, feature_names = prepare_x(X)
        if feature_names is not None:
            self.feature_names_in_ = feature_names

        n = len(y_arr)
        k = X_arr.shape[1]
        self.n_features_in_ = k

        W = prepare_w(W)
        validate_inputs(X_arr, y_arr, W, n)

        WX = W @ X_arr
        X_aug = np.column_stack([X_arr, WX])

        # Delegate to QuantSAR with augmented X
        self._sar = QuantSAR(
            tau=self.tau,
            method=self.method,
            inference=self.inference,
            nboot=self.nboot,
            alpha=self.alpha,
            rhomat=self.rhomat,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self._sar.fit(X_aug, y_arr, W)

        # Unpack results
        all_coefs = self._sar.coef_  # [x1..xk, Wx1..Wxk]
        self.intercept_ = self._sar.intercept_
        self.coef_ = all_coefs[:k]
        self.theta_ = all_coefs[k:]
        self.rho_ = self._sar.rho_
        self.se_ = self._sar.se_
        self.zvalues_ = self._sar.zvalues_
        self.pvalues_ = self._sar.pvalues_

        # Rebuild results table with proper names
        var_names = self._sdm_var_names(k)
        self.results_ = self._sar.results_.copy()
        self.results_.index = var_names[:len(self.results_)]

        # Impacts -- for SDM the marginal effect includes both beta and
        # theta through the spatial multiplier
        if hasattr(self._sar, "impacts_"):
            self._build_sdm_impacts(W, n, k)

        if hasattr(self._sar, "rho_path_"):
            self.rho_path_ = self._sar.rho_path_
            self.rho_grid_ = self._sar.rho_grid_

        return self

    # ------------------------------------------------------------------
    # SDM-specific impacts
    # ------------------------------------------------------------------

    def _build_sdm_impacts(
        self,
        W: WeightMatrix,
        n: int,
        k: int,
    ) -> None:
        """Compute SDM impacts: direct/indirect/total per LeSage & Pace.

        For the SDM, the impact of variable j is:

            S_j(W) = (I - rho W)^{-1} (beta_j I + theta_j W)

        Direct_j  = trace(S_j) / n
        Total_j   = 1' S_j 1 / n
        Indirect_j = Total_j - Direct_j
        """
        S_inv = solve_spatial_multiplier(self.rho_, W, n)
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

        ones = np.ones(n)
        for j in range(k):
            S_j = S_inv @ (
                self.coef_[j] * np.eye(n) + self.theta_[j] * W_dense
            )
            direct[j] = np.trace(S_j) / n
            total[j] = (ones @ S_j @ ones) / n
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
    ) -> np.ndarray:
        """Predict target values.

        When W and y are provided, the full SDM prediction is returned.

        Parameters
        ----------
        X : array-like of shape (n, k)
        W : array-like or sparse, optional
        y : array-like, optional

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
                wy = W @ y_obs
                y_pred = y_pred + self.rho_ * wy
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
            "two_stage": "Kim & Muller Two-Stage Quantile Regression",
            "grid_search": "Chernozhukov & Hansen IV Quantile Regression",
        }
        print("=" * 75)
        print(f"Spatial Durbin Model -- {method_name[self.method]}")
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _sdm_var_names(self, k: int) -> list[str]:
        """Build variable names for the SDM results table."""
        if hasattr(self, "feature_names_in_"):
            x_names = list(self.feature_names_in_)
        else:
            x_names = [f"x{i + 1}" for i in range(k)]
        wx_names = [f"W*{name}" for name in x_names]
        return ["(Intercept)"] + x_names + wx_names + ["WY"]
