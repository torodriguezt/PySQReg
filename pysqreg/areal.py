import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from scipy.sparse import issparse, csc_matrix, eye as speye
from scipy.sparse.linalg import splu
import warnings


class MoranResult:
    """Results container for Moran's I test for spatial autocorrelation.

    Attributes
    ----------
    I : float
        Moran's I statistic.
    EI : float
        Expected value of I under the null hypothesis of no spatial
        autocorrelation, equal to -1/(n-1).
    VI : float
        Variance of I under the chosen assumption.
    z : float
        Standardised z-score  (I - E[I]) / sqrt(Var[I]).
    p_value : float
        p-value for the chosen alternative.
    n : int
        Number of observations.
    alternative : str
        Alternative hypothesis used.
    assumption : str
        Distributional assumption used for the variance.
    """

    def __init__(self, I, EI, VI, z, p_value, n, alternative, assumption):
        self.I = I
        self.EI = EI
        self.VI = VI
        self.z = z
        self.p_value = p_value
        self.n = n
        self.alternative = alternative
        self.assumption = assumption

    def __repr__(self):
        return (
            f"MoranResult(I={self.I:.6f}, E[I]={self.EI:.6f}, "
            f"z={self.z:.4f}, p={self.p_value:.4f})"
        )

    def summary(self):
        """Print a formatted summary of the Moran's I test."""
        print("=" * 55)
        print("  Moran's I Test for Spatial Autocorrelation")
        print("=" * 55)
        print(f"  Moran's I statistic : {self.I: .6f}")
        print(f"  Expected value E[I] : {self.EI: .6f}")
        print(f"  Variance Var[I]     : {self.VI: .6f}")
        print(f"  Z-score             : {self.z: .4f}")
        print(f"  p-value ({self.alternative:>9s}) : {self.p_value: .6f}")
        print(f"  Assumption          : {self.assumption}")
        print(f"  N                   : {self.n}")
        print("-" * 55)
        if self.p_value < 0.05:
            if self.I > self.EI:
                print("  Significant POSITIVE spatial autocorrelation")
            else:
                print("  Significant NEGATIVE spatial autocorrelation")
        else:
            print("  No significant spatial autocorrelation detected")
        print("=" * 55)


def moran_test(x, W, alternative='two-sided', assumption='randomization'):
    """Moran's I test for spatial autocorrelation in lattice data.

    Computes the Moran's I statistic (Cliff & Ord, 1981) and tests for
    spatial autocorrelation.  This test can be applied to raw variables
    to detect spatial structure, or to model residuals to check whether
    a fitted model has captured all spatial dependence.

    Parameters
    ----------
    x : array-like of shape (n,)
        Variable to test — either raw observations or model residuals.
    W : array-like or sparse of shape (n, n)
        Spatial weight matrix (typically row-standardised).
    alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis.

        * ``'two-sided'`` — spatial autocorrelation (positive or negative).
        * ``'greater'`` — positive spatial autocorrelation (clustering
          of similar values).
        * ``'less'`` — negative spatial autocorrelation (dispersion /
          checkerboard pattern).
    assumption : {'randomization', 'normality'}
        Distributional assumption used to compute Var[I].

        * ``'randomization'`` — distribution-free variance that adjusts
          for the sample kurtosis (more robust; default).
        * ``'normality'`` — assumes the data are normally distributed.

    Returns
    -------
    result : MoranResult
        Object with attributes ``I``, ``EI``, ``VI``, ``z``,
        ``p_value``, ``n``, ``alternative``, and ``assumption``.
        Call ``result.summary()`` for a formatted printout.

    Notes
    -----
    Moran's I is defined as

    .. math::

        I = \\frac{n}{S_0} \\, \\frac{\\mathbf{z}' \\mathbf{W} \\mathbf{z}}
                                     {\\mathbf{z}' \\mathbf{z}}

    where :math:`\\mathbf{z} = \\mathbf{x} - \\bar{x}` and
    :math:`S_0 = \\sum_{i,j} w_{ij}`.

    Under the null hypothesis of no spatial autocorrelation,
    :math:`E[I] = -1/(n-1)`.  The variance depends on the
    distributional assumption chosen.

    Examples
    --------
    >>> from pysqreg import moran_test
    >>> result = moran_test(y, W)
    >>> result.summary()

    Test model residuals:

    >>> model = QuantSAR(tau=0.5, method='two_stage')
    >>> model.fit(X, y, W)
    >>> residuals = y - model.predict(X, W, y)
    >>> moran_test(residuals, W).summary()
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)

    if n < 3:
        raise ValueError("Need at least 3 observations for Moran's I test.")

    z = x - x.mean()
    z2 = z @ z

    if z2 < 1e-15:
        raise ValueError(
            "The variable has (near-)zero variance; "
            "Moran's I is not defined for a constant."
        )

    # ---- Spatial-weight summaries ----
    sparse = issparse(W)
    if sparse:
        Wz = W @ z
        S0 = float(W.sum())
        WpWT = W + W.T
        S1 = 0.5 * float(WpWT.multiply(WpWT).sum())
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        col_sums = np.asarray(W.sum(axis=0)).ravel()
        S2 = float(((row_sums + col_sums) ** 2).sum())
    else:
        W = np.asarray(W, dtype=float)
        Wz = W @ z
        S0 = float(W.sum())
        WpWT = W + W.T
        S1 = 0.5 * float((WpWT ** 2).sum())
        S2 = float(((W.sum(axis=1) + W.sum(axis=0)) ** 2).sum())

    # ---- Moran's I ----
    zWz = float(z @ Wz)
    I = (n / S0) * (zWz / z2)

    # ---- Expected value (same under both assumptions) ----
    EI = -1.0 / (n - 1)

    # ---- Variance ----
    S0sq = S0 ** 2

    if assumption == 'normality':
        VI = (
            (n ** 2 * S1 - n * S2 + 3 * S0sq) / (S0sq * (n ** 2 - 1))
            - EI ** 2
        )
    elif assumption == 'randomization':
        m2 = z2 / n
        m4 = float((z ** 4).sum()) / n
        b2 = m4 / (m2 ** 2)                       # kurtosis

        A = n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0sq)
        B = b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0sq)
        C = (n - 1) * (n - 2) * (n - 3) * S0sq

        VI = (A - B) / C - EI ** 2
    else:
        raise ValueError(
            f"assumption must be 'normality' or 'randomization', "
            f"got '{assumption}'."
        )

    if VI < 0:
        warnings.warn("Computed variance of I is negative; setting to 0.")
        VI = 0.0

    # ---- Z-score and p-value ----
    z_score = (I - EI) / np.sqrt(VI) if VI > 0 else np.inf

    if alternative == 'two-sided':
        p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))
    elif alternative == 'greater':
        p_value = 1.0 - norm.cdf(z_score)
    elif alternative == 'less':
        p_value = norm.cdf(z_score)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', "
            f"got '{alternative}'."
        )

    return MoranResult(
        I=I, EI=EI, VI=VI, z=z_score,
        p_value=p_value, n=n,
        alternative=alternative, assumption=assumption,
    )


class QuantSAR:
    """Spatial Quantile Regression for areal (lattice) data.

    Estimates the SAR model  y = rho * W @ y + X @ beta + u  at a given
    quantile tau using instrumental-variable approaches designed for
    areal / lattice spatial data with a row-standardized contiguity
    weight matrix W.

    Two estimation strategies are available:

    * ``'two_stage'`` — Kim & Muller (2004) two-stage quantile regression
      with bootstrap inference.
    * ``'grid_search'`` — Chernozhukov & Hansen (2006) IV quantile
      regression that finds the optimal rho via grid search.

    Parameters
    ----------
    tau : float
        Quantile to estimate, strictly between 0 and 1.
    method : {'two_stage', 'grid_search'}
        Estimation method.
    nboot : int
        Number of bootstrap replications (only used by ``'two_stage'``).
    alpha : float
        Significance level for confidence intervals.
    rhomat : array-like or None
        Grid of candidate rho values (only used by ``'grid_search'``).
        Defaults to ``np.arange(-1, 1.01, 0.01)`` when *None*.
    verbose : int
        Verbosity level.  0 = silent, 1 = progress messages.

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
    conf_int_ : ndarray of shape (k + 2, 2)
        Confidence intervals (only for ``'two_stage'``; percentile bootstrap).
    results_ : DataFrame
        Full results table.
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

    def __init__(self, tau=0.5, method='two_stage', nboot=100, alpha=0.05,
                 rhomat=None, verbose=0, random_state=None):
        if not 0 < tau < 1:
            raise ValueError("tau must be between 0 and 1 (exclusive)")
        self.tau = tau
        self.method = method
        self.nboot = nboot
        self.alpha = alpha
        self.rhomat = rhomat
        self.verbose = verbose
        self.random_state = random_state

    def __repr__(self):
        params = ', '.join(
            f'{k}={v!r}' for k, v in self.get_params().items()
        )
        return f'{self.__class__.__name__}({params})'

    # ------------------------------------------------------------------
    # scikit-learn compatible parameter interface
    # ------------------------------------------------------------------

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Ignored.  Present for scikit-learn compatibility.

        Returns
        -------
        params : dict
        """
        return {
            'tau': self.tau,
            'method': self.method,
            'nboot': self.nboot,
            'alpha': self.alpha,
            'rhomat': self.rhomat,
            'verbose': self.verbose,
            'random_state': self.random_state,
        }

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(
                    f"Invalid parameter '{key}' for "
                    f"{self.__class__.__name__}."
                )
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y, W, wy=None, inst=None, winst=None):
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
        y = np.asarray(y, dtype=float).ravel()

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns)

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = len(y)
        self.n_features_in_ = X.shape[1]

        if not issparse(W):
            W = np.asarray(W, dtype=float)

        if wy is None:
            wy = W @ y
        else:
            wy = np.asarray(wy, dtype=float).ravel()

        X_with_const = sm.add_constant(X)
        k_with_const = X_with_const.shape[1]

        if inst is None and winst is None:
            WX = W @ X
            Z = np.column_stack([X_with_const, WX])
        elif inst is None and winst is not None:
            winst = np.asarray(winst, dtype=float)
            W_winst = W @ winst
            Z = np.column_stack([X_with_const, W_winst])
        elif inst is not None and winst is None:
            Z = np.asarray(inst, dtype=float)
            if Z.ndim == 1:
                Z = Z.reshape(-1, 1)
        else:
            inst_arr = np.asarray(inst, dtype=float)
            winst_arr = np.asarray(winst, dtype=float)
            W_winst = W @ winst_arr
            Z = np.column_stack([inst_arr, W_winst])

        if np.allclose(Z[:, 0], Z[0, 0]):
            Z = Z[:, 1:]

        nk = k_with_const + 1

        if self.method == 'two_stage':
            self._fit_two_stage(y, X, X_with_const, wy, Z, W, n, nk)
        elif self.method == 'grid_search':
            self._fit_grid_search(y, X, X_with_const, wy, Z, W, n, nk)
        else:
            raise ValueError(
                f"Method '{self.method}' not recognized. "
                "Use 'two_stage' or 'grid_search'."
            )

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, X, W=None, y=None):
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

    def summary(self):
        """Print coefficient results and spatial impacts (if available).

        Returns
        -------
        results : DataFrame
        """
        self._check_is_fitted()

        method_name = {
            'two_stage': 'Kim & Muller Two-Stage Quantile Regression',
            'grid_search': 'Chernozhukov & Hansen IV Quantile Regression',
        }
        print("=" * 75)
        print(f"{method_name[self.method]}")
        print(f"Quantile: tau = {self.tau}")
        print("=" * 75)

        print("1. MODEL COEFFICIENTS")
        print("-" * 75)
        print(self.results_.to_string(float_format=lambda x: f"{x:.5f}"))

        if hasattr(self, 'impacts_'):
            print("\n2. SPATIAL IMPACTS (LeSage & Pace)")
            print("-" * 75)
            print(self.impacts_.to_string(float_format=lambda x: f"{x:.5f}"))

        print("=" * 75)
        return self.results_



    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_is_fitted(self):
        """Raise if the model has not been fitted."""
        if not hasattr(self, 'coef_'):
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this "
                "method."
            )

    @staticmethod
    def _qriv(wy, Z, y, X_with_const, tau):
        """Two-stage IV quantile regression helper."""
        Z_with_const = sm.add_constant(Z)
        model_stage1 = sm.QuantReg(wy, Z_with_const)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_stage1 = model_stage1.fit(q=tau, max_iter=5000)

        wyhat = fit_stage1.predict(Z_with_const)

        X_stage2 = np.column_stack([X_with_const, wyhat])
        model_stage2 = sm.QuantReg(y, X_stage2)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_stage2 = model_stage2.fit(q=tau, max_iter=5000)

        return fit_stage2.params

    # ------------------------------------------------------------------
    # Two-stage (Kim & Muller 2004)
    # ------------------------------------------------------------------

    def _fit_two_stage(self, y, X, X_with_const, wy, Z, W, n, nk):
        # 1. Fit on the original sample
        bmat = self._qriv(wy, Z, y, X_with_const, self.tau)

        # 2. Initialise matrices for coefficients and impacts
        bootmat = np.zeros((self.nboot, nk))

        k_vars = X.shape[1]
        boot_direct = np.zeros((self.nboot, k_vars))
        boot_indirect = np.zeros((self.nboot, k_vars))
        boot_total = np.zeros((self.nboot, k_vars))

        # 3. Bootstrap loop
        rng = np.random.default_rng(self.random_state)
        for iboot in range(self.nboot):
            bobs = rng.choice(n, size=n, replace=True)

            try:
                coefs = self._qriv(
                    wy[bobs], Z[bobs], y[bobs],
                    X_with_const[bobs], self.tau,
                )
                bootmat[iboot, :len(coefs)] = coefs

                # Compute impacts for this bootstrap draw
                beta_boot = coefs[1:-1]
                rho_boot = coefs[-1]
                tr_n, sum_n = self._impact_scalars(
                    rho_boot, W, n,
                )
                d, ind, t = self._compute_impacts(
                    beta_boot, rho_boot, W, n,
                    tr_n=tr_n, sum_n=sum_n,
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

        # 4. Drop non-converged iterations
        valid_mask = ~np.any(np.isnan(bootmat), axis=1)
        bootmat_valid = bootmat[valid_mask]
        b_dir_valid = boot_direct[valid_mask]
        b_ind_valid = boot_indirect[valid_mask]
        b_tot_valid = boot_total[valid_mask]

        if len(bootmat_valid) < self.nboot * 0.5:
            warnings.warn(
                f"Only {len(bootmat_valid)}/{self.nboot} bootstrap "
                "iterations converged."
            )

        # 5. Coefficient inference
        se = np.std(bootmat_valid, axis=0, ddof=0)
        z_values = bmat / se
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # 6. Impact inference (LeSage & Pace)
        tr_n, sum_n = self._impact_scalars(bmat[-1], W, n)
        d_pt, ind_pt, t_pt = self._compute_impacts(
            bmat[1:-1], bmat[-1], W, n,
            tr_n=tr_n, sum_n=sum_n,
        )
        se_dir = np.std(b_dir_valid, axis=0, ddof=0)
        se_ind = np.std(b_ind_valid, axis=0, ddof=0)
        se_tot = np.std(b_tot_valid, axis=0, ddof=0)

        # --- Store fitted attributes ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:-1]
        self.rho_ = bmat[-1]
        self.se_ = se
        self.zvalues_ = z_values
        self.pvalues_ = p_values

        var_names = self._var_names(X)
        feat_names = var_names[1:-1]

        self.results_ = pd.DataFrame({
            'Coef.': bmat,
            'Bootstrap SE': se,
            'Z-values': z_values,
            'Pr(>|z|)': p_values,
        }, index=var_names[:nk])

        self.impacts_ = pd.DataFrame({
            'Direct': d_pt,
            'SE Direct': se_dir,
            'Indirect (Spillover)': ind_pt,
            'SE Indirect': se_ind,
            'p-val Indirect': 2 * (1 - norm.cdf(np.abs(ind_pt / se_ind))),
            'Total': t_pt,
            'SE Total': se_tot,
        }, index=feat_names)

    # ------------------------------------------------------------------
    # Grid search (Chernozhukov & Hansen 2006)
    # ------------------------------------------------------------------

    def _fit_grid_search(self, y, X, X_with_const, wy, Z, W, n, nk):
        if self.rhomat is None:
            rhomat = np.arange(-1, 1.01, 0.01)
        else:
            rhomat = np.asarray(self.rhomat)

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

        S = np.column_stack([wyhat, X_with_const])
        smat = S.T @ S

        e = fit_final.resid
        h = self._bofinger_bandwidth(n, self.tau)

        tau_upper = min(self.tau + h, 0.9999)
        tau_lower = max(self.tau - h, 0.0001)

        q_upper = np.quantile(e, tau_upper)
        q_lower = np.quantile(e, tau_lower)
        dq = q_upper - q_lower

        fe_scalar = (tau_upper - tau_lower) / dq if dq > 1e-8 else 0.0

        phistar = (fe_scalar * wyhat).reshape(-1, 1)
        xstar = X_with_const * fe_scalar

        A = np.column_stack([phistar, xstar])
        B = np.column_stack([wy, X_with_const])
        jmat = np.linalg.solve(A.T @ B, np.eye(A.shape[1]))

        var_mat = self.tau * (1 - self.tau) * (jmat @ smat @ jmat.T)
        se_all = np.sqrt(np.diag(var_mat))
        se_reordered = np.append(se_all[1:], se_all[0])

        z_values = bmat / se_reordered
        p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

        # --- Store fitted attributes ---
        self.intercept_ = bmat[0]
        self.coef_ = bmat[1:-1]
        self.rho_ = minrho
        self.se_ = se_reordered
        self.zvalues_ = z_values
        self.pvalues_ = p_values
        self.rho_path_ = rhohat
        self.rho_grid_ = rhomat

        var_names = self._var_names(X)

        self.results_ = pd.DataFrame({
            'Coef.': bmat,
            'Std. Err.': se_reordered,
            'Z-Values': z_values,
            'Pr(>|z|)': p_values,
        }, index=var_names[:nk])

        # --- Spatial impacts (LeSage & Pace) with Delta method SEs ---
        beta_gs = bmat[1:-1]
        k_vars = len(beta_gs)
        feat_names = var_names[1:-1]

        S_inv = self._solve_multiplier(minrho, W, n)
        d_pt, ind_pt, t_pt = self._compute_impacts(
            beta_gs, minrho, W, n, S_inv=S_inv,
        )

        if not issparse(S_inv):
            # Delta method — var_mat is ordered [rho, intercept, x1, ..., xk]
            if issparse(W):
                W_dm = W.toarray()
            else:
                W_dm = W
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

            self.impacts_ = pd.DataFrame({
                'Direct': d_pt,
                'SE Direct': se_dir,
                'Indirect (Spillover)': ind_pt,
                'SE Indirect': se_ind,
                'p-val Indirect': 2 * (
                    1 - norm.cdf(np.abs(ind_pt / se_ind))
                ),
                'Total': t_pt,
                'SE Total': se_tot,
            }, index=feat_names)
        else:
            self.impacts_ = pd.DataFrame({
                'Direct': d_pt,
                'Indirect (Spillover)': ind_pt,
                'Total': t_pt,
            }, index=feat_names)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _bofinger_bandwidth(n, tau):
        """Bofinger (1975) bandwidth for sparsity estimation.

        h = n^{-1/5} * (4.5 * phi^4(xi) / (2*xi^2 + 1)^2)^{1/5}

        where xi = Phi^{-1}(tau) and phi is the standard normal density.
        """
        xi = norm.ppf(tau)
        phi_xi = norm.pdf(xi)
        return n ** (-1 / 5) * (
            4.5 * phi_xi ** 4 / (2 * xi ** 2 + 1) ** 2
        ) ** (1 / 5)

    def _var_names(self, X):
        """Build variable-name list from feature names or defaults."""
        k = X.shape[1] if X.ndim > 1 else 1
        if hasattr(self, 'feature_names_in_'):
            names = list(self.feature_names_in_)
        else:
            names = [f'x{i + 1}' for i in range(k)]
        return ['(Intercept)'] + names + ['WY']

    def _solve_multiplier(self, rho, W, n):
        """Compute (I - rho * W)^{-1} via sparse LU decomposition.

        Returns a dense array of shape (n, n).  Used by ``grid_search``
        where the full inverse is needed for the delta-method SEs.
        """
        W_csc = csc_matrix(W)
        A = speye(n, format='csc') - rho * W_csc
        lu = splu(A)
        return lu.solve(np.eye(n))

    def _impact_scalars(self, rho, W, n):
        """Compute trace(S^{-1})/n and sum(S^{-1})/n without full inversion.

        Uses a single sparse LU factorisation of A = (I - rho * W):

        * **sum** — one linear solve  A x = 1_n, then sum(x)/n.
        * **trace** — Hutchinson stochastic estimator with 50 Rademacher
          vectors, reusing the same factorisation.

        This is O(nnz) per solve instead of O(n^3) for full inversion,
        making it suitable for the bootstrap loop in ``two_stage``.
        """
        W_csc = csc_matrix(W)
        A = speye(n, format='csc') - rho * W_csc
        lu = splu(A)

        # sum((I - rho W)^{-1}) / n
        sum_n = lu.solve(np.ones(n)).sum() / n

        # trace((I - rho W)^{-1}) / n  — Hutchinson estimator
        m = 50
        rng = np.random.default_rng(self.random_state)
        tr = 0.0
        for _ in range(m):
            z = rng.choice([-1.0, 1.0], size=n)
            tr += z @ lu.solve(z)
        tr_n = tr / (m * n)

        return tr_n, sum_n

    def _compute_impacts(self, beta, rho, W, n, S_inv=None, tr_n=None,
                         sum_n=None):
        """Compute Direct, Indirect and Total impacts (LeSage & Pace).

        Parameters
        ----------
        S_inv : array-like, optional
            Pre-computed (I - rho * W)^{-1} to avoid redundant work.
        tr_n : float, optional
            Pre-computed trace((I - rho * W)^{-1}) / n.
        sum_n : float, optional
            Pre-computed sum((I - rho * W)^{-1}) / n.

        When *tr_n* and *sum_n* are both provided the full inverse is
        never constructed, which is much faster for large *n*.
        """
        if tr_n is None or sum_n is None:
            if S_inv is None:
                S_inv = self._solve_multiplier(rho, W, n)

            if issparse(S_inv):
                tr_n = S_inv.diagonal().sum() / n
                sum_n = S_inv.sum() / n
            else:
                tr_n = np.trace(S_inv) / n
                sum_n = S_inv.sum() / n

        direct = beta * tr_n
        total = beta * sum_n
        indirect = total - direct

        return direct, indirect, total
