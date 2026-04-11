"""Base class and shared utilities for spatial quantile regression on areal data."""

from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import statsmodels.api as sm
from scipy.sparse import csc_matrix, eye as speye, issparse, spmatrix
from scipy.sparse.linalg import splu
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
ArrayLike = Union[npt.ArrayLike, pd.DataFrame, pd.Series]
WeightMatrix = Union[npt.ArrayLike, spmatrix]


# ---------------------------------------------------------------------------
# Input preparation helpers
# ---------------------------------------------------------------------------

def prepare_y(y: ArrayLike) -> np.ndarray:
    """Convert *y* to a 1-D float array."""
    return np.asarray(y, dtype=float).ravel()


def prepare_x(X: ArrayLike) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert *X* to a 2-D float array.

    Returns
    -------
    X : ndarray of shape (n, k)
    feature_names : ndarray of str or None
        Column names when *X* is a DataFrame.
    """
    feature_names = np.array(X.columns) if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return X, feature_names


def prepare_w(W: WeightMatrix) -> WeightMatrix:
    """Ensure *W* is either a sparse matrix or a dense float array."""
    if not issparse(W):
        W = np.asarray(W, dtype=float)
    return W


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_inputs(
    X: np.ndarray,
    y: np.ndarray,
    W: WeightMatrix,
    n: int,
) -> None:
    """Validate dimensions and basic properties of model inputs.

    Parameters
    ----------
    X : ndarray of shape (n, k)
    y : ndarray of shape (n,)
    W : array-like or sparse of shape (n, n)
    n : int
        Expected number of observations.

    Raises
    ------
    ValueError
        On dimension mismatches or insufficient observations.
    """
    if X.shape[0] != n:
        raise ValueError(
            f"X has {X.shape[0]} rows but y has {n} observations."
        )

    w_rows, w_cols = W.shape
    if w_rows != w_cols:
        raise ValueError(
            f"W must be square, got shape ({w_rows}, {w_cols})."
        )
    if w_rows != n:
        raise ValueError(
            f"W has {w_rows} rows but y has {n} observations."
        )

    # Row-standardisation check (warn, don't error)
    if issparse(W):
        row_sums = np.asarray(W.sum(axis=1)).ravel()
    else:
        row_sums = W.sum(axis=1)
    non_zero_rows = row_sums[row_sums != 0]
    if len(non_zero_rows) > 0 and not np.allclose(non_zero_rows, 1.0, atol=1e-6):
        warnings.warn(
            "W does not appear to be row-standardised. "
            "Row sums range from {:.4f} to {:.4f}. "
            "Results assume a row-standardised weight matrix.".format(
                non_zero_rows.min(), non_zero_rows.max()
            ),
            UserWarning,
            stacklevel=4,
        )

    if n < X.shape[1] + 2:
        raise ValueError(
            f"Need at least k + 2 = {X.shape[1] + 2} observations, "
            f"got {n}."
        )


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def bofinger_bandwidth(n: int, tau: float) -> float:
    """Bofinger (1975) bandwidth for sparsity estimation.

    h = n^{-1/5} * (4.5 * phi^4(xi) / (2*xi^2 + 1)^2)^{1/5}

    where xi = Phi^{-1}(tau) and phi is the standard normal density.
    """
    xi = norm.ppf(tau)
    phi_xi = norm.pdf(xi)
    return n ** (-1 / 5) * (
        4.5 * phi_xi ** 4 / (2 * xi ** 2 + 1) ** 2
    ) ** (1 / 5)


def solve_spatial_multiplier(
    rho: float,
    W: WeightMatrix,
    n: int,
) -> np.ndarray:
    """Compute (I - rho * W)^{-1} via sparse LU decomposition.

    Returns a dense array of shape (n, n).
    """
    W_csc = csc_matrix(W)
    A = speye(n, format="csc") - rho * W_csc
    lu = splu(A)
    return lu.solve(np.eye(n))


def impact_scalars(
    rho: float,
    W: WeightMatrix,
    n: int,
    random_state: int | None = None,
) -> tuple[float, float]:
    """Compute trace(S^{-1})/n and sum(S^{-1})/n without full inversion.

    Uses a single sparse LU factorisation of A = (I - rho * W):

    * **sum** -- one linear solve  A x = 1_n, then sum(x)/n.
    * **trace** -- Hutchinson stochastic estimator with 50 Rademacher
      vectors, reusing the same factorisation.

    This is O(nnz) per solve instead of O(n^3) for full inversion,
    making it suitable for the bootstrap loop.
    """
    W_csc = csc_matrix(W)
    A = speye(n, format="csc") - rho * W_csc
    lu = splu(A)

    # sum((I - rho W)^{-1}) / n
    sum_n = lu.solve(np.ones(n)).sum() / n

    # trace((I - rho W)^{-1}) / n  -- Hutchinson estimator
    m = 50
    rng = np.random.default_rng(random_state)
    tr = 0.0
    for _ in range(m):
        z = rng.choice([-1.0, 1.0], size=n)
        tr += z @ lu.solve(z)
    tr_n = tr / (m * n)

    return tr_n, sum_n


def compute_impacts(
    beta: np.ndarray,
    rho: float,
    W: WeightMatrix,
    n: int,
    S_inv: np.ndarray | None = None,
    tr_n: float | None = None,
    sum_n: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Direct, Indirect, and Total impacts (LeSage & Pace).

    Parameters
    ----------
    beta : ndarray
        Coefficient vector (excluding intercept and rho).
    rho : float
        Spatial autoregressive parameter.
    W : array-like or sparse of shape (n, n)
    n : int
    S_inv : ndarray, optional
        Pre-computed (I - rho * W)^{-1}.
    tr_n : float, optional
        Pre-computed trace((I - rho * W)^{-1}) / n.
    sum_n : float, optional
        Pre-computed sum((I - rho * W)^{-1}) / n.

    Returns
    -------
    direct, indirect, total : tuple of ndarrays
    """
    if tr_n is None or sum_n is None:
        if S_inv is None:
            S_inv = solve_spatial_multiplier(rho, W, n)

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


# ---------------------------------------------------------------------------
# IV quantile regression helper
# ---------------------------------------------------------------------------

def qriv_two_stage(
    wy: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    X_with_const: np.ndarray,
    tau: float,
) -> np.ndarray:
    """Two-stage IV quantile regression (Kim & Muller, 2004).

    Stage 1: quantile regression of Wy on instruments Z.
    Stage 2: quantile regression of y on [X, Wy_hat].

    Returns the full coefficient vector from stage 2.
    """
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


# ---------------------------------------------------------------------------
# Instrument construction
# ---------------------------------------------------------------------------

def build_instruments(
    X_with_const: np.ndarray,
    W: WeightMatrix,
    X: np.ndarray,
    inst: ArrayLike | None = None,
    winst: ArrayLike | None = None,
) -> np.ndarray:
    """Build the instrument matrix Z for IV estimation.

    Parameters
    ----------
    X_with_const : ndarray of shape (n, k+1)
        Design matrix with intercept column.
    W : array-like or sparse of shape (n, n)
        Spatial weight matrix.
    X : ndarray of shape (n, k)
        Explanatory variables without intercept.
    inst : array-like or None
        Additional instruments (not pre-multiplied by W).
    winst : array-like or None
        Variables to pre-multiply by W to create instruments.

    Returns
    -------
    Z : ndarray
        Instrument matrix (constant column removed if present).
    """
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

    # Drop constant column if present
    if np.allclose(Z[:, 0], Z[0, 0]):
        Z = Z[:, 1:]

    return Z


# ---------------------------------------------------------------------------
# Base estimator class
# ---------------------------------------------------------------------------

class BaseSpatialQuantReg:
    """Abstract base class for spatial quantile regression estimators.

    Provides shared parameter handling, validation, and utility methods
    used by all areal spatial quantile regression models.

    Subclasses must define ``_param_names`` as a tuple of parameter names.
    """

    _param_names: tuple[str, ...] = ()

    def __repr__(self) -> str:
        params = ", ".join(
            f"{k}={v!r}" for k, v in self.get_params().items()
        )
        return f"{self.__class__.__name__}({params})"

    # ------------------------------------------------------------------
    # scikit-learn compatible parameter interface
    # ------------------------------------------------------------------

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            Ignored. Present for scikit-learn compatibility.

        Returns
        -------
        params : dict
        """
        return {name: getattr(self, name) for name in self._param_names}

    def set_params(self, **params: object) -> BaseSpatialQuantReg:
        """Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self
        """
        valid = self.get_params()
        for key, value in params.items():
            if key not in valid:
                raise ValueError(
                    f"Invalid parameter '{key}' for "
                    f"{self.__class__.__name__}."
                )
            setattr(self, key, value)
        return self

    # ------------------------------------------------------------------
    # Fitted state
    # ------------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        """Raise if the model has not been fitted."""
        if not hasattr(self, "coef_"):
            raise RuntimeError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this "
                "method."
            )

    # ------------------------------------------------------------------
    # Variable name helpers
    # ------------------------------------------------------------------

    def _build_var_names(
        self,
        k: int,
        suffix_names: list[str] | None = None,
    ) -> list[str]:
        """Build a list of variable names for the results table.

        Parameters
        ----------
        k : int
            Number of explanatory variables.
        suffix_names : list of str, optional
            Extra names to append (e.g. ``['WY']`` for SAR models).

        Returns
        -------
        names : list of str
        """
        if hasattr(self, "feature_names_in_"):
            names = list(self.feature_names_in_)
        else:
            names = [f"x{i + 1}" for i in range(k)]
        result = ["(Intercept)"] + names
        if suffix_names:
            result.extend(suffix_names)
        return result
