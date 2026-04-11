"""Moran's I test for spatial autocorrelation on areal (lattice) data."""

from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import issparse
from scipy.stats import norm

from ._base import ArrayLike, WeightMatrix


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

    def __init__(
        self,
        I: float,
        EI: float,
        VI: float,
        z: float,
        p_value: float,
        n: int,
        alternative: str,
        assumption: str,
    ) -> None:
        self.I = I
        self.EI = EI
        self.VI = VI
        self.z = z
        self.p_value = p_value
        self.n = n
        self.alternative = alternative
        self.assumption = assumption

    def __repr__(self) -> str:
        return (
            f"MoranResult(I={self.I:.6f}, E[I]={self.EI:.6f}, "
            f"z={self.z:.4f}, p={self.p_value:.4f})"
        )

    def summary(self) -> None:
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


def moran_test(
    x: ArrayLike,
    W: WeightMatrix,
    alternative: str = "two-sided",
    assumption: str = "randomization",
) -> MoranResult:
    """Moran's I test for spatial autocorrelation in lattice data.

    Computes the Moran's I statistic (Cliff & Ord, 1981) and tests for
    spatial autocorrelation.  This test can be applied to raw variables
    to detect spatial structure, or to model residuals to check whether
    a fitted model has captured all spatial dependence.

    Parameters
    ----------
    x : array-like of shape (n,)
        Variable to test -- either raw observations or model residuals.
    W : array-like or sparse of shape (n, n)
        Spatial weight matrix (typically row-standardised).
    alternative : {'two-sided', 'greater', 'less'}
        Alternative hypothesis.

        * ``'two-sided'`` -- spatial autocorrelation (positive or negative).
        * ``'greater'`` -- positive spatial autocorrelation (clustering
          of similar values).
        * ``'less'`` -- negative spatial autocorrelation (dispersion /
          checkerboard pattern).
    assumption : {'randomization', 'normality'}
        Distributional assumption used to compute Var[I].

        * ``'randomization'`` -- distribution-free variance that adjusts
          for the sample kurtosis (more robust; default).
        * ``'normality'`` -- assumes the data are normally distributed.

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

    if assumption == "normality":
        VI = (
            (n ** 2 * S1 - n * S2 + 3 * S0sq) / (S0sq * (n ** 2 - 1))
            - EI ** 2
        )
    elif assumption == "randomization":
        m2 = z2 / n
        m4 = float((z ** 4).sum()) / n
        b2 = m4 / (m2 ** 2)  # kurtosis

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

    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - norm.cdf(np.abs(z_score)))
    elif alternative == "greater":
        p_value = 1.0 - norm.cdf(z_score)
    elif alternative == "less":
        p_value = norm.cdf(z_score)
    else:
        raise ValueError(
            f"alternative must be 'two-sided', 'greater', or 'less', "
            f"got '{alternative}'."
        )

    return MoranResult(
        I=I,
        EI=EI,
        VI=VI,
        z=z_score,
        p_value=p_value,
        n=n,
        alternative=alternative,
        assumption=assumption,
    )
