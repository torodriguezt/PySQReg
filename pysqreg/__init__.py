"""pysqreg — Spatial Quantile Regression for areal (lattice) data.

Provides instrumental-variable quantile regression estimators for the
SAR model y = rho * W @ y + X @ beta + u on areal / lattice spatial data.

Quick start
-----------
>>> from pysqreg import QuantSAR
>>> model = QuantSAR(tau=0.5, method='two_stage')
>>> model.fit(X, y, W)
>>> print(model.coef_)
>>> model.summary()
"""

from .areal import QuantSAR, moran_test, MoranResult
from .plots import (
    plot_moran,
    fit_quantile_process,
    plot_quantile_process,
    plot_rho_path,
    QuantileProcessResult,
)

__all__ = [
    "QuantSAR",
    "moran_test",
    "MoranResult",
    "plot_moran",
    "fit_quantile_process",
    "plot_quantile_process",
    "plot_rho_path",
    "QuantileProcessResult",
]
__version__ = "0.1.0"
