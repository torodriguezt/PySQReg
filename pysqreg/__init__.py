"""pysqreg -- Spatial Quantile Regression for areal (lattice) data.

Provides instrumental-variable quantile regression estimators for
spatial and spatio-temporal models on areal / lattice data.

Cross-sectional (areal) models
------------------------------
>>> from pysqreg import QuantSAR
>>> model = QuantSAR(tau=0.5, method='two_stage')
>>> model.fit(X, y, W)

Spatio-temporal (panel) models
------------------------------
>>> from pysqreg import DynQuantSAR
>>> model = DynQuantSAR(tau=0.5, method='two_stage')
>>> model.fit(X, y, W, n_units=50, n_periods=10)
"""

# --- Cross-sectional (areal) models ---
from .areal import QuantSAR, QuantSLX, QuantSDM, moran_test, MoranResult

# --- Spatio-temporal (panel) models ---
from .spatiotemporal import DynQuantSAR, DynQuantSLX, DynQuantSDM

# --- Visualisation ---
from .plots import (
    plot_moran,
    fit_quantile_process,
    plot_quantile_process,
    plot_rho_path,
    QuantileProcessResult,
)

__all__ = [
    # Areal (cross-sectional)
    "QuantSAR",
    "QuantSLX",
    "QuantSDM",
    "moran_test",
    "MoranResult",
    # Spatio-temporal (panel)
    "DynQuantSAR",
    "DynQuantSLX",
    "DynQuantSDM",
    # Visualisation
    "plot_moran",
    "fit_quantile_process",
    "plot_quantile_process",
    "plot_rho_path",
    "QuantileProcessResult",
]
__version__ = "0.2.0"
