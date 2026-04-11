"""pysqreg.areal -- Spatial quantile regression for areal (lattice) data.

This sub-package provides cross-sectional spatial quantile regression
models for areal / lattice data, along with diagnostic tests.

Models
------
QuantSAR : Spatial Autoregressive (SAR) quantile regression.
QuantSLX : Spatial Lag of X (SLX) quantile regression.
QuantSDM : Spatial Durbin Model (SDM) quantile regression.

Diagnostics
-----------
moran_test : Moran's I test for spatial autocorrelation.
MoranResult : Results container for the Moran's I test.

Utilities
---------
ArrayLike : Type alias for array-like inputs.
WeightMatrix : Type alias for weight matrix inputs.
"""

from ._base import ArrayLike, WeightMatrix
from ._diagnostics import MoranResult, moran_test
from .sar import QuantSAR
from .sdm import QuantSDM
from .slx import QuantSLX

__all__ = [
    "QuantSAR",
    "QuantSLX",
    "QuantSDM",
    "moran_test",
    "MoranResult",
    "ArrayLike",
    "WeightMatrix",
]
