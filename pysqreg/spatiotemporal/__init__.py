"""pysqreg.spatiotemporal -- Dynamic spatial quantile regression for panel data.

This sub-package provides spatio-temporal quantile regression models
for areal (lattice) panel data, extending the cross-sectional models
in :mod:`pysqreg.areal` with temporal dynamics and fixed effects.

Models
------
DynQuantSAR : Dynamic SAR quantile regression (panel).
DynQuantSLX : Dynamic SLX quantile regression (panel).
DynQuantSDM : Dynamic Spatial Durbin quantile regression (panel).

Panel utilities
---------------
PanelStructure : Panel data organisation and validation.
"""

from ._panel import PanelStructure
from .sar import DynQuantSAR
from .sdm import DynQuantSDM
from .slx import DynQuantSLX

__all__ = [
    "DynQuantSAR",
    "DynQuantSLX",
    "DynQuantSDM",
    "PanelStructure",
]
