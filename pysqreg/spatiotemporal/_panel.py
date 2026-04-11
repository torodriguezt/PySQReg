"""Panel data utilities for spatio-temporal quantile regression.

Handles the transformation of stacked panel data into the structures
needed for estimation: temporal lags, spatial-temporal lags, fixed-
effect dummies, and temporal instrument matrices.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse

from ..areal._base import ArrayLike, WeightMatrix


# ---------------------------------------------------------------------------
# Panel structure
# ---------------------------------------------------------------------------

class PanelStructure:
    """Organise stacked panel data into (N, T) structure.

    The data must be provided as long-format arrays of length N*T,
    sorted by unit and then by time within each unit.  The class
    validates this assumption and exposes views that make it easy
    to extract temporal lags, apply spatial weight matrices per
    period, and construct instrument sets.

    Parameters
    ----------
    n_units : int
        Number of cross-sectional units (N).
    n_periods : int
        Number of time periods (T).
    unit_ids : array-like of shape (N*T,), optional
        Unit identifiers.  When provided, the class validates that
        there are exactly *n_units* unique IDs, each with *n_periods*
        observations.  Data is re-sorted by (unit, time) internally.
    time_ids : array-like of shape (N*T,), optional
        Time identifiers.  Required together with *unit_ids*.
    """

    def __init__(
        self,
        n_units: int,
        n_periods: int,
        unit_ids: ArrayLike | None = None,
        time_ids: ArrayLike | None = None,
    ) -> None:
        if n_units < 2:
            raise ValueError("Need at least 2 cross-sectional units.")
        if n_periods < 2:
            raise ValueError("Need at least 2 time periods.")

        self.n_units = n_units
        self.n_periods = n_periods
        self.n_total = n_units * n_periods

        # Sort index (identity when no IDs are given)
        if unit_ids is not None and time_ids is not None:
            self._sort_idx = self._build_sort_index(unit_ids, time_ids)
        else:
            self._sort_idx = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def sort_data(self, arr: np.ndarray) -> np.ndarray:
        """Re-order an array to (unit, time) stacking if IDs were given."""
        if self._sort_idx is not None:
            return arr[self._sort_idx]
        return arr

    def reshape_to_panel(self, arr: np.ndarray) -> np.ndarray:
        """Reshape a stacked (N*T,) or (N*T, k) array into (N, T, ...).

        The first axis becomes units, the second becomes time.
        """
        if arr.ndim == 1:
            return arr.reshape(self.n_units, self.n_periods)
        return arr.reshape(self.n_units, self.n_periods, -1)

    def flatten_panel(self, panel: np.ndarray) -> np.ndarray:
        """Flatten an (N, T, ...) panel back to (N*T, ...)."""
        if panel.ndim == 2:
            return panel.ravel()
        return panel.reshape(self.n_total, -1)

    def validate_length(self, arr: np.ndarray, name: str) -> None:
        """Check that an array has N*T rows."""
        n = arr.shape[0] if arr.ndim > 1 else len(arr)
        if n != self.n_total:
            raise ValueError(
                f"{name} has {n} observations but expected "
                f"N*T = {self.n_units}*{self.n_periods} = {self.n_total}."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_sort_index(
        self,
        unit_ids: ArrayLike,
        time_ids: ArrayLike,
    ) -> np.ndarray:
        """Build a permutation index that sorts data by (unit, time)."""
        unit_ids = np.asarray(unit_ids)
        time_ids = np.asarray(time_ids)

        if len(unit_ids) != self.n_total:
            raise ValueError(
                f"unit_ids has {len(unit_ids)} elements but expected "
                f"N*T = {self.n_total}."
            )

        unique_units = np.unique(unit_ids)
        unique_times = np.unique(time_ids)

        if len(unique_units) != self.n_units:
            raise ValueError(
                f"Expected {self.n_units} unique units, "
                f"found {len(unique_units)}."
            )
        if len(unique_times) != self.n_periods:
            raise ValueError(
                f"Expected {self.n_periods} unique time periods, "
                f"found {len(unique_times)}."
            )

        # Lexicographic sort by (unit, time)
        return np.lexsort((time_ids, unit_ids))


# ---------------------------------------------------------------------------
# Temporal lag construction
# ---------------------------------------------------------------------------

def build_temporal_lag(
    y: np.ndarray,
    panel: PanelStructure,
    lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct temporal lag y_{i, t-lag} from stacked panel data.

    Parameters
    ----------
    y : ndarray of shape (N*T,)
        Stacked dependent variable, ordered by (unit, time).
    panel : PanelStructure
        Panel dimensions and structure.
    lag : int
        Number of periods to lag (default 1).

    Returns
    -------
    y_lagged : ndarray of shape (N*(T-lag),)
        Lagged values, with the first *lag* periods dropped per unit.
    valid_mask : ndarray of shape (N*T,), dtype bool
        Boolean mask indicating which rows in the original stacked
        data have a valid lag (True for t >= lag+1).
    """
    if lag < 1:
        raise ValueError("lag must be >= 1.")
    if lag >= panel.n_periods:
        raise ValueError(
            f"lag={lag} must be < n_periods={panel.n_periods}."
        )

    y_panel = panel.reshape_to_panel(y)  # (N, T)

    # y_{i, t-lag} for t = lag, lag+1, ..., T-1
    y_lagged_panel = y_panel[:, :-lag]  # (N, T-lag)
    # corresponding y values are at t = lag, ..., T-1
    # but the lagged value for observation at t is y_{t-lag}

    # Build mask: valid observations are t >= lag (0-indexed)
    valid_panel = np.zeros((panel.n_units, panel.n_periods), dtype=bool)
    valid_panel[:, lag:] = True
    valid_mask = valid_panel.ravel()

    y_lagged = y_lagged_panel.ravel()
    return y_lagged, valid_mask


def build_spatiotemporal_lag(
    y: np.ndarray,
    W: WeightMatrix,
    panel: PanelStructure,
    lag: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct spatio-temporal lag W @ y_{i, t-lag}.

    Parameters
    ----------
    y : ndarray of shape (N*T,)
        Stacked dependent variable.
    W : array-like or sparse of shape (N, N)
        Spatial weight matrix (cross-sectional, applied per period).
    panel : PanelStructure
        Panel structure.
    lag : int
        Temporal lag.

    Returns
    -------
    wy_lagged : ndarray of shape (N*(T-lag),)
        Spatially weighted temporal lag.
    valid_mask : ndarray of shape (N*T,), dtype bool
        Boolean mask for valid observations.
    """
    y_panel = panel.reshape_to_panel(y)  # (N, T)

    # Apply W to each lagged period: W @ y_{:, t-lag}
    wy_lagged_periods = []
    for t in range(panel.n_periods - lag):
        y_t = y_panel[:, t]  # y at period t (this is the lag for t+lag)
        wy_lagged_periods.append(W @ y_t)

    wy_lagged = np.column_stack(wy_lagged_periods).ravel(order="F")

    # Same valid mask as temporal lag
    valid_panel = np.zeros((panel.n_units, panel.n_periods), dtype=bool)
    valid_panel[:, lag:] = True
    valid_mask = valid_panel.ravel()

    return wy_lagged, valid_mask


def build_spatial_lag_panel(
    y: np.ndarray,
    W: WeightMatrix,
    panel: PanelStructure,
) -> np.ndarray:
    """Compute contemporaneous spatial lag W @ y_{it} for panel data.

    Applies W within each time period independently.

    Parameters
    ----------
    y : ndarray of shape (N*T,)
    W : array-like or sparse of shape (N, N)
    panel : PanelStructure

    Returns
    -------
    wy : ndarray of shape (N*T,)
    """
    y_panel = panel.reshape_to_panel(y)  # (N, T)
    wy_panel = np.zeros_like(y_panel)
    for t in range(panel.n_periods):
        wy_panel[:, t] = W @ y_panel[:, t]
    return wy_panel.ravel()


# ---------------------------------------------------------------------------
# Subsetting helpers
# ---------------------------------------------------------------------------

def subset_to_valid(
    valid_mask: np.ndarray,
    *arrays: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Subset stacked arrays to rows where valid_mask is True.

    Parameters
    ----------
    valid_mask : ndarray of shape (N*T,), dtype bool
    *arrays : ndarrays of shape (N*T, ...) or (N*T,)

    Returns
    -------
    subsetted : tuple of ndarrays
    """
    return tuple(arr[valid_mask] for arr in arrays)


# ---------------------------------------------------------------------------
# Fixed-effects dummy construction
# ---------------------------------------------------------------------------

def build_fixed_effects_dummies(panel: PanelStructure) -> np.ndarray:
    """Build unit fixed-effect dummy matrix.

    Parameters
    ----------
    panel : PanelStructure

    Returns
    -------
    D : ndarray of shape (N*T, N)
        Dummy matrix where D[i*T + t, i] = 1.
    """
    D = np.zeros((panel.n_total, panel.n_units))
    for i in range(panel.n_units):
        start = i * panel.n_periods
        end = start + panel.n_periods
        D[start:end, i] = 1.0
    return D


# ---------------------------------------------------------------------------
# Temporal instruments
# ---------------------------------------------------------------------------

def build_temporal_instruments(
    y: np.ndarray,
    panel: PanelStructure,
    valid_mask: np.ndarray,
    max_lag: int = 2,
) -> np.ndarray:
    """Build Arellano-Bond style temporal instruments.

    Uses deeper lags y_{i, t-2}, y_{i, t-3}, ... as instruments for
    the endogenous temporal lag y_{i, t-1}.

    Parameters
    ----------
    y : ndarray of shape (N*T,)
        Full stacked dependent variable.
    panel : PanelStructure
        Panel structure.
    valid_mask : ndarray of shape (N*T,), dtype bool
        Mask defining the estimation sample (t >= 1 typically).
    max_lag : int
        Maximum lag depth for instruments (default 2 gives y_{t-2}).

    Returns
    -------
    Z_temporal : ndarray of shape (n_valid, n_instruments)
        Temporal instrument columns.  Missing lags are filled with 0.
    """
    y_panel = panel.reshape_to_panel(y)  # (N, T)
    n_valid = valid_mask.sum()

    # Determine which periods are in the valid sample
    valid_panel = valid_mask.reshape(panel.n_units, panel.n_periods)
    # first valid period index (same for all units)
    first_valid_t = np.argmax(valid_panel[0])

    instruments = []
    for lag_depth in range(2, max_lag + 1):
        col = np.zeros(n_valid)
        idx = 0
        for i in range(panel.n_units):
            for t in range(first_valid_t, panel.n_periods):
                t_src = t - lag_depth
                if t_src >= 0:
                    col[idx] = y_panel[i, t_src]
                # else stays 0 (missing instrument)
                idx += 1
        instruments.append(col)

    if not instruments:
        raise ValueError(
            f"max_lag={max_lag} must be >= 2 to produce at least one "
            "temporal instrument."
        )

    return np.column_stack(instruments)
