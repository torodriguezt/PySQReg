"""
pysqreg.plots
=============
Publication-quality visualisations for spatial quantile regression.

Requires **matplotlib** (install via ``pip install pysqreg[plot]``).

Public API
----------
plot_moran              Moran's I scatterplot with quadrant colouring
fit_quantile_process    Estimate the model across a grid of quantiles
plot_quantile_process   Coefficient-process ribbon plot
plot_rho_path           Chernozhukov-Hansen rho-selection path
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import norm
import statsmodels.api as sm

from .areal import QuantSAR, moran_test, ArrayLike, WeightMatrix


# ═══════════════════════════════════════════════════════════════════════════
#  Visual identity
# ═══════════════════════════════════════════════════════════════════════════

# Core palette — derived from the ColorBrewer RdBu diverging scheme,
# chosen for print safety, projected-slide legibility, and
# colorblind accessibility (deuteranopia / protanopia safe).
_BLUE = '#3B82F6'
_RED = '#EF4444'
_CORAL = '#F87171'
_TEAL = '#14B8A6'
_AMBER = '#F59E0B'
_LBLUE = '#93C5FD'
_PURPLE = '#A855F7'
_GREEN = '#22C55E'
_DARK = '#1E293B'
_GRAY = '#64748B'
_LGRAY = '#CBD5E1'

# Moran-scatter quadrant colours
_QUAD = {
    'HH': _CORAL,   # hot-spot cluster
    'LL': _BLUE,     # cold-spot cluster
    'LH': _LBLUE,   # spatial outlier (low surrounded by high)
    'HL': _AMBER,   # spatial outlier (high surrounded by low)
}

# Colour cycle for multi-series plots
_CYCLE = [_BLUE, _CORAL, _TEAL, _PURPLE, _AMBER, _GREEN]

# Matplotlib rcParams applied inside every public function so the
# caller's global state is never mutated.
_RC = {
    'figure.facecolor':     '#FAFBFC',
    'figure.dpi':           180,
    'savefig.dpi':          300,
    'axes.facecolor':       '#FFFFFF',
    'axes.edgecolor':       '#E2E8F0',
    'axes.linewidth':       0.7,
    'axes.grid':            True,
    'axes.grid.axis':       'both',
    'axes.titlesize':       15,
    'axes.titlepad':        16,
    'axes.labelsize':       12,
    'axes.labelcolor':      '#1E293B',
    'axes.labelpad':        10,
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'grid.color':           '#F1F5F9',
    'grid.linewidth':       0.5,
    'grid.alpha':           1.0,
    'xtick.labelsize':      10.5,
    'ytick.labelsize':      10.5,
    'xtick.color':          '#475569',
    'ytick.color':          '#475569',
    'xtick.direction':      'out',
    'ytick.direction':      'out',
    'xtick.major.size':     4,
    'ytick.major.size':     4,
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.major.pad':      7,
    'ytick.major.pad':      7,
    'legend.fontsize':      10,
    'legend.frameon':       True,
    'legend.framealpha':    0.95,
    'legend.edgecolor':     '#E2E8F0',
    'legend.fancybox':      True,
    'font.family':          'sans-serif',
    'font.size':            11.5,
    'text.color':           '#1E293B',
}


@contextmanager
def _style():
    """Apply the PySQReg visual style for the duration of a block."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from cycler import cycler

    rc = dict(_RC)
    rc['axes.prop_cycle'] = cycler('color', _CYCLE)
    with mpl.rc_context(rc):
        yield plt


# ═══════════════════════════════════════════════════════════════════════════
#  Moran scatterplot
# ═══════════════════════════════════════════════════════════════════════════

def plot_moran(
    x: ArrayLike,
    W: WeightMatrix,
    *,
    ax: object | None = None,
    annotate: bool = True,
    quadrant_labels: bool = True,
    figsize: tuple[float, float] = (7, 6.2),
    title: str | None = None,
) -> object:
    """Moran's I scatterplot with quadrant classification.

    Plots the spatial lag *Wz* against the standardised variable *z*,
    coloured by LISA-style quadrant (HH, LL, LH, HL).  The OLS line
    through the origin has slope equal to Moran's I (for a
    row-standardised *W*).

    Parameters
    ----------
    x : array-like of shape (n,)
        Variable to visualise (raw observations or residuals).
    W : array-like or sparse of shape (n, n)
        Row-standardised spatial weight matrix.
    ax : matplotlib Axes, optional
        Target axes.  A new figure is created when *None*.
    annotate : bool, default True
        Display a statistics box (I, E[I], z, p).
    quadrant_labels : bool, default True
        Print HH / LL / LH / HL watermarks in the corners.
    figsize : tuple, default (7, 6.2)
        Figure size when creating a new figure.
    title : str, optional
        Plot title.  Defaults to ``"Moran's I Scatterplot"``.

    Returns
    -------
    ax : matplotlib Axes
    """
    with _style() as plt:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # ── Standardise ────────────────────────────────────────────────
        x_arr = np.asarray(x, dtype=float).ravel()
        z = (x_arr - x_arr.mean()) / x_arr.std(ddof=0)

        if issparse(W):
            Wz = np.asarray(W @ z).ravel()
        else:
            Wz = np.asarray(W, dtype=float) @ z

        # ── Quadrant masks ─────────────────────────────────────────────
        masks = {
            'HH': (z >= 0) & (Wz >= 0),
            'LL': (z < 0) & (Wz < 0),
            'LH': (z < 0) & (Wz >= 0),
            'HL': (z >= 0) & (Wz < 0),
        }

        # ── Symmetric axis limits with padding ─────────────────────────
        pad = 1.15
        lim = max(np.abs(z).max(), np.abs(Wz).max()) * pad

        # ── Quadrant background fills ──────────────────────────────────
        quad_coords = {
            'HH': ([0, lim, lim, 0], [0, 0, lim, lim]),
            'LH': ([-lim, 0, 0, -lim], [0, 0, lim, lim]),
            'LL': ([-lim, 0, 0, -lim], [-lim, -lim, 0, 0]),
            'HL': ([0, lim, lim, 0], [-lim, -lim, 0, 0]),
        }
        for name, (xs, ys) in quad_coords.items():
            ax.fill(xs, ys, color=_QUAD[name], alpha=0.045, zorder=0)

        # ── Reference lines at zero ────────────────────────────────────
        ref_kw = dict(color='#C0C0C0', linewidth=0.7, zorder=1)
        ax.axhline(0, **ref_kw)
        ax.axvline(0, **ref_kw)

        # ── Scatter by quadrant ────────────────────────────────────────
        pt_kw = dict(s=55, edgecolors='white', linewidths=0.6,
                     alpha=0.82, zorder=3)
        for name, mask in masks.items():
            if mask.any():
                ax.scatter(z[mask], Wz[mask], c=_QUAD[name],
                           label=name, **pt_kw)

        # ── Regression line (slope = Moran's I for row-std W) ──────────
        slope = float(z @ Wz) / float(z @ z)
        xs_line = np.array([-lim, lim])
        ax.plot(xs_line, slope * xs_line, color=_DARK, linewidth=2.5,
                zorder=4, label=f'slope = {slope:.4f}',
                solid_capstyle='round')

        # ── Axes ───────────────────────────────────────────────────────
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel(r'$z$  (standardised variable)')
        ax.set_ylabel(r'$Wz$  (spatial lag)')
        ax.set_title(title or "Moran's I Scatterplot",
                     fontweight='semibold')

        # ── Quadrant watermarks ────────────────────────────────────────
        if quadrant_labels:
            kw = dict(fontsize=18, fontweight='bold', alpha=0.16,
                      ha='center', va='center', zorder=2)
            off = 0.55
            ax.text(lim * off, lim * off, 'HH',
                    color=_QUAD['HH'], **kw)
            ax.text(-lim * off, lim * off, 'LH',
                    color=_QUAD['LH'], **kw)
            ax.text(-lim * off, -lim * off, 'LL',
                    color=_QUAD['LL'], **kw)
            ax.text(lim * off, -lim * off, 'HL',
                    color=_QUAD['HL'], **kw)

        # ── Statistics box ─────────────────────────────────────────────
        if annotate:
            result = moran_test(x_arr, W)
            sig = ('***' if result.p_value < 0.001 else
                   '**' if result.p_value < 0.01 else
                   '*' if result.p_value < 0.05 else '')
            text = (
                f"Moran's $I$ = {result.I: .4f} {sig}\n"
                f"$E[I]$           = {result.EI: .4f}\n"
                f"$z$-score      = {result.z: .3f}\n"
                f"$p$-value      = {result.p_value: .4f}"
            )
            ax.text(
                0.03, 0.97, text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                          edgecolor='#E2E8F0', alpha=0.96),
            )

        # ── Legend ─────────────────────────────────────────────────────
        ax.legend(loc='lower right', fontsize=10, framealpha=0.95,
                  edgecolor='#E2E8F0', markerscale=1.3,
                  handletextpad=0.5, columnspacing=0.8,
                  borderpad=0.7)

        fig = ax.get_figure()
        fig.tight_layout()
        return ax


# ═══════════════════════════════════════════════════════════════════════════
#  Quantile process
# ═══════════════════════════════════════════════════════════════════════════

class QuantileProcessResult:
    """Container returned by :func:`fit_quantile_process`.

    Attributes
    ----------
    data : DataFrame
        Long-format table with columns ``tau``, ``variable``, ``coef``,
        ``se``, ``ci_lower``, ``ci_upper``.
    taus : ndarray
        Grid of quantile levels.
    ols : DataFrame or None
        Non-spatial OLS reference (``variable``, ``coef``, ``se``,
        ``ci_lower``, ``ci_upper``).
    """

    def __init__(self, data: pd.DataFrame, taus: np.ndarray,
                 ols: pd.DataFrame | None = None) -> None:
        self.data = data
        self.taus = taus
        self.ols = ols

    def plot(self, **kwargs: object) -> object:
        """Shortcut for ``plot_quantile_process(self, **kwargs)``."""
        return plot_quantile_process(self, **kwargs)

    def __repr__(self) -> str:
        n_taus = len(self.taus)
        n_vars = self.data['variable'].nunique()
        return (f"QuantileProcessResult("
                f"{n_taus} quantiles, {n_vars} variables)")


def fit_quantile_process(
    X: ArrayLike,
    y: ArrayLike,
    W: WeightMatrix,
    *,
    taus: npt.ArrayLike | None = None,
    method: str = 'two_stage',
    alpha: float = 0.05,
    include_ols: bool = True,
    verbose: int = 1,
    **model_kws: object,
) -> QuantileProcessResult:
    """Fit spatial quantile regression across a grid of quantiles.

    Parameters
    ----------
    X : array-like of shape (n, k)
        Explanatory variables (without intercept).
    y : array-like of shape (n,)
        Dependent variable.
    W : array-like or sparse of shape (n, n)
        Row-standardised spatial weight matrix.
    taus : array-like, optional
        Quantile grid.  Defaults to ``np.arange(0.05, 0.96, 0.05)``.
    method : {'two_stage', 'grid_search'}
        Estimation method passed to each :class:`QuantSAR` call.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    include_ols : bool, default True
        Also fit a non-spatial OLS as a visual reference.
    verbose : int, default 1
        Print progress (0 = silent).
    **model_kws
        Extra keyword arguments forwarded to :class:`QuantSAR`
        (e.g. ``nboot``, ``rhomat``, ``random_state``).

    Returns
    -------
    result : QuantileProcessResult
        Call ``result.plot()`` for a quick visualisation, or inspect
        ``result.data`` for the raw coefficient table.
    """
    if taus is None:
        taus = np.arange(0.05, 0.96, 0.05)
    taus = np.asarray(taus, dtype=float)

    z_crit = norm.ppf(1.0 - alpha / 2.0)

    rows = []
    for i, tau in enumerate(taus):
        if verbose:
            print(f"\r  Fitting quantile {tau:.2f}  "
                  f"[{i + 1}/{len(taus)}]", end='', flush=True)

        model = QuantSAR(tau=float(tau), method=method, **model_kws)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(X, y, W)
        except Exception as exc:
            if verbose:
                print(f"  — skipped ({exc})")
            continue

        var_names = list(model.results_.index)
        coefs = np.concatenate(
            [[model.intercept_], model.coef_, [model.rho_]]
        )
        ses = model.se_

        for j, name in enumerate(var_names):
            rows.append({
                'tau': float(tau),
                'variable': name,
                'coef': coefs[j],
                'se': ses[j],
                'ci_lower': coefs[j] - z_crit * ses[j],
                'ci_upper': coefs[j] + z_crit * ses[j],
            })

    if verbose:
        print()

    data = pd.DataFrame(rows)

    # ── Non-spatial OLS reference ──────────────────────────────────────
    ols_df = None
    if include_ols:
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        y_arr = np.asarray(y, dtype=float).ravel()

        ols_fit = sm.OLS(y_arr, sm.add_constant(X_arr)).fit()

        if isinstance(X, pd.DataFrame):
            ols_names = ['(Intercept)'] + list(X.columns)
        else:
            k = X_arr.shape[1]
            ols_names = ['(Intercept)'] + [f'x{j + 1}' for j in range(k)]

        ols_rows = []
        for j, name in enumerate(ols_names):
            ols_rows.append({
                'variable': name,
                'coef': ols_fit.params[j],
                'se': ols_fit.bse[j],
                'ci_lower': ols_fit.params[j] - z_crit * ols_fit.bse[j],
                'ci_upper': ols_fit.params[j] + z_crit * ols_fit.bse[j],
            })
        ols_df = pd.DataFrame(ols_rows)

    return QuantileProcessResult(data=data, taus=taus, ols=ols_df)


def plot_quantile_process(
    result: QuantileProcessResult,
    *,
    variables: list[str] | str | None = None,
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
) -> object:
    """Coefficient-process ribbon plot across quantiles.

    Displays how each coefficient evolves over quantile levels, with
    layered confidence bands (inner ±1 SE, outer 95 % CI) and an
    optional OLS reference line.

    Parameters
    ----------
    result : QuantileProcessResult
        Output of :func:`fit_quantile_process`.
    variables : list of str, ``'all'``, or None
        Which variables to plot.  *None* selects covariates only
        (excludes ``'(Intercept)'`` and ``'WY'``).  ``'all'`` plots
        every estimated parameter.
    figsize : tuple, optional
        Figure size.  Computed automatically when *None*.
    title : str, optional
        Figure-level super-title.

    Returns
    -------
    fig : matplotlib Figure
    """
    with _style() as plt:
        data = result.data
        ols = result.ols

        # ── Variable selection ─────────────────────────────────────────
        if variables is None:
            all_vars = list(data['variable'].unique())
            variables = [v for v in all_vars
                         if v not in ('(Intercept)', 'WY')]
        elif variables == 'all':
            variables = list(data['variable'].unique())

        n_vars = len(variables)
        if n_vars == 0:
            raise ValueError("No variables to plot.")

        # ── Subplot layout ─────────────────────────────────────────────
        ncols = min(n_vars, 3)
        nrows = int(np.ceil(n_vars / ncols))
        if figsize is None:
            figsize = (5.2 * ncols, 4.0 * nrows + 0.5)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                                 squeeze=False, sharex=True)

        for idx, var_name in enumerate(variables):
            row, col = divmod(idx, ncols)
            ax = axes[row, col]

            subset = (data[data['variable'] == var_name]
                      .sort_values('tau'))
            taus_v = subset['tau'].values
            coefs = subset['coef'].values
            ses = subset['se'].values

            # Confidence bands — two layers for a gradient effect ───────
            ax.fill_between(
                taus_v,
                coefs - 1.96 * ses, coefs + 1.96 * ses,
                color=_BLUE, alpha=0.10, linewidth=0, zorder=1,
                label='95 % CI',
            )
            ax.fill_between(
                taus_v,
                coefs - ses, coefs + ses,
                color=_BLUE, alpha=0.18, linewidth=0, zorder=1,
            )

            # Coefficient line ──────────────────────────────────────────
            ax.plot(taus_v, coefs, color=_BLUE, linewidth=2.5,
                    zorder=3, label='Spatial QR',
                    solid_capstyle='round')

            # OLS reference ─────────────────────────────────────────────
            if ols is not None:
                ols_match = ols[ols['variable'] == var_name]
                if len(ols_match):
                    ols_row = ols_match.iloc[0]
                    ax.axhline(ols_row['coef'], color=_GRAY,
                               linewidth=1.3, linestyle='--',
                               zorder=2, label='OLS')
                    ax.axhspan(ols_row['ci_lower'], ols_row['ci_upper'],
                               color=_GRAY, alpha=0.07, zorder=0)

            # Zero reference ────────────────────────────────────────────
            ax.axhline(0, color='#CCCCCC', linewidth=0.6, zorder=0)

            # Key quantile guides ───────────────────────────────────────
            for q in (0.25, 0.50, 0.75):
                if taus_v.min() <= q <= taus_v.max():
                    ax.axvline(q, color='#E0E0E0', linewidth=0.5,
                               linestyle=':', zorder=0)

            # Formatting ────────────────────────────────────────────────
            ax.set_title(var_name, fontweight='bold', fontsize=14)
            if row == nrows - 1:
                ax.set_xlabel(r'$\tau$  (quantile)')
            if col == 0:
                ax.set_ylabel('Coefficient')
            ax.set_xlim(taus_v.min(), taus_v.max())

        # ── Hide unused subplots ───────────────────────────────────────
        for idx in range(n_vars, nrows * ncols):
            r, c = divmod(idx, ncols)
            axes[r, c].set_visible(False)

        # ── Shared legend ──────────────────────────────────────────────
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles, labels,
                loc='lower center', ncol=len(handles),
                fontsize=10.5, frameon=True, framealpha=0.95,
                edgecolor='#E2E8F0',
                bbox_to_anchor=(0.5, -0.01),
                borderpad=0.8,
            )

        # ── Super-title ────────────────────────────────────────────────
        if title:
            fig.suptitle(title, fontweight='bold', fontsize=16, y=1.02)

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Rho path
# ═══════════════════════════════════════════════════════════════════════════

def plot_rho_path(
    model: QuantSAR,
    *,
    ax: object | None = None,
    figsize: tuple[float, float] = (7.5, 4.5),
    title: str | None = None,
) -> object:
    """Chernozhukov-Hansen rho-selection path.

    Visualises *alpha(rho)* — the coefficient on the predicted spatial
    lag for each candidate rho in the grid — together with the optimal
    rho that minimises |alpha(rho)|.

    Parameters
    ----------
    model : QuantSAR
        A fitted model with ``method='grid_search'``.
    ax : matplotlib Axes, optional
        Target axes.  A new figure is created when *None*.
    figsize : tuple, default (7.5, 4.5)
        Figure size when creating a new figure.
    title : str, optional
        Plot title.

    Returns
    -------
    ax : matplotlib Axes

    Raises
    ------
    ValueError
        If *model* was not fitted with ``method='grid_search'``.
    """
    if not hasattr(model, 'rho_path_'):
        raise ValueError(
            "plot_rho_path requires a QuantSAR fitted with "
            "method='grid_search'."
        )

    with _style() as plt:
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        grid = model.rho_grid_
        path = model.rho_path_
        rho_star = model.rho_
        j_opt = np.argmin(np.abs(path))
        alpha_star = path[j_opt]

        # ── Fill area between curve and zero ───────────────────────────
        ax.fill_between(grid, path, 0, color=_BLUE, alpha=0.08,
                        zorder=1)

        # ── Main curve ─────────────────────────────────────────────────
        ax.plot(grid, path, color=_BLUE, linewidth=2.8, zorder=3,
                solid_capstyle='round')

        # ── Zero reference ─────────────────────────────────────────────
        ax.axhline(0, color=_GRAY, linewidth=0.9, linestyle='--',
                   zorder=2)

        # ── Optimal rho marker ─────────────────────────────────────────
        ax.axvline(rho_star, color=_CORAL, linewidth=1.4,
                   linestyle='--', alpha=0.85, zorder=2)
        ax.scatter([rho_star], [alpha_star], color=_CORAL, s=80,
                   edgecolors='white', linewidths=1.8, zorder=5)

        # ── Annotation ─────────────────────────────────────────────────
        y_range = path.max() - path.min()
        x_range = grid[-1] - grid[0]
        ax.annotate(
            f'$\\hat{{\\rho}}$ = {rho_star:.3f}',
            xy=(rho_star, alpha_star),
            xytext=(rho_star + x_range * 0.12,
                    alpha_star + y_range * 0.20),
            fontsize=13, fontweight='bold', color=_CORAL,
            arrowprops=dict(arrowstyle='->', color=_CORAL, lw=1.5,
                            connectionstyle='arc3,rad=0.15'),
            zorder=6,
        )

        # ── Axes ───────────────────────────────────────────────────────
        ax.set_xlabel(r'$\rho$')
        ax.set_ylabel(r'$\alpha(\rho)$')
        ax.set_title(
            title or r'Chernozhukov-Hansen $\rho$ Path',
            fontweight='semibold',
        )

        fig = ax.get_figure()
        fig.tight_layout()
        return ax
