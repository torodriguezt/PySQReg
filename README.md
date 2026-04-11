# PySQReg

Spatial and Spatio-Temporal Quantile Regression for areal (lattice) data in Python.

PySQReg provides a family of quantile regression estimators for spatial and dynamic spatial panel models on areal / lattice data with row-standardized weight matrices:

### Cross-sectional (areal) models

| Model | Specification | Class |
|-------|--------------|-------|
| **SAR** | $y = \rho W y + X \beta + u$ | `QuantSAR` |
| **SLX** | $y = X \beta + W X \theta + u$ | `QuantSLX` |
| **SDM** | $y = \rho W y + X \beta + W X \theta + u$ | `QuantSDM` |

### Spatio-temporal (panel) models

| Model | Specification | Class |
|-------|--------------|-------|
| **Dynamic SLX** | $y_{it} = X_{it} \beta + W X_{it} \theta + \gamma y_{i,t-1} + \alpha_i + u_{it}$ | `DynQuantSLX` |
| **Dynamic SAR** | $y_{it} = \rho W y_{it} + X_{it} \beta + \gamma y_{i,t-1} + \alpha_i + u_{it}$ | `DynQuantSAR` |
| **Dynamic SDM** | $y_{it} = \rho W y_{it} + X_{it} \beta + W X_{it} \theta + \gamma y_{i,t-1} + \delta W y_{i,t-1} + \alpha_i + u_{it}$ | `DynQuantSDM` |

## Why spatial quantile regression?

Standard spatial regression (e.g., spatial lag models estimated by ML or 2SLS) focuses on the **conditional mean**. PySQReg lets you estimate effects across the **entire conditional distribution** -- revealing how spatial spillovers differ at the tails versus the center, which is critical for understanding inequality, risk, and heterogeneous policy impacts.

The spatio-temporal extension adds **temporal dynamics** and **unit fixed effects**, enabling analysis of panel data where spatial dependence and persistence over time coexist.

## Features

- **Six model specifications:** SAR, SLX, SDM (cross-sectional) and their dynamic panel counterparts
- **Two IV estimation methods:**
  - Kim & Muller (2004) two-stage IV quantile regression with bootstrap inference
  - Chernozhukov & Hansen (2006) IV quantile regression via grid search
- **Panel data support:**
  - Koenker (2004) $\ell_1$-penalized fixed effects (solves the incidental parameters problem)
  - Block bootstrap preserving within-unit temporal structure
  - Arellano-Bond style temporal instruments
- **Spatial impacts** (LeSage & Pace): direct, indirect (spillover), and total effects
- **Moran's I test** for spatial autocorrelation in variables or residuals
- **Publication-quality visualizations:** Moran scatterplot, quantile-process ribbon plots, rho-selection path
- **scikit-learn compatible** API: `fit()`, `predict()`, `get_params()`, `set_params()`
- Supports **sparse weight matrices** and **large datasets**

## Installation

```bash
pip install pysqreg
```

With visualization support:

```bash
pip install pysqreg[plot]
```

Or install from source:

```bash
git clone https://github.com/torodriguezt/PySQReg.git
cd PySQReg
pip install -e ".[plot]"
```

## Quick start

### Cross-sectional model

```python
import numpy as np
import pandas as pd
from pysqreg import QuantSAR

# Load data
data = pd.read_csv("data_spatial.csv")
W = pd.read_csv("w_matrix.csv", index_col=0).values

y = data["y"].values
X = data[["x1"]]

# Fit at the median (tau = 0.5)
model = QuantSAR(tau=0.5, method="two_stage", nboot=200)
model.fit(X, y, W)
model.summary()

# Access results
print(model.rho_)        # spatial autoregressive parameter
print(model.coef_)       # covariate coefficients
print(model.impacts_)    # direct, indirect, and total effects
```

### Spatio-temporal (panel) model

```python
from pysqreg import DynQuantSAR

# Panel data: N units observed over T periods, stacked as (N*T, k)
model = DynQuantSAR(
    tau=0.5,
    method="two_stage",
    fixed_effects="penalized",  # Koenker (2004) L1-penalized FE
    nboot=200,
)
model.fit(X_panel, y_panel, W, n_units=50, n_periods=10)
model.summary()

# Access results
print(model.rho_)      # spatial autoregressive parameter
print(model.gamma_)    # temporal autoregressive parameter
print(model.coef_)     # covariate coefficients
print(model.impacts_)  # spatial impacts
```

For a complete walkthrough see the [tutorial notebook](examples/tutorial.ipynb).

## Cross-sectional models

### `QuantSAR` -- Spatial Autoregressive Model

Estimates the SAR model at a given quantile $\tau$ using IV approaches:

$$y = \rho W y + X \beta + u, \qquad Q_\tau(u \mid X) = 0$$

The endogenous spatial lag $Wy$ is instrumented using spatial lags of the exogenous variables ($WX$, $W^2 X$).

### `QuantSLX` -- Spatial Lag of X

$$y = X \beta + W X \theta + u, \qquad Q_\tau(u \mid X) = 0$$

A purely exogenous model -- no endogenous spatial lag -- so standard quantile regression applies directly. Spatial spillovers are captured by $\theta$, the coefficients on the $WX$ terms.

### `QuantSDM` -- Spatial Durbin Model

$$y = \rho W y + X \beta + W X \theta + u, \qquad Q_\tau(u \mid X) = 0$$

Nests both `QuantSAR` ($\theta = 0$) and `QuantSLX` ($\rho = 0$) as special cases, combining the endogenous spatial lag with exogenous spillover terms.

## Spatio-temporal models

All panel models accept data in long (stacked) format with `n_units` and `n_periods` arguments, or `unit_ids` / `time_ids` for automatic sorting.

### `DynQuantSLX` -- Dynamic Spatial Lag of X

$$y_{it} = X_{it} \beta + W X_{it} \theta + \gamma \, y_{i,t-1} + \alpha_i + u_{it}$$

No endogenous spatial lag, so only the temporal lag requires attention. Fixed effects $\alpha_i$ are handled via Koenker (2004) penalized QR.

### `DynQuantSAR` -- Dynamic Spatial Autoregressive Model

$$y_{it} = \rho \, W y_{it} + X_{it} \beta + \gamma \, y_{i,t-1} + \alpha_i + u_{it}$$

Both the contemporaneous spatial lag $Wy_{it}$ and the temporal lag $y_{i,t-1}$ are endogenous. Spatial instruments ($WX$, $W^2 X$) and temporal instruments ($y_{i,t-2}$, $y_{i,t-3}$, ...) are combined in the IV procedure.

### `DynQuantSDM` -- Dynamic Spatial Durbin Model

$$y_{it} = \rho \, W y_{it} + X_{it} \beta + W X_{it} \theta + \gamma \, y_{i,t-1} + \delta \, W y_{i,t-1} + \alpha_i + u_{it}$$

The most general model -- nests all others as special cases. Includes the spatio-temporal lag $W y_{i,t-1}$ capturing lagged spatial diffusion.

### Fixed effects strategies

| Strategy | Argument | Description |
|----------|----------|-------------|
| **Penalized** (default) | `fixed_effects='penalized'` | Koenker (2004) $\ell_1$-penalized FE -- solves the incidental parameters problem in short panels |
| **Dummies** | `fixed_effects='dummies'` | Raw FE dummies (biased for small $T$, useful for long panels) |
| **None** | `fixed_effects='none'` | Pooled estimation, no unit effects |

The penalty parameter $\lambda$ can be set manually or selected automatically via BIC.

## Estimation methods

### Kim & Muller (2004) -- `method='two_stage'`

A two-stage instrumental-variable quantile regression:

1. **Stage 1:** Quantile regression of $Wy$ on instruments $Z = [X, WX]$ to obtain $\widehat{Wy}$.
2. **Stage 2:** Quantile regression of $y$ on $[X, \widehat{Wy}]$ to obtain $(\hat{\beta}, \hat{\rho})$.

Inference is based on the bootstrap (block bootstrap for panel models, resampling whole units to preserve temporal dependence).

### Chernozhukov & Hansen (2006) -- `method='grid_search'`

An IV quantile regression that searches over a grid of candidate $\rho$ values. For each $\rho$, the model transforms the dependent variable as $\tilde{y} = y - \rho \, Wy$ and estimates a quantile regression including $\widehat{Wy}$ as an additional regressor. The optimal $\hat{\rho}$ is the value that drives the coefficient on $\widehat{Wy}$ closest to zero:

$$\hat{\rho} = \arg\min_{\rho \in \mathcal{G}} \left| \hat{\alpha}(\rho) \right|$$

Inference uses a sandwich variance estimator.

## Spatial impacts

For models with an endogenous spatial lag ($\rho \neq 0$), the marginal effect of $X_j$ propagates through the spatial multiplier $(I - \rho W)^{-1}$. Following LeSage & Pace (2009):

$$\text{Direct}_j = \frac{1}{n} \operatorname{tr}\!\left[(I - \rho W)^{-1}\right] \beta_j, \qquad \text{Total}_j = \frac{1}{n} \mathbf{1}^\top (I - \rho W)^{-1} \mathbf{1} \, \beta_j$$

$$\text{Indirect}_j = \text{Total}_j - \text{Direct}_j$$

For the Spatial Durbin Model (SDM), the impact of variable $j$ also includes $\theta_j$:

$$S_j(W) = (I - \rho W)^{-1} \left( \beta_j I_n + \theta_j W \right)$$

## Diagnostics

### `moran_test(x, W)`

Moran's I test for spatial autocorrelation (Cliff & Ord, 1981):

$$I = \frac{n}{S_0} \frac{\mathbf{z}^\top W \mathbf{z}}{\mathbf{z}^\top \mathbf{z}}, \qquad \mathbf{z} = \mathbf{x} - \bar{x}$$

Apply to raw variables to detect spatial structure or to model residuals to check whether a fitted model has captured all spatial dependence.

```python
from pysqreg import moran_test

result = moran_test(model.resid_, W)
result.summary()
```

## Visualization

Requires `matplotlib` (install via `pip install pysqreg[plot]`).

### `plot_moran(x, W)`

Moran's I scatterplot with LISA-style quadrant classification (HH, LL, LH, HL).

### `fit_quantile_process(X, y, W)` + `plot_quantile_process(result)`

Estimate the model across a grid of quantiles $\tau \in (0, 1)$ and visualize how each coefficient evolves, with layered confidence bands and an optional OLS reference line.

```python
from pysqreg import fit_quantile_process, plot_quantile_process

result = fit_quantile_process(X, y, W, method="two_stage")
fig = plot_quantile_process(result)
```

### `plot_rho_path(model)`

Visualize the Chernozhukov-Hansen $\rho$-selection path -- the coefficient $\hat{\alpha}(\rho)$ on the predicted spatial lag for each candidate $\rho$ -- for models fitted with `method='grid_search'`.

## API reference

### Cross-sectional models

#### `QuantSAR(tau, method, inference, nboot, alpha, rhomat, verbose, random_state)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate ($0 < \tau < 1$) |
| `method` | str | `'two_stage'` | `'two_stage'` or `'grid_search'` |
| `inference` | str | None | `'bootstrap'` or `'analytical'` (defaults follow method) |
| `nboot` | int | 100 | Bootstrap replications |
| `alpha` | float | 0.05 | Significance level for confidence intervals |
| `rhomat` | array | None | Grid of candidate $\rho$ values (grid_search only) |
| `verbose` | int | 0 | 0 = silent, 1 = progress |
| `random_state` | int | None | Random seed for reproducibility |

#### `QuantSLX(tau, inference, nboot, alpha, verbose, random_state)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate ($0 < \tau < 1$) |
| `inference` | str | `'analytical'` | `'bootstrap'` or `'analytical'` |
| `nboot` | int | 100 | Bootstrap replications (bootstrap only) |
| `alpha` | float | 0.05 | Significance level for confidence intervals |
| `verbose` | int | 0 | 0 = silent, 1 = progress |
| `random_state` | int | None | Random seed for reproducibility |

#### `QuantSDM(tau, method, inference, nboot, alpha, rhomat, verbose, random_state)`

Same parameters as `QuantSAR`. Nests SAR and SLX as special cases.

### Spatio-temporal models

#### `DynQuantSAR(tau, method, fixed_effects, lam, inference, nboot, alpha, rhomat, max_temporal_lag, verbose, random_state)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate ($0 < \tau < 1$) |
| `method` | str | `'two_stage'` | `'two_stage'` or `'grid_search'` |
| `fixed_effects` | str | `'penalized'` | `'penalized'`, `'dummies'`, or `'none'` |
| `lam` | float | None | Penalty $\lambda$ (None = auto-select via BIC) |
| `inference` | str | None | `'bootstrap'` or `'analytical'` |
| `nboot` | int | 100 | Bootstrap replications |
| `alpha` | float | 0.05 | Significance level |
| `rhomat` | array | None | Grid of candidate $\rho$ values (grid_search only) |
| `max_temporal_lag` | int | 3 | Deepest lag for temporal instruments |
| `verbose` | int | 0 | 0 = silent, 1 = progress |
| `random_state` | int | None | Random seed |

#### `DynQuantSLX(tau, fixed_effects, lam, inference, nboot, alpha, verbose, random_state)`

Same as `DynQuantSAR` but without `method`, `rhomat`, or `max_temporal_lag` (no endogenous spatial lag).

#### `DynQuantSDM(tau, method, fixed_effects, lam, inference, nboot, alpha, rhomat, max_temporal_lag, verbose, random_state)`

Same parameters as `DynQuantSAR`. The most general panel model.

### Common methods

- **`model.fit(X, y, W)`** -- Fit the model. $X$ should **not** include an intercept (added automatically). $W$ must be row-standardized.
- **`model.fit(X, y, W, n_units=N, n_periods=T)`** -- Panel models require panel dimensions or `unit_ids`/`time_ids`.
- **`model.predict(X, W=None, y=None)`** -- Predict target values. With $W$ and $y$, returns the full spatial prediction.
- **`model.summary()`** -- Print formatted coefficient table and spatial impacts.

### Fitted attributes

| Attribute | Models | Description |
|-----------|--------|-------------|
| `coef_` | all | Estimated $\hat{\beta}$ for $X$ covariates |
| `intercept_` | all | Estimated intercept |
| `rho_` | SAR, SDM, DynSAR, DynSDM | Spatial autoregressive parameter $\hat{\rho}$ |
| `theta_` | SLX, SDM, DynSLX, DynSDM | Spillover coefficients $\hat{\theta}$ for $WX$ terms |
| `gamma_` | DynSLX, DynSAR, DynSDM | Temporal autoregressive parameter $\hat{\gamma}$ |
| `delta_` | DynSDM | Spatio-temporal lag parameter $\hat{\delta}$ ($Wy_{t-1}$) |
| `alpha_i_` | DynSLX, DynSAR, DynSDM | Estimated unit fixed effects $\hat{\alpha}_i$ |
| `se_` | all | Standard errors |
| `pvalues_` | all | Two-sided p-values |
| `results_` | all | Full results as a DataFrame |
| `impacts_` | SAR, SDM, DynSAR, DynSDM | Spatial impacts table (direct, indirect, total) |
| `panel_` | DynSLX, DynSAR, DynSDM | Panel structure metadata |

## Package structure

```
pysqreg/
    areal/                  # Cross-sectional spatial models
        _base.py            # Shared base class, validation, IV helpers
        _diagnostics.py     # Moran's I test
        sar.py              # QuantSAR
        slx.py              # QuantSLX
        sdm.py              # QuantSDM
    spatiotemporal/         # Dynamic panel spatial models
        _panel.py           # Panel data utilities, lag construction
        _base.py            # Penalized FE, panel IV, base class
        slx.py              # DynQuantSLX
        sar.py              # DynQuantSAR
        sdm.py              # DynQuantSDM
    plots.py                # Visualization functions
```

## References

- Kim, T.-H., & Muller, C. (2004). Two-stage quantile regression when the first stage is based on quantile regression. *The Econometrics Journal*, 7(1), 218--231.
- Chernozhukov, V., & Hansen, C. (2006). Instrumental quantile regression inference for structural and treatment effect models. *Journal of Econometrics*, 132(2), 491--525.
- Koenker, R. (2004). Quantile regression for longitudinal data. *Journal of Multivariate Analysis*, 91(1), 74--89.
- Galvao, A. F. (2011). Quantile regression for dynamic panel data with fixed effects. *Journal of Econometrics*, 164(1), 142--157.
- Harding, M., & Lamarche, C. (2014). Estimating and testing a quantile regression model with interactive effects. *Journal of Econometrics*, 178(1), 101--113.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.

## License

MIT
