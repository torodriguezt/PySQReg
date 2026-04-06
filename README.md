# PySQReg

Spatial Quantile Regression for areal (lattice) data in Python.

PySQReg provides a family of quantile regression estimators for spatial models on areal / lattice data with row-standardized weight matrices:

| Model | Specification | Class |
|-------|--------------|-------|
| **SAR** | `y = rho * W @ y + X @ beta + u` | `QuantSAR` |
| **SLX** | `y = X @ beta + W @ X @ theta + u` | `QuantSLX` |
| **SDM** | `y = rho * W @ y + X @ beta + W @ X @ theta + u` | `QuantSDM` |

## Why spatial quantile regression?

Standard spatial regression (e.g., spatial lag models estimated by ML or 2SLS) focuses on the **conditional mean**. PySQReg lets you estimate effects across the **entire conditional distribution** — revealing how spatial spillovers differ at the tails versus the center, which is critical for understanding inequality, risk, and heterogeneous policy impacts.

## Features

- **Three spatial model specifications:** SAR, SLX, and Spatial Durbin (SDM)
- **Two IV estimation methods:**
  - Kim & Muller (2004) two-stage IV quantile regression with bootstrap inference
  - Chernozhukov & Hansen (2006) IV quantile regression via grid search
- **Spatial impacts** (LeSage & Pace): direct, indirect (spillover), and total effects
- **Moran's I test** for spatial autocorrelation in variables or residuals
- **Publication-quality visualizations:** Moran scatterplot, quantile-process ribbon plots, rho-selection path
- **scikit-learn compatible** API: `fit()`, `predict()`, `get_params()`, `set_params()`
- Supports **sparse weight matrices** and **large datasets** (power-series approximation for n > 2000)

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

For a complete walkthrough see the [tutorial notebook](examples/tutorial.ipynb).

## Models

### `QuantSAR` — Spatial Autoregressive Model

Estimates `y = rho * W @ y + X @ beta + u` at a given quantile using IV approaches. The endogenous spatial lag `W @ y` is instrumented using spatial lags of the exogenous variables.

### `QuantSLX` — Spatial Lag of X

Estimates `y = X @ beta + W @ X @ theta + u`. A purely exogenous model — no endogenous spatial lag — so standard quantile regression applies directly. Spatial spillovers are captured by the `W @ X` terms.

### `QuantSDM` — Spatial Durbin Model

Estimates `y = rho * W @ y + X @ beta + W @ X @ theta + u`. Nests both `QuantSAR` (theta=0) and `QuantSLX` (rho=0) as special cases, combining the endogenous spatial lag with exogenous spillover terms.

## Estimation methods

### Kim & Muller (2004) — `method='two_stage'`

A two-stage instrumental-variable quantile regression. The spatial lag `W @ y` is instrumented using `W @ X` in a first-stage quantile regression, then the fitted values enter the second-stage quantile regression. Inference is based on the bootstrap.

### Chernozhukov & Hansen (2006) — `method='grid_search'`

An IV quantile regression that searches over a grid of candidate `rho` values. For each `rho`, the model transforms the dependent variable as `y - rho * W @ y` and estimates a quantile regression including the fitted spatial lag as an additional regressor. The optimal `rho` is the value that drives the coefficient on the fitted lag closest to zero. Inference uses a sandwich variance estimator.

## Diagnostics

### `moran_test(x, W)`

Moran's I test for spatial autocorrelation (Cliff & Ord, 1981). Apply to raw variables to detect spatial structure or to model residuals to check whether a fitted model has captured all spatial dependence.

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

Estimate the model across a grid of quantiles and visualize how each coefficient evolves over quantile levels, with layered confidence bands and an optional OLS reference line.

```python
from pysqreg import fit_quantile_process, plot_quantile_process

result = fit_quantile_process(X, y, W, method="two_stage")
fig = plot_quantile_process(result)
```

### `plot_rho_path(model)`

Visualize the Chernozhukov-Hansen rho-selection path — the coefficient on the predicted spatial lag for each candidate rho — for models fitted with `method='grid_search'`.

## API reference

### `QuantSAR(tau, method, nboot, alpha, rhomat, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate (0 < tau < 1) |
| `method` | str | `'two_stage'` | `'two_stage'` or `'grid_search'` |
| `nboot` | int | 100 | Bootstrap replications (two_stage only) |
| `alpha` | float | 0.05 | Significance level for confidence intervals |
| `rhomat` | array | None | Grid of candidate rho values (grid_search only) |
| `verbose` | int | 0 | 0 = silent, 1 = progress |

### `QuantSLX(tau, inference, nboot, alpha, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate (0 < tau < 1) |
| `inference` | str | `'analytical'` | `'bootstrap'` or `'analytical'` |
| `nboot` | int | 100 | Bootstrap replications (bootstrap only) |
| `alpha` | float | 0.05 | Significance level for confidence intervals |
| `verbose` | int | 0 | 0 = silent, 1 = progress |

### `QuantSDM(tau, method, inference, nboot, alpha, rhomat, verbose)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tau` | float | 0.5 | Quantile to estimate (0 < tau < 1) |
| `method` | str | `'two_stage'` | `'two_stage'` or `'grid_search'` |
| `inference` | str | None | `'bootstrap'` or `'analytical'` (defaults follow method) |
| `nboot` | int | 100 | Bootstrap replications |
| `alpha` | float | 0.05 | Significance level for confidence intervals |
| `rhomat` | array | None | Grid of candidate rho values (grid_search only) |
| `verbose` | int | 0 | 0 = silent, 1 = progress |

### Common methods

- **`model.fit(X, y, W)`** — Fit the model. `X` should **not** include an intercept (added automatically). `W` must be row-standardized.
- **`model.predict(X, W=None, y=None)`** — Predict target values. With `W` and `y`, returns the full spatial prediction.
- **`model.summary()`** — Print formatted coefficient table and spatial impacts.

### Fitted attributes

| Attribute | Models | Description |
|-----------|--------|-------------|
| `coef_` | all | Estimated coefficients for X covariates |
| `intercept_` | all | Estimated intercept |
| `rho_` | SAR, SDM | Spatial autoregressive parameter |
| `theta_` | SLX, SDM | Coefficients for WX spillover terms |
| `se_` | all | Standard errors |
| `pvalues_` | all | Two-sided p-values |
| `results_` | all | Full results as a DataFrame |
| `impacts_` | SAR, SDM | Spatial impacts table (direct, indirect, total) |

## References

- Kim, T.-H., & Muller, C. (2004). Two-stage quantile regression when the first stage is based on quantile regression. *The Econometrics Journal*, 7(1), 218-231.
- Chernozhukov, V., & Hansen, C. (2006). Instrumental quantile regression inference for structural and treatment effect models. *Journal of Econometrics*, 132(2), 491-525.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.
- Cliff, A. D., & Ord, J. K. (1981). *Spatial Processes: Models and Applications*. Pion.

## License

MIT
