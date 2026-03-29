# PySQReg

Spatial Quantile Regression for areal (lattice) data in Python.

PySQReg estimates the Spatial Autoregressive (SAR) quantile regression model:

```
y = rho * W @ y + X @ beta + u
```

where `W` is a row-standardized spatial weight matrix, `rho` captures spatial dependence, and estimation is performed at a given quantile `tau` using instrumental-variable approaches.

## Why spatial quantile regression?

Standard spatial regression (e.g., spatial lag models estimated by ML or 2SLS) focuses on the **conditional mean**. PySQReg lets you estimate effects across the **entire conditional distribution** — revealing how spatial spillovers differ at the tails versus the center, which is critical for understanding inequality, risk, and heterogeneous policy impacts.

## Features

- **Two estimation methods:**
  - Kim & Muller (2004) two-stage IV quantile regression with bootstrap inference
  - Chernozhukov & Hansen (2006) IV quantile regression via grid search
- **Spatial impacts** (LeSage & Pace): direct, indirect (spillover), and total effects
- **scikit-learn compatible** API: `fit()`, `predict()`, `get_params()`, `set_params()`
- Supports **sparse weight matrices** and **large datasets** (power-series approximation for n > 2000)

## Installation

```bash
pip install pysqreg
```

Or install from source:

```bash
git clone https://github.com/torodriguezt/PySQReg.git
cd PySQReg
pip install -e .
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

## Methods

### Kim & Muller (2004) — `method='two_stage'`

A two-stage instrumental-variable quantile regression. The spatial lag `W @ y` is instrumented using `W @ X` in a first-stage quantile regression, then the fitted values enter the second-stage quantile regression. Inference is based on the bootstrap.

### Chernozhukov & Hansen (2006) — `method='grid_search'`

An IV quantile regression that searches over a grid of candidate `rho` values. For each `rho`, the model transforms the dependent variable as `y - rho * W @ y` and estimates a quantile regression including the fitted spatial lag as an additional regressor. The optimal `rho` is the value that drives the coefficient on the fitted lag closest to zero. Inference uses a sandwich variance estimator.

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

### `model.fit(X, y, W, wy=None, inst=None, winst=None)`

Fit the SAR quantile regression. `X` should **not** include an intercept (added automatically). `W` must be a row-standardized spatial weight matrix.

### `model.predict(X, W=None, y=None)`

Predict target values. With `W` and `y`, returns the full SAR prediction including the spatial lag component.

### `model.summary()`

Print formatted coefficient table and spatial impacts.

### Fitted attributes

| Attribute | Description |
|-----------|-------------|
| `coef_` | Estimated coefficients for covariates |
| `intercept_` | Estimated intercept |
| `rho_` | Estimated spatial autoregressive parameter |
| `se_` | Standard errors |
| `pvalues_` | Two-sided p-values |
| `results_` | Full results as a DataFrame |
| `impacts_` | Spatial impacts table (two_stage only) |

## References

- Kim, T.-H., & Muller, C. (2004). Two-stage quantile regression when the first stage is based on quantile regression. *The Econometrics Journal*, 7(1), 218-231.
- Chernozhukov, V., & Hansen, C. (2006). Instrumental quantile regression inference for structural and treatment effect models. *Journal of Econometrics*, 132(2), 491-525.
- LeSage, J., & Pace, R. K. (2009). *Introduction to Spatial Econometrics*. CRC Press.

## License

MIT
