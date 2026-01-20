# Projections and PCA

The `fca.projections` module provides utilities for computing PCA, measuring explained variance, and analyzing information flow through matrix projections.

## Principal Component Analysis

### `perform_pca()`

Performs PCA on a data matrix using either SVD or eigendecomposition.

```python
from fca.projections import perform_pca

# Collect activations
activations = torch.randn(1000, 768)  # 1000 samples, 768 dimensions

# Perform PCA
result = perform_pca(
    X=activations,
    n_components=100,      # Number of components (None = all)
    scale=True,            # Scale features to unit variance
    center=True,           # Center features to zero mean
    transform_data=False,  # Whether to return transformed data
    use_eigen=True,        # Use eigendecomposition (faster)
    batch_size=500,        # Batch size for covariance computation
    verbose=True,          # Show progress
)

# Access results
components = result["components"]           # Shape: (n_components, D)
explained_var = result["explained_variance"]  # Shape: (n_components,)
prop_var = result["proportion_expl_var"]    # Shape: (n_components,)
means = result["means"]                      # Shape: (D,)
stds = result["stds"]                        # Shape: (D,)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | Tensor | required | Data matrix (N, D) |
| `n_components` | int | D | Number of components to return |
| `scale` | bool | True | Scale features to unit variance |
| `center` | bool | True | Center features to zero mean |
| `transform_data` | bool | False | Return transformed data |
| `use_eigen` | bool | True | Use eigendecomposition (faster) |
| `randomized` | bool | False | Use randomized SVD (when use_eigen=False) |
| `batch_size` | int | None | Batch size for covariance matrix |
| `verbose` | bool | True | Show progress |

**Returns:** Dictionary with:
- `components`: Principal components as row vectors
- `explained_variance`: Variance explained by each component
- `proportion_expl_var`: Proportion of total variance for each component
- `means`: Feature means used for centering
- `stds`: Feature stds used for scaling
- `transformed_X`: Transformed data (if `transform_data=True`)

### `perform_eigen_pca()`

PCA using eigendecomposition of the covariance matrix. Called internally by `perform_pca()` when `use_eigen=True`.

```python
from fca.projections import perform_eigen_pca

result = perform_eigen_pca(
    X=activations,
    n_components=100,
    scale=True,
    center=True,
    batch_size=500,
)
```

## Variance Analysis

### `explained_variance()`

Computes the explained variance between predictions and labels.

```python
from fca.projections import explained_variance

# After projection and reconstruction
reconstructed = fca.projinv(activations)

# Compute explained variance per dimension
expl_var = explained_variance(
    preds=reconstructed,
    labels=activations,
    eps=1e-8,              # Prevent division by zero
    mean_over_dims=False,  # Return per-dimension or mean
)
print(expl_var.shape)  # (D,) or (1,)

# Get single value
mean_expl_var = explained_variance(
    preds=reconstructed,
    labels=activations,
    mean_over_dims=True,
)
print(f"Mean explained variance: {mean_expl_var:.2%}")
```

**Formula:**
```
explained_variance = 1 - var(labels - preds) / var(labels)
```

### `projinv_expl_variance()`

Projects data through a weight matrix and back, then computes explained variance.

```python
from fca.projections import projinv_expl_variance

# Weight matrix (e.g., from FCA or PCA)
W = fca.weight.T  # Shape: (D, R)

# Compute explained variance
expl_var = projinv_expl_variance(activations, W)
print(f"Explained variance: {expl_var.mean():.2%}")
```

### `lost_variance()`

Computes the variance lost when projecting through a weight matrix.

```python
from fca.projections import lost_variance

lost = lost_variance(activations, W)
print(f"Lost variance: {lost.mean():.2%}")
# lost = 1 - explained_variance
```

### `component_wise_expl_var()`

Computes explained variance for each component individually and cumulatively.

```python
from fca.projections import component_wise_expl_var

# Analyze FCA or PCA components
individual_vars, cumulative_vars = component_wise_expl_var(
    actvs=activations,
    weight=fca.weight,
    eps=1e-6,
)

# individual_vars: explained variance of each component alone
# cumulative_vars: explained variance of components 0..i combined

print("Component 0 explains:", individual_vars[0].mean())
print("Components 0-9 explain:", cumulative_vars[9].mean())
```

## Matrix Projections

### `matrix_projinv()`

Projects activations into a weight space and inverts the projection.

```python
from fca.projections import matrix_projinv

# x: (B, D) activations
# W: (D, P) weight matrix

# Project to W's column space and back
reconstructed = matrix_projinv(x, W)
# Equivalent to: x @ W @ pinv(W)
```

This is useful for analyzing how much information is preserved when passing through a linear layer.

### `get_cor_mtx()`

Computes correlation matrix between two sets of features using GPU.

```python
from fca.projections import get_cor_mtx

# X: (T, C) - T timepoints, C features
# Y: (T, K) - T timepoints, K features

cor_mtx = get_cor_mtx(
    X=activations_layer1,
    Y=activations_layer2,
    batch_size=500,    # Batch for memory efficiency
    to_numpy=False,    # Return numpy array?
    zscore=True,       # Z-score normalize features?
    device=None,       # GPU device
)
# cor_mtx: (C, K) correlation matrix
```

## Usage with FCA

### Analyzing FCA Components

```python
from fca import FunctionalComponentAnalysis
from fca.projections import explained_variance, component_wise_expl_var

# Create and train FCA
fca = FunctionalComponentAnalysis(size=768, init_rank=50)
# ... training ...

# How well does FCA reconstruct activations?
reconstructed = fca.projinv(test_activations)
expl_var = explained_variance(
    reconstructed, test_activations, mean_over_dims=True
)
print(f"FCA explains {expl_var:.1%} of variance")

# Analyze individual components
indiv, cumul = component_wise_expl_var(test_activations, fca.weight)
for i in range(5):
    print(f"Component {i}: {indiv[i].mean():.2%} individually, {cumul[i].mean():.2%} cumulative")
```

### Comparing FCA to PCA

```python
from fca import FunctionalComponentAnalysis, PCAFunctionalComponentAnalysis
from fca.projections import explained_variance

# Data for comparison
activations = collect_activations(model, data)["layer5"]

# Standard FCA
fca = FunctionalComponentAnalysis(size=activations.shape[-1], init_rank=10)
# ... train FCA ...

# PCA-initialized FCA
pca_fca = PCAFunctionalComponentAnalysis(
    X=activations,
    max_rank=10,
)

# Compare explained variance
fca_var = explained_variance(
    fca.projinv(activations), activations, mean_over_dims=True
)
pca_var = explained_variance(
    pca_fca.projinv(activations), activations, mean_over_dims=True
)

print(f"FCA explains: {fca_var:.1%}")
print(f"PCA explains: {pca_var:.1%}")
```

### Finding Optimal Rank

```python
from fca.projections import perform_pca

# Perform PCA to find explained variance curve
result = perform_pca(activations, n_components=None)

# Plot cumulative explained variance
import matplotlib.pyplot as plt

cumsum = result["proportion_expl_var"].cumsum()
plt.plot(cumsum.numpy())
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.axhline(0.95, color='r', linestyle='--', label='95%')
plt.legend()
plt.show()

# Find rank needed for 95% variance
rank_95 = (cumsum < 0.95).sum().item() + 1
print(f"Rank needed for 95% variance: {rank_95}")
```

## Performance Tips

1. **Use `use_eigen=True`** for PCA when D < N (more features than samples)
2. **Use `batch_size`** for large datasets to avoid OOM
3. **Use `randomized=True`** when you only need a few components from very high-dimensional data
4. **Move to GPU** before computing correlations for large matrices:
   ```python
   cor_mtx = get_cor_mtx(X.cuda(), Y.cuda(), device=0)
   ```
