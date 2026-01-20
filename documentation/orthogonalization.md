# Orthogonalization Methods

FCA relies on orthogonalization to ensure all component vectors are mutually orthogonal and unit-length. This document explains the available methods and their trade-offs.

## The Problem: Numerical Stability

Classical Gram-Schmidt orthogonalization suffers from **catastrophic cancellation** - a numerical issue where floating-point precision is lost during subtraction operations.

When orthogonalizing vector `v` against an orthonormal basis `{u1, ..., un}`:

```
v' = v - sum(dot(v, ui) * ui)
```

The subtraction can lose significant digits when `v` is nearly parallel to the span of the basis. This error compounds as the number of vectors (rank) increases.

**Error bounds:**
- Classical Gram-Schmidt: O(epsilon * kappa^2)
- Modified Gram-Schmidt: O(epsilon * kappa)

Where epsilon is machine precision and kappa is the condition number.

## Available Methods

### Classical Gram-Schmidt (`"classical"`)

The default method. Computes all projections first, then subtracts them in one step.

```python
# Pseudocode
proj_sum = 0
for u in basis:
    proj_sum += dot(v, u) * u
v = v - proj_sum
```

**Advantages:**
- Fast: can use matrix multiplication
- Supports covariance matrix caching for O(r^2) speedup
- GPU-friendly (parallelizable)

**Disadvantages:**
- Less numerically stable at high ranks
- Error grows with rank squared

**Best for:** Low to medium ranks (<100), when speed is critical.

```python
fca = FunctionalComponentAnalysis(
    size=512,
    init_rank=50,
    orth_method="classical",
)
```

### Modified Gram-Schmidt (`"modified"`)

Updates `v` incrementally after each projection instead of computing all projections first.

```python
# Pseudocode
for u in basis:
    v = v - dot(v, u) * u  # Update v immediately
```

**Advantages:**
- More numerically stable (one order of magnitude better)
- Same computational complexity O(r * S)
- Simple implementation

**Disadvantages:**
- Sequential computation (harder to parallelize)
- Loses the covariance matrix optimization

**Best for:** High ranks (>100), when stability matters more than speed.

```python
fca = FunctionalComponentAnalysis(
    size=512,
    init_rank=200,
    orth_method="modified",
)
```

### Householder QR (`"householder"`)

Uses QR decomposition with Householder reflections. This is the gold standard for numerical stability.

```python
# Uses PyTorch's optimized LAPACK backend
Q, R = torch.linalg.qr(matrix)
```

**Advantages:**
- Most numerically stable
- Highly optimized in PyTorch/LAPACK
- Single function call

**Disadvantages:**
- Requires all vectors at once (not incremental)
- Returns full Q matrix
- More memory usage

**Best for:** Batch initialization, re-orthogonalization, maximum stability needs.

```python
fca = FunctionalComponentAnalysis(
    size=512,
    init_rank=100,
    orth_method="householder",
)
```

### Hybrid (`"hybrid"`)

Uses classical method for frozen components (fast) and modified Gram-Schmidt for trainable components (stable).

**Advantages:**
- Best of both worlds for training scenarios
- Fast for frozen components via matrix caching
- Stable for actively trained components

**Disadvantages:**
- More complex logic
- Only beneficial during training with frozen components

**Best for:** Training scenarios with frozen and trainable components.

```python
fca = FunctionalComponentAnalysis(
    size=512,
    init_rank=10,
    orth_method="hybrid",
)

# Train first few components
# ...

# Freeze and add more
fca.freeze_parameters()
fca.add_component()  # New components use MGS for stability
```

## Comparison Table

| Method | Stability | Speed | Incremental | GPU-friendly | Use Case |
|--------|-----------|-------|-------------|--------------|----------|
| `classical` | Poor | Fast | Yes | Yes | Low ranks, inference |
| `modified` | Good | Fast | Yes | Less | High ranks, training |
| `householder` | Excellent | Fast | No | Yes | Batch init, cleanup |
| `hybrid` | Good | Balanced | Yes | Partial | Training with frozen |

## Double Precision

For additional stability, enable double precision orthogonalization:

```python
fca = FunctionalComponentAnalysis(
    size=1024,
    init_rank=500,
    orth_with_doubles=True,  # Compute in float64, return float32
)
```

This performs orthogonalization in float64 but returns results in the original dtype. It roughly doubles memory usage during orthogonalization but significantly improves stability.

## Re-orthogonalization

During long training runs at high ranks, numerical drift can accumulate. Use `reorthogonalize()` periodically to correct this:

```python
# In training loop
for step, batch in enumerate(data_loader):
    # ... training code ...

    # Periodic cleanup
    if step % 1000 == 0 and fca.rank > 50:
        fca.reorthogonalize()  # Uses Householder QR
```

## Standalone Functions

FCA provides standalone orthogonalization functions:

### `orthogonalize_vector()`

Orthogonalize a single vector against previous vectors.

```python
from fca import orthogonalize_vector

v = torch.randn(100)
basis = [torch.randn(100) for _ in range(10)]

# Classical method
v_ortho = orthogonalize_vector(v, basis, method="classical")

# Modified method
v_ortho = orthogonalize_vector(v, basis, method="modified")

# With double precision
v_ortho = orthogonalize_vector(v, basis, double_precision=True)

# Without normalization
v_ortho = orthogonalize_vector(v, basis, norm=False)
```

### `orthogonalize_vector_mgs()`

Direct access to Modified Gram-Schmidt.

```python
from fca import orthogonalize_vector_mgs

v_ortho = orthogonalize_vector_mgs(
    new_vector,
    prev_vectors,
    norm=True,
    double_precision=False,
)
```

### `orthogonalize_batch_qr()`

Orthogonalize multiple vectors at once using QR decomposition.

```python
from fca import orthogonalize_batch_qr

vectors = [torch.randn(100) for _ in range(20)]
ortho_vectors = orthogonalize_batch_qr(vectors, double_precision=True)
```

### `gram_schmidt()`

Classical Gram-Schmidt for a list of vectors.

```python
from fca import gram_schmidt

vectors = [torch.randn(100) for _ in range(10)]
ortho_vectors = gram_schmidt(vectors, double_precision=True)
```

## Measuring Orthogonality

To check how orthogonal your vectors are:

```python
def max_orthogonality_error(weight):
    """Returns maximum |dot(vi, vj)| for i != j."""
    max_error = 0.0
    for i in range(weight.shape[0]):
        for j in range(i):
            dot = abs(torch.dot(weight[i], weight[j]).item())
            max_error = max(max_error, dot)
    return max_error

# Check your FCA
error = max_orthogonality_error(fca.weight)
print(f"Max orthogonality error: {error:.2e}")
```

Typical values:
- float32, rank 50: ~1e-6 to 1e-5
- float32, rank 500: ~1e-4 to 1e-3
- float64, rank 500: ~1e-12 to 1e-10

## Recommendations

1. **Start with `"classical"`** - it's fast and sufficient for most cases

2. **Switch to `"modified"` or `"hybrid"`** when:
   - Rank exceeds 100
   - You see orthogonality degradation
   - Training becomes unstable

3. **Use `"householder"`** when:
   - Initializing from scratch with many components
   - You need guaranteed numerical stability
   - Periodically during training via `reorthogonalize()`

4. **Enable `orth_with_doubles`** when:
   - Rank exceeds 200-300
   - Working with poorly-conditioned data
   - Maximum precision is required

5. **Call `reorthogonalize()` periodically** during long training runs with high-rank FCAs
