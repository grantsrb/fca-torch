# Core Classes

This document describes the main FCA classes for learning and applying orthogonal functional components.

## FunctionalComponentAnalysis

The primary class for learning orthogonal vectors that represent functional components of a neural network layer.

### Constructor

```python
FunctionalComponentAnalysis(
    size,
    max_rank=None,
    use_complement_in_hook=False,
    component_mask=None,
    means=None,
    stds=None,
    orthogonalization_vectors=None,
    init_rank=None,
    init_vectors=None,
    init_noise=None,
    orth_with_doubles=False,
    orth_method="classical",
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | int | required | Dimensionality of the representation space |
| `max_rank` | int | `size` | Maximum number of components allowed |
| `init_rank` | int | 1 | Initial number of components to create |
| `use_complement_in_hook` | bool | False | If True, hook removes FCA components instead of keeping them |
| `component_mask` | Tensor | None | Index tensor to select specific components |
| `means` | Tensor | None | Mean vector for centering representations |
| `stds` | Tensor | None | Std vector for scaling representations |
| `orthogonalization_vectors` | list | None | Additional vectors to orthogonalize against |
| `init_vectors` | list | None | Vectors to initialize components from |
| `init_noise` | float | 0.1 | Noise added to initialization vectors |
| `orth_with_doubles` | bool | False | Use double precision for orthogonalization |
| `orth_method` | str | "classical" | Orthogonalization method (see below) |

### Orthogonalization Methods

The `orth_method` parameter controls which algorithm is used:

| Method | Description | Best For |
|--------|-------------|----------|
| `"classical"` | Fast matrix-based Gram-Schmidt with covariance caching | Low to medium ranks (<100) |
| `"modified"` | Modified Gram-Schmidt with incremental updates | High ranks, better stability |
| `"householder"` | QR decomposition via PyTorch LAPACK | Batch operations, maximum stability |
| `"hybrid"` | Classical for frozen, MGS for trainable | Training scenarios |

### Properties

#### `weight`
Returns the orthonormalized weight matrix.

```python
matrix = fca.weight  # Shape: (rank, size)
```

#### `rank`
Returns the number of components.

```python
n_components = fca.rank
```

#### `n_components`
Alias for `rank`.

### Methods

#### `forward(x, inverse=False, components=None)`

Project activations through FCA.

```python
# Forward: project to FCA space
projected = fca(activations)  # (B, D) -> (B, R)

# Inverse: project back to original space
reconstructed = fca(projected, inverse=True)  # (B, R) -> (B, D)

# Use specific components
projected = fca(activations, components=torch.tensor([0, 1, 2]))
```

**Parameters:**
- `x`: Input tensor of shape `(..., D)` for forward, `(..., R)` for inverse
- `inverse`: If True, projects from FCA space back to representation space
- `components`: Optional index tensor to select specific components

#### `projinv(x, components=None)`

Projects to FCA space and back in one step.

```python
# Equivalent to: fca(fca(x), inverse=True)
reconstructed = fca.projinv(activations)
```

#### `add_component(init_vector=None)`

Adds a new component to the FCA.

```python
fca.add_component()  # Random initialization
fca.add_component(init_vector=my_vector)  # Custom initialization
```

#### `remove_component(idx=None)`

Removes a component by index (default: last).

```python
fca.remove_component()  # Remove last
fca.remove_component(idx=2)  # Remove third component
```

#### `freeze_parameters(freeze=True)`

Freezes or unfreezes all parameters.

```python
fca.freeze_parameters(True)   # Freeze all
fca.freeze_parameters(False)  # Unfreeze all
```

When freezing, parameters are orthonormalized and their data is updated in-place.

#### `add_orthogonalization_vectors(new_vectors)`

Adds vectors that components must be orthogonal to (but aren't part of the output).

```python
# Make FCA orthogonal to these vectors
fca.add_orthogonalization_vectors([vec1, vec2, vec3])
```

#### `reorthogonalize()`

Re-orthogonalizes all parameters using Householder QR. Call periodically during training at high ranks to correct numerical drift.

```python
# During training loop
if step % 1000 == 0:
    fca.reorthogonalize()
```

#### `set_cached(cached=True)`

Enable/disable weight caching for faster inference.

```python
fca.set_cached(True)   # Cache the weight matrix
# ... multiple forward passes ...
fca.set_cached(False)  # Disable caching
```

#### `get_forward_hook(comms_dict=None, output_extractor=None, shape_transform=None, inverse_transform=None)`

Returns a forward hook function for attaching to PyTorch modules.

```python
hook = fca.get_forward_hook()
handle = layer.register_forward_hook(hook)
```

#### `hook_model_layer(model, layer, comms_dict=None, rep_type="auto", ...)`

Helper to register hook on a named layer.

```python
handle = fca.hook_model_layer(
    model=model,
    layer="encoder.layer.5",
    rep_type="sequence",
)
```

**rep_type options:**
- `"auto"`: Automatic detection
- `"image"` / `"images"`: For (B, C, H, W) tensors
- `"sequence"` / `"language"`: For (B, S, D) tensors
- `"flat"` / `"mlp"`: For (B, D) tensors

#### `interchange_intervention(trg, src)`

Performs a DAS-style interchange intervention.

```python
# Replace FCA components of trg with those from src
result = fca.interchange_intervention(trg_activations, src_activations)
```

### Example: Full Training Loop

```python
import torch
from fca import FunctionalComponentAnalysis
from fca.wrappers import CategoricalModelWrapper, wrap_data

# Setup
model = CategoricalModelWrapper(my_model)
fca = FunctionalComponentAnalysis(
    size=hidden_size,
    init_rank=1,
    max_rank=50,
    orth_method="hybrid",
)

# Attach FCA
handle = fca.hook_model_layer(model.model, layer_name)

# Training loop
optimizer = torch.optim.Adam(fca.parameters(), lr=0.001)

for epoch in range(100):
    for batch in data_loader:
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()

    # Add component when plateauing
    if should_add_component:
        fca.freeze_parameters()
        fca.add_component()
        optimizer = torch.optim.Adam(fca.parameters(), lr=0.001)

handle.remove()
```

---

## PCAFunctionalComponentAnalysis

FCA initialized with PCA components. Useful when you want to start from data-driven directions.

### Constructor

```python
PCAFunctionalComponentAnalysis(
    X,
    scale=True,
    center=True,
    **kwargs  # All FunctionalComponentAnalysis parameters
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `X` | Tensor | required | Data matrix (N, D) to compute PCA on |
| `scale` | bool | True | Scale data before PCA |
| `center` | bool | True | Center data before PCA |

### Additional Methods

#### `proportion_expl_var(rank=None, actvs=None)`

Returns proportion of explained variance.

```python
# Using stored PCA info
var = pca_fca.proportion_expl_var(rank=10)

# Using new activations
var = pca_fca.proportion_expl_var(actvs=test_activations)
```

#### `set_max_rank(max_rank)`

Updates max rank and reinitializes components.

```python
pca_fca.set_max_rank(100)
```

### Example

```python
from fca import PCAFunctionalComponentAnalysis

# Collect activations first
activations = collect_activations_from_model(model, data)

# Create PCA-initialized FCA
pca_fca = PCAFunctionalComponentAnalysis(
    X=activations,
    scale=True,
    center=True,
    max_rank=50,
)

# Check explained variance
print(f"Top 10 components explain: {pca_fca.proportion_expl_var(10):.2%}")
```

---

## UnnormedFCA

A variant of FCA that doesn't normalize vectors. Useful for debugging or when you need unnormalized components.

### Constructor

Same as `FunctionalComponentAnalysis`.

### Differences from FCA

- `orthogonalize_vector()` has `norm=False` by default
- Components are orthogonal but not unit-length

### Example

```python
from fca import UnnormedFCA

# Components will be orthogonal but not normalized
unnormed_fca = UnnormedFCA(size=64, init_rank=10)
```

---

## OrthogonalProjection

A simpler alternative to FCA when you need trainable parameters that are differentiably constrained to be orthogonal to a set of fixed reference vectors.

**Key differences from FCA:**
- The output vectors do NOT need to be orthogonal to each other
- The fixed vectors do NOT need to be orthogonal to each other
- Simpler API focused on the orthogonalization constraint

### Constructor

```python
OrthogonalProjection(
    size,
    n_params,
    fixed_vectors=None,
    normalize=True,
    init_noise=0.1,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `size` | int | required | Dimensionality of vectors (D) |
| `n_params` | int | required | Number of trainable parameter vectors (N) |
| `fixed_vectors` | list/Tensor | None | Vectors that outputs must be orthogonal to |
| `normalize` | bool | True | Normalize output vectors to unit length |
| `init_noise` | float | 0.1 | Std dev of random initialization |

### Methods

#### `forward()` / `weight`

Returns the trainable parameters projected to be orthogonal to all fixed vectors. Fully differentiable.

```python
ortho_params = proj()  # Shape: (N, D)
# or
ortho_params = proj.weight
```

#### `set_fixed_vectors(vectors)`

Set the fixed vectors that outputs must be orthogonal to.

```python
proj.set_fixed_vectors([vec1, vec2, vec3])
# or
proj.set_fixed_vectors(tensor_of_shape_K_D)
```

#### `add_fixed_vectors(vectors)`

Add additional fixed vectors to the existing set.

```python
proj.add_fixed_vectors([new_vec1, new_vec2])
```

#### `clear_fixed_vectors()`

Remove all fixed vectors.

#### `check_orthogonality(tol=1e-5)`

Verify that outputs are orthogonal to fixed vectors.

```python
result = proj.check_orthogonality()
print(result["is_orthogonal"])     # True/False
print(result["max_dot_product"])   # Should be ~0
```

### Properties

- `fixed_vectors`: Returns the fixed vectors (K, D) or None
- `num_fixed`: Number of fixed vectors
- `basis_rank`: Rank of the fixed vector subspace

### Example

```python
import torch
from fca import OrthogonalProjection

# Create module with 5 trainable vectors of dimension 100
proj = OrthogonalProjection(size=100, n_params=5)

# Add fixed vectors that outputs must be orthogonal to
# (these don't need to be orthogonal to each other)
fixed_vecs = [torch.randn(100) for _ in range(10)]
proj.set_fixed_vectors(fixed_vecs)

# Get orthogonalized parameters (differentiable)
ortho_params = proj()  # Shape: (5, 100)

# Use in training loop
optimizer = torch.optim.Adam(proj.parameters(), lr=0.01)

for step in range(100):
    optimizer.zero_grad()

    ortho_params = proj()  # Orthogonalization happens here
    loss = my_loss_function(ortho_params)

    loss.backward()  # Gradients flow through orthogonalization
    optimizer.step()

# Verify orthogonality
result = proj.check_orthogonality()
print(f"Max dot product with fixed vectors: {result['max_dot_product']:.2e}")
```

### Use Cases

- Learning directions orthogonal to known features
- Subspace learning with constraints
- When you need differentiable orthogonalization but don't need mutual orthogonality between learned vectors

---

## Loading Saved FCAs

### `load_fca_from_path(save_dir)`

Loads a single FCA from a checkpoint directory.

```python
from fca import load_fca_from_path

fca = load_fca_from_path("/path/to/checkpoint/")
```

### `load_fcas_from_path(file_path)`

Loads multiple FCAs from a single checkpoint file.

```python
from fca import load_fcas_from_path

fcas = load_fcas_from_path("/path/to/checkpoint.pt")
# fcas is a dict: {layer_name: FCA_instance}
```

### `load_fcas(model, load_path, use_complement_in_hook=True)`

Loads FCAs and attaches them to a model.

```python
from fca import load_fcas

fcas, handles = load_fcas(model, "/path/to/checkpoint.pt")

# Clean up
for h in handles.values():
    h.remove()
```

---

## Convenience Functions

### `attach_fca(model, layer_name, fca_instance, output_format="auto", ...)`

High-level function to attach FCA to any layer. See [Getting Started](getting-started.md) for examples.

### `get_layer_output_size(model, layer_name, sample_input=None)`

Determines the output size of a layer for FCA initialization.

```python
from fca import get_layer_output_size

size = get_layer_output_size(model, "layer1.0.conv2")
fca = FunctionalComponentAnalysis(size=size, init_rank=10)
```
