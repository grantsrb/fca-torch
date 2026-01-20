# API Reference

Complete reference for all public functions and classes in the FCA package.

## fca (main module)

### Classes

#### FunctionalComponentAnalysis
```python
class FunctionalComponentAnalysis(nn.Module)
```
Main class for learning orthogonal functional components.

**Constructor:**
```python
FunctionalComponentAnalysis(
    size: int,
    max_rank: int = None,
    use_complement_in_hook: bool = False,
    component_mask: Tensor = None,
    means: Tensor = None,
    stds: Tensor = None,
    orthogonalization_vectors: list = None,
    init_rank: int = None,
    init_vectors: list = None,
    init_noise: float = None,
    orth_with_doubles: bool = False,
    orth_method: str = "classical",
)
```

**Methods:**
- `forward(x, inverse=False, components=None) -> Tensor`
- `projinv(x, components=None) -> Tensor`
- `add_component(init_vector=None) -> Parameter`
- `remove_component(idx=None)`
- `freeze_parameters(freeze=True)`
- `add_orthogonalization_vectors(new_vectors)`
- `reorthogonalize()`
- `set_cached(cached=True)`
- `reset_cached_weight()`
- `get_forward_hook(comms_dict=None, output_extractor=None, shape_transform=None, inverse_transform=None) -> callable`
- `hook_model_layer(model, layer, comms_dict=None, rep_type="auto", ...) -> handle`
- `interchange_intervention(trg, src) -> Tensor`

**Properties:**
- `weight: Tensor` - Orthonormalized weight matrix (R, D)
- `rank: int` - Number of components
- `n_components: int` - Alias for rank

---

#### PCAFunctionalComponentAnalysis
```python
class PCAFunctionalComponentAnalysis(FunctionalComponentAnalysis)
```
FCA initialized with PCA components.

**Constructor:**
```python
PCAFunctionalComponentAnalysis(
    X: Tensor,              # Data matrix (N, D)
    scale: bool = True,
    center: bool = True,
    **kwargs,               # FunctionalComponentAnalysis params
)
```

**Additional Methods:**
- `proportion_expl_var(rank=None, actvs=None) -> float`
- `set_max_rank(max_rank)`

---

#### UnnormedFCA
```python
class UnnormedFCA(FunctionalComponentAnalysis)
```
FCA variant that doesn't normalize vectors.

---

#### OrthogonalProjection
```python
class OrthogonalProjection(nn.Module)
```
Trainable parameters differentiably constrained to be orthogonal to fixed vectors.

**Constructor:**
```python
OrthogonalProjection(
    size: int,
    n_params: int,
    fixed_vectors: list = None,
    normalize: bool = True,
    init_noise: float = 0.1,
)
```

**Methods:**
- `forward() -> Tensor` - Returns orthogonalized parameters (N, D)
- `set_fixed_vectors(vectors)` - Set vectors to be orthogonal to
- `add_fixed_vectors(vectors)` - Add more fixed vectors
- `clear_fixed_vectors()` - Remove all fixed vectors
- `check_orthogonality(tol=1e-5) -> dict` - Verify orthogonality

**Properties:**
- `weight: Tensor` - Alias for forward()
- `fixed_vectors: Tensor` - The fixed vectors (K, D)
- `num_fixed: int` - Number of fixed vectors
- `basis_rank: int` - Rank of fixed vector subspace

---

### Functions

#### attach_fca
```python
def attach_fca(
    model: nn.Module,
    layer_name: str,
    fca_instance: FunctionalComponentAnalysis,
    output_format: str = "auto",
    use_complement: bool = None,
    output_extractor: callable = None,
    shape_transform: callable = None,
    inverse_transform: callable = None,
    comms_dict: dict = None,
) -> handle
```
High-level function to attach FCA to a model layer.

**output_format options:** `"auto"`, `"tensor"`, `"image"`, `"sequence"`, `"tuple"`, `"dict"`, `"dict:key_name"`

---

#### get_layer_output_size
```python
def get_layer_output_size(
    model: nn.Module,
    layer_name: str,
    sample_input: Tensor = None,
) -> int
```
Determine output size of a layer.

---

#### Orthogonalization Functions

```python
def orthogonalize_vector(
    new_vector: Tensor,
    prev_vectors: list,
    prev_is_mtx_sqr: bool = False,
    norm: bool = True,
    double_precision: bool = False,
    method: str = "classical",
) -> Tensor
```

```python
def orthogonalize_vector_mgs(
    new_vector: Tensor,
    prev_vectors: list,
    norm: bool = True,
    double_precision: bool = False,
) -> Tensor
```

```python
def orthogonalize_batch_qr(
    vectors: list,
    double_precision: bool = False,
) -> list
```

```python
def gram_schmidt(
    vectors: list,
    old_vectors: list = None,
    double_precision: bool = False,
) -> list
```

---

#### Loading Functions

```python
def load_fca_from_path(save_dir: str) -> FunctionalComponentAnalysis
```

```python
def load_fcas_from_path(file_path: str) -> dict
```

```python
def load_fcas(
    model: nn.Module,
    load_path: str,
    use_complement_in_hook: bool = True,
    ret_paths: bool = False,
    verbose: bool = False,
) -> tuple
```

```python
def load_ortho_fcas(
    fca: FunctionalComponentAnalysis,
    fca_save_list: list,
)
```

---

## fca.utils

### Activation Collection

```python
def collect_activations(
    model: nn.Module,
    input_data: Tensor,
    attention_mask: Tensor = None,
    pad_mask: Tensor = None,
    task_mask: Tensor = None,
    layers: list = None,
    comms_dict: dict = None,
    batch_size: int = 500,
    to_cpu: bool = True,
    ret_attns: bool = False,
    ret_preds: bool = False,
    tforce: bool = False,
    n_steps: int = 0,
    ret_gtruth: bool = False,
    verbose: bool = False,
) -> dict
```

```python
def collect_activations_using_loader(
    model: nn.Module,
    data_loader,
    layers: list = None,
    comms_dict: dict = None,
    to_cpu: bool = True,
    ret_attns: bool = False,
    ret_preds: bool = False,
    tforce: bool = False,
    n_steps: int = 0,
    ret_gtruth: bool = False,
    padding_side: str = "right",
    verbose: bool = False,
) -> dict
```

### Output Extractors

```python
def identity_extractor(output) -> Tensor
def first_element_extractor(output) -> Tensor
def dict_extractor(key: str = "hidden_states") -> callable
def last_hidden_state_extractor(output) -> Tensor
```

### Shape Transforms

```python
def image_to_flat(tensor: Tensor) -> tuple[Tensor, tuple]
def flat_to_image(tensor: Tensor, original_shape: tuple) -> Tensor

def sequence_to_flat(tensor: Tensor) -> tuple[Tensor, tuple]
def flat_to_sequence(tensor: Tensor, original_shape: tuple) -> Tensor

def vit_to_flat(tensor: Tensor, include_cls: bool = True) -> tuple[Tensor, tuple]
def flat_to_vit(tensor: Tensor, original_shape: tuple) -> Tensor

def fca_image_prep(X: Tensor, og_shape: tuple = None, inverse: bool = False) -> Tensor
```

### File I/O

```python
def load_json(file_name: str) -> dict
def load_yaml(file_name: str) -> dict
def load_json_or_yaml(file_name: str) -> dict
def save_json(data: dict, file_name: str)
```

### Model Utilities

```python
def get_output_size(
    model: nn.Module,
    layer_name: str,
    data_sample: dict = None,
) -> int
```

```python
def save_model_checkpt(
    model: nn.Module,
    save_path: str,
    config: dict = None,
)
```

### Miscellaneous

```python
def pad_to(
    arr,
    tot_len: int,
    fill_val = 0,
    side: str = "right",
    dim: int = -1,
)
```

```python
def arglast(mask: Tensor, dim: int = None, axis: int = -1) -> Tensor
```

```python
def extract_ids(string: str, tokenizer) -> Tensor
```

```python
def get_command_line_args(args: list = None) -> dict
```

---

## fca.projections

### PCA

```python
def perform_pca(
    X: Tensor,
    n_components: int = None,
    scale: bool = True,
    center: bool = True,
    transform_data: bool = False,
    full_matrices: bool = False,
    randomized: bool = False,
    use_eigen: bool = True,
    batch_size: int = None,
    verbose: bool = True,
) -> dict
```

```python
def perform_eigen_pca(
    X: Tensor,
    n_components: int = None,
    scale: bool = True,
    center: bool = True,
    transform_data: bool = False,
    batch_size: int = None,
    verbose: bool = True,
) -> dict
```

### Variance Analysis

```python
def explained_variance(
    preds: Tensor,
    labels: Tensor,
    eps: float = 1e-8,
    mean_over_dims: bool = False,
) -> Tensor
```

```python
def projinv_expl_variance(x: Tensor, W: Tensor) -> Tensor
```

```python
def lost_variance(x: Tensor, W: Tensor) -> Tensor
```

```python
def component_wise_expl_var(
    actvs: Tensor,
    weight: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]
```

### Matrix Operations

```python
def matrix_projinv(x: Tensor, W: Tensor) -> Tensor
```

```python
def get_cor_mtx(
    X: Tensor,
    Y: Tensor,
    batch_size: int = 500,
    to_numpy: bool = False,
    zscore: bool = True,
    device = None,
) -> Tensor
```

---

## fca.wrappers

### Model Wrappers

```python
class CategoricalModelWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: callable = None,
        acc_fn: callable = None,
    )

    def forward(self, fca_ref=None, **kwargs) -> dict
```

```python
class ContinuousModelWrapper(CategoricalModelWrapper):
    # Same interface, uses MSE loss
```

### Data Wrappers

```python
class DataWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict,
        complement_data: dict = None,
        collate_fn: callable = None,
    )
```

```python
def wrap_data(
    data: dict,
    complement_data: dict = None,
    shuffle: bool = True,
    batch_size: int = 128,
) -> DataLoader
```

### Utilities

```python
def wrapped_kl_divergence(
    preds: Tensor,
    labels: Tensor,
    preds_are_logits: bool = True,
) -> Tensor
```

---

## fca.schedulers

### Learning Rate Schedulers

```python
class DecayScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps: int = 100,
        last_epoch: int = -1,
        verbose: bool = False,
        min_lr: float = 1e-10,
        lr: float = 1,
        lr_decay_exp: float = 0.25,
    )

    @staticmethod
    def calc_lr(
        step: int,
        warmup_steps: int,
        max_lr: float = 0.005,
        min_lr: float = 1e-7,
        decay_exp: float = 0.5,
    ) -> float
```

### Training Utilities

```python
class PlateauTracker:
    def __init__(
        self,
        patience: int = 20,
        plateau: float = 0.01,
        measure: str = "acc",
    )

    def update(self, loss: float, acc: float) -> bool
    def reset(self)
```

---

## Module Exports

### fca (top-level)
```python
__all__ = [
    # Core classes
    "FunctionalComponentAnalysis",
    "PCAFunctionalComponentAnalysis",
    "UnnormedFCA",
    # Convenience functions
    "attach_fca",
    "get_layer_output_size",
    # Loading functions
    "load_fca_from_path",
    "load_fcas_from_path",
    "load_fcas",
    "load_ortho_fcas",
    # Utilities
    "gram_schmidt",
    "orthogonalize_vector",
    "orthogonalize_vector_mgs",
    "orthogonalize_batch_qr",
    # Extractors
    "identity_extractor",
    "first_element_extractor",
    "dict_extractor",
    "last_hidden_state_extractor",
    # Transforms
    "image_to_flat",
    "flat_to_image",
    "sequence_to_flat",
    "flat_to_sequence",
    "vit_to_flat",
    "flat_to_vit",
]
```

### fca.utils
```python
__all__ = [
    "collect_activations",
    "collect_activations_using_loader",
    "identity_extractor",
    "first_element_extractor",
    "dict_extractor",
    "last_hidden_state_extractor",
    "image_to_flat",
    "flat_to_image",
    "sequence_to_flat",
    "flat_to_sequence",
    "vit_to_flat",
    "flat_to_vit",
    "fca_image_prep",
]
```

### fca.projections
```python
__all__ = [
    "matrix_projinv",
    "projinv_expl_variance",
    "lost_variance",
    "explained_variance",
    "component_wise_expl_var",
    "perform_pca",
]
```

### fca.wrappers
```python
__all__ = [
    "wrap_data",
    "DataWrapper",
    "ContinuousModelWrapper",
    "CategoricalModelWrapper",
]
```

### fca.schedulers
```python
__all__ = [
    "PlateauTracker",
    "DecayScheduler",
]
```
