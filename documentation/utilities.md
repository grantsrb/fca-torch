# Utilities

The `fca.utils` module provides helper functions for collecting activations, extracting tensors from model outputs, and transforming tensor shapes.

## Activation Collection

### `collect_activations()`

Collects activations from specified layers during a forward pass.

```python
from fca.utils import collect_activations

outputs = collect_activations(
    model=model,
    input_data=inputs,           # (N, ...) input tensor
    attention_mask=None,         # Optional attention mask
    layers=["layer1", "layer5"], # Layers to collect from
    batch_size=500,              # Process in batches
    to_cpu=True,                 # Move results to CPU
    ret_preds=False,             # Return predictions
    verbose=False,               # Show progress bar
)

# outputs is a dict: {layer_name: activations_tensor}
activations = outputs["layer1"]  # Shape depends on layer
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | nn.Module | The model to collect from |
| `input_data` | Tensor | Input data (N, ...) |
| `attention_mask` | Tensor | Optional attention mask |
| `pad_mask` | Tensor | Optional padding mask |
| `task_mask` | Tensor | Optional task mask |
| `layers` | list[str] | Layer names to collect from |
| `batch_size` | int | Batch size for processing |
| `to_cpu` | bool | Move results to CPU |
| `ret_attns` | bool | Return attention values |
| `ret_preds` | bool | Return prediction IDs and logits |
| `verbose` | bool | Show progress bar |

### `collect_activations_using_loader()`

Same as above but accepts a DataLoader instead of raw data.

```python
from fca.utils import collect_activations_using_loader

outputs = collect_activations_using_loader(
    model=model,
    data_loader=data_loader,
    layers=["encoder.layer.5"],
    to_cpu=True,
    verbose=True,
)
```

## Output Extractors

These functions extract tensors from various model output formats. Use them with `attach_fca()` or `get_forward_hook()`.

### `identity_extractor(output)`

For models that output raw tensors directly.

```python
from fca.utils import identity_extractor

# Returns the tensor unchanged
tensor = identity_extractor(model_output)
```

### `first_element_extractor(output)`

For models that output `(tensor, ...)` tuples.

```python
from fca.utils import first_element_extractor

# Returns first element of tuple, or tensor if not tuple
tensor = first_element_extractor(model_output)
```

### `dict_extractor(key)`

Factory function for models that output dictionaries.

```python
from fca.utils import dict_extractor

# Create an extractor for a specific key
extractor = dict_extractor("hidden_states")
tensor = extractor(model_output)

# Works with both dicts and objects with attributes
extractor = dict_extractor("last_hidden_state")
```

### `last_hidden_state_extractor(output)`

Specifically for HuggingFace model outputs.

```python
from fca.utils import last_hidden_state_extractor

# Works with HuggingFace BaseModelOutput
tensor = last_hidden_state_extractor(model_output)
```

## Shape Transforms

These functions transform tensors between different shapes for FCA processing. Each returns `(flat_tensor, original_shape)` where `original_shape` is used by the inverse transform.

### Image Transforms

#### `image_to_flat(tensor)`

Transforms `(B, C, H, W)` image tensors to `(B*H*W, C)` for FCA.

```python
from fca.utils import image_to_flat, flat_to_image

# Image tensor: 2 images, 64 channels, 32x32 pixels
img = torch.randn(2, 64, 32, 32)

# Flatten for FCA
flat, shape = image_to_flat(img)
print(flat.shape)  # (2048, 64) - 2*32*32 samples, 64 features

# After FCA processing, restore shape
restored = flat_to_image(processed_flat, shape)
print(restored.shape)  # (2, 64, 32, 32)
```

#### `flat_to_image(tensor, original_shape)`

Inverse of `image_to_flat`.

### Sequence Transforms

#### `sequence_to_flat(tensor)`

Transforms `(B, S, D)` sequence tensors to `(B*S, D)` for FCA.

```python
from fca.utils import sequence_to_flat, flat_to_sequence

# Sequence tensor: 4 sequences, 128 tokens, 768 hidden dim
seq = torch.randn(4, 128, 768)

# Flatten for FCA
flat, shape = sequence_to_flat(seq)
print(flat.shape)  # (512, 768) - 4*128 samples, 768 features

# After FCA processing, restore shape
restored = flat_to_sequence(processed_flat, shape)
print(restored.shape)  # (4, 128, 768)
```

#### `flat_to_sequence(tensor, original_shape)`

Inverse of `sequence_to_flat`.

### Vision Transformer Transforms

#### `vit_to_flat(tensor, include_cls=True)`

Transforms ViT outputs `(B, N+1, D)` to flat format, optionally excluding CLS token.

```python
from fca.utils import vit_to_flat, flat_to_vit

# ViT output: 2 images, 196 patches + 1 CLS token, 768 dim
vit_out = torch.randn(2, 197, 768)

# Include CLS token
flat, shape = vit_to_flat(vit_out, include_cls=True)
print(flat.shape)  # (394, 768)

# Exclude CLS token (only patches)
flat, shape = vit_to_flat(vit_out, include_cls=False)
print(flat.shape)  # (392, 768) - 2*196 patches
```

#### `flat_to_vit(tensor, original_shape)`

Inverse of `vit_to_flat`.

### Legacy Transform

#### `fca_image_prep(X, og_shape=None, inverse=False)`

Older image preparation function. Use `image_to_flat`/`flat_to_image` for new code.

```python
from fca.utils import fca_image_prep

# Forward
flat = fca_image_prep(img)

# Inverse
restored = fca_image_prep(flat, og_shape=original_shape, inverse=True)
```

## Using Transforms with attach_fca

```python
from fca import FunctionalComponentAnalysis, attach_fca
from fca.utils import image_to_flat, flat_to_image

fca = FunctionalComponentAnalysis(size=64, init_rank=10)

# For image models
handle = attach_fca(
    model=resnet,
    layer_name="layer1.0.conv2",
    fca_instance=fca,
    shape_transform=image_to_flat,
    inverse_transform=flat_to_image,
)

# Or use the shorthand
handle = attach_fca(
    model=resnet,
    layer_name="layer1.0.conv2",
    fca_instance=fca,
    output_format="image",  # Automatically uses image transforms
)
```

## File I/O Utilities

### `load_json(file_name)`

Load a JSON file as a dict.

```python
from fca.utils import load_json
config = load_json("config.json")
```

### `load_yaml(file_name)`

Load a YAML file as a dict.

```python
from fca.utils import load_yaml
config = load_yaml("config.yaml")
```

### `load_json_or_yaml(file_name)`

Automatically detects format from extension.

```python
from fca.utils import load_json_or_yaml
config = load_json_or_yaml("config.yaml")  # or "config.json"
```

### `save_json(data, file_name)`

Save a dict to JSON with automatic type conversion.

```python
from fca.utils import save_json
save_json({"key": "value"}, "output.json")
```

## Model Utilities

### `get_output_size(model, layer_name, data_sample=None)`

Determines the output size of a layer.

```python
from fca.utils import get_output_size

# From model structure
size = get_output_size(model, "layer1.0.conv2")

# With sample data for accuracy
sample = {"input_ids": torch.randint(0, 1000, (1, 10))}
size = get_output_size(model, "encoder.layer.5", data_sample=sample)
```

### `save_model_checkpt(model, save_path, config=None)`

Save model state dict and optional config.

```python
from fca.utils import save_model_checkpt

save_model_checkpt(
    model=model,
    save_path="checkpoints/model",
    config={"epochs": 100, "lr": 0.001},
)
# Creates: checkpoints/model.pt, checkpoints/model_config.json
```

## Padding Utilities

### `pad_to(arr, tot_len, fill_val=0, side="right", dim=-1)`

Pad arrays/tensors to a target length.

```python
from fca.utils import pad_to

# Pad list
lst = [1, 2, 3]
padded = pad_to(lst, 5)  # [1, 2, 3, 0, 0]

# Pad tensor
tensor = torch.randn(4, 10)
padded = pad_to(tensor, 15, dim=-1)  # Shape: (4, 15)

# Left padding
padded = pad_to(tensor, 15, side="left")
```

## Miscellaneous

### `arglast(mask, dim=None)`

Find the index of the last True value along a dimension.

```python
from fca.utils import arglast

mask = torch.tensor([True, True, False, True, False])
idx = arglast(mask)  # 3
```

### `extract_ids(string, tokenizer)`

Extract token IDs from a string (for HuggingFace tokenizers).

```python
from fca.utils import extract_ids

ids = extract_ids("Hello world", tokenizer)
```

### `get_command_line_args(args=None)`

Parse command line arguments with automatic type conversion.

```python
from fca.utils import get_command_line_args

# python script.py config.yaml lr=0.001 epochs=100
config = get_command_line_args()
# config = {"lr": 0.001, "epochs": 100, ...config.yaml contents...}
```
