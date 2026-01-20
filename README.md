# `fca`: Functional Component Analysis

**Functional Component Analysis (FCA)** is a toolkit for identifying **functionally sufficient bottleneck circuits** in PyTorch-based neural networks. It enables researchers and developers to analyze trained models and isolate representational subspaces in artificial neural networks that are sufficient to produce the model's behavior. The provided tools additionally assist with performing the Gram-Schmidt process which can be used for efficient subspace training in methods like [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) and [Model Alignment Search](https://arxiv.org/abs/2501.06164).

## 🔍 Overview

Neural networks often have redundant representations which can be compressed into a lower dimensionality while maintaining the same behavior. FCA can be useful for finding compressed representations, for proving that orthogonal subspaces can produce the same behavior, and for separating behaviorally relevant activity into orthogonal subspaces. Furthermore, the tools provided in this package can assist in building efficient tools for causal interpretability methods such as DAS and MAS.

FCA provides tools to:
- **Probe network sufficiency** by identifying behaviorally sufficient representations.
- **Isolate behavioral activity** to compartmentalize behavioral components of a distributed network.
- Support **interpretability**, **sparsification**, and **circuit-level analysis** of black-box models.
- Examine information loss across matrix multiplications

## 🚀 Installation

You can install `fca` via pip:

```bash
pip install fca
```

## 🧠 Key Features

- ⚙️ Drop-in analysis tools for trained PyTorch models
- 🔬 Methods to isolate sufficient sets of units for a given task or output  
- 🔁 Support for iterative pruning and greedy selection strategies  
- 📉 Hooks for neuron ablation and reconstruction
- 🔌 Compatible with arbitrary pytorch-based architectures (CNNs, RNNs, transformers, etc.)  

## 🧪 Example Usage

### Quick Start: Simple MLP

```python
import torch
import torch.nn as nn
from fca import FunctionalComponentAnalysis, attach_fca

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)
model.eval()

# Create FCA for the first linear layer (64 output features)
fca = FunctionalComponentAnalysis(size=64, init_rank=5)

# Attach FCA to the layer
handle = attach_fca(
    model=model,
    layer_name="0",  # First layer in Sequential
    fca_instance=fca,
    output_format="tensor",
)

# Run inference - FCA will project activations
output = model(torch.randn(4, 10))
handle.remove()  # Clean up when done
```

### Vision Models (ResNet, CNN)

```python
import torch
import torchvision.models as models
from fca import FunctionalComponentAnalysis, attach_fca

# Load pretrained ResNet
model = models.resnet18(weights="IMAGENET1K_V1").eval()

# FCA for layer1 (64 channels)
fca = FunctionalComponentAnalysis(size=64, init_rank=10)

# Attach with image format - handles (B, C, H, W) tensors
handle = attach_fca(
    model=model,
    layer_name="layer1.0.conv2",
    fca_instance=fca,
    output_format="image",  # Automatically reshapes (B,C,H,W) -> (B*H*W,C)
)

# Run inference
output = model(torch.randn(1, 3, 224, 224))
handle.remove()

# For deeper layers with more channels
fca_deep = FunctionalComponentAnalysis(size=512, init_rank=20)
handle = attach_fca(
    model=model,
    layer_name="layer4.1.conv2",
    fca_instance=fca_deep,
    output_format="image",
)
```

### Language Models (HuggingFace Transformers)

```python
import torch
from transformers import AutoModel, AutoTokenizer
from fca import FunctionalComponentAnalysis, attach_fca

# Load BERT
model = AutoModel.from_pretrained("bert-base-uncased").eval()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# FCA for transformer hidden states (768 dimensions)
fca = FunctionalComponentAnalysis(size=768, init_rank=10)

# Attach to encoder layer output
handle = attach_fca(
    model=model,
    layer_name="encoder.layer.5.output.dense",
    fca_instance=fca,
    output_format="sequence",  # Handles (B, S, D) -> (B*S, D)
)

# Run inference
inputs = tokenizer("Hello world!", return_tensors="pt")
outputs = model(**inputs)
handle.remove()
```

### Custom Architectures

For models with custom output formats, use extractors and transforms:

```python
import torch
from fca import (
    FunctionalComponentAnalysis,
    attach_fca,
    dict_extractor,
    sequence_to_flat,
    flat_to_sequence,
)

# For a model that outputs {"features": tensor, "attention": tensor}
fca = FunctionalComponentAnalysis(size=256, init_rank=8)

handle = attach_fca(
    model=my_model,
    layer_name="encoder.layer3",
    fca_instance=fca,
    output_extractor=dict_extractor("features"),  # Extract 'features' key
    shape_transform=sequence_to_flat,
    inverse_transform=flat_to_sequence,
)
```

### Finding Sufficient Components (Full Training)

```python
from collections import OrderedDict
import torch
import torch.nn as nn
from fca import FunctionalComponentAnalysis
from fca.wrappers import wrap_data, CategoricalModelWrapper

# Define model
model = CategoricalModelWrapper(
    nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(2, 24)),
        ("layer2", nn.Linear(24, 2)),
    ]))
)
for p in model.parameters():
    p.requires_grad = False

# Create FCA
fca = FunctionalComponentAnalysis(
    size=24,
    init_rank=1,
    max_rank=10,
    orth_with_doubles=True,
)

# Prepare data
data = {
    "inputs": torch.randn(100, 2),
    "labels": torch.randint(2, (100,)),
}
data_loader = wrap_data(data, shuffle=True, batch_size=32)

# Train to find minimal sufficient components
fca.find_sufficient_components(
    model=model,
    layer="layer1",
    data_loader=data_loader,
    n_epochs=100,
    acc_threshold=0.99,
)

print(f"Found {fca.rank} sufficient components")
print("Component vectors:", fca.weight)
```

## 🧰 API Highlights

### Core Classes

#### `FunctionalComponentAnalysis(size, init_rank=1, max_rank=None, orth_method="classical", ...)`
The main FCA class for learning orthogonal functional components.

**Orthogonalization Methods** (`orth_method` parameter):
- `"classical"` (default): Fast matrix-based orthogonalization using covariance matrix caching. Best performance but may lose orthogonality at very high ranks (>100).
- `"modified"`: Modified Gram-Schmidt with O(e*k) error vs O(e*k^2) for classical. More numerically stable, recommended for high ranks.
- `"householder"`: QR decomposition using PyTorch's optimized LAPACK backend. Most stable method, best for batch initialization.
- `"hybrid"`: Classical for frozen components (fast), MGS for trainable components (stable). Recommended for training scenarios.

```python
# Example: Using different orthogonalization methods
fca_fast = FunctionalComponentAnalysis(size=512, init_rank=20, orth_method="classical")
fca_stable = FunctionalComponentAnalysis(size=512, init_rank=100, orth_method="modified")
fca_most_stable = FunctionalComponentAnalysis(size=512, init_rank=50, orth_method="householder")
fca_training = FunctionalComponentAnalysis(size=512, init_rank=10, orth_method="hybrid")

# Periodic re-orthogonalization during training
fca_training.reorthogonalize()  # Uses Householder QR to correct numerical drift
```

#### `PCAFunctionalComponentAnalysis(X, scale=True, center=True, ...)`
FCA initialized with PCA components from data.

### Convenience Functions

#### `attach_fca(model, layer_name, fca_instance, output_format="auto", ...)`
High-level function to attach FCA to any PyTorch model layer. Handles common output formats automatically.

#### `get_layer_output_size(model, layer_name, sample_input=None)`
Helper to determine the output size of a layer for FCA initialization.

### Output Extractors

Built-in extractors for common model output formats:
- `identity_extractor` - For raw tensor outputs
- `first_element_extractor` - For tuple outputs `(tensor, ...)`
- `dict_extractor(key)` - For dict outputs with custom keys
- `last_hidden_state_extractor` - For HuggingFace model outputs

### Shape Transforms

Built-in transforms for different tensor shapes:
- `image_to_flat` / `flat_to_image` - For (B, C, H, W) image tensors
- `sequence_to_flat` / `flat_to_sequence` - For (B, S, D) sequence tensors
- `vit_to_flat` / `flat_to_vit` - For Vision Transformer outputs

## 📚 Documentation

Full documentation and tutorials are available at: [https://your-docs-url](https://your-docs-url)

Includes:
- End-to-end examples
- Supported strategies (greedy, backward elimination, randomized search)
- Integration with experiment tracking frameworks

## 📦 Dependencies

- Python 3.7+
- PyTorch ≥ 1.9
- NumPy

Install requirements with:

```bash
pip install -r requirements.txt
```

## 🧑‍🔬 Citation

If you use `fca` in your research, please cite:

> Satchel Grant (2025). *Functional Component Analysis: Understanding Functionally Sufficient Circuits in Neural Networks*. [arXiv preprint](https://arxiv.org/abs/xxxx.xxxxx)

## 🛠 Development

Clone the repo and install in editable mode:

```bash
git clone https://github.com/grantsrb/fca.git
cd fca
pip install -e .
```

Run tests:

```bash
pytest tests/
```

## 🙌 Contributing

Contributions, suggestions, and issues are welcome! Open a pull request or file an issue.

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
