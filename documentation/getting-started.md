# Getting Started with FCA

This guide walks you through the basics of using Functional Component Analysis (FCA) to analyze and manipulate neural network representations.

## Installation

```bash
# Install from PyPI
pip install fca

# Or install from source for development
git clone https://github.com/grantsrb/fca.git
cd fca
pip install -e .
```

## Basic Concepts

### What Does FCA Do?

FCA learns a set of orthogonal vectors that can be used to transform neural network activations. When you attach FCA to a layer:

1. The layer's output activations are intercepted
2. Activations are projected onto the FCA subspace
3. The projected activations are projected back to the original space
4. This modified output continues through the rest of the network

This allows you to:
- Find minimal sufficient representations
- Remove specific information from activations
- Analyze what information is necessary for a task

### Key Parameters

- **size**: The dimensionality of the representation (number of features)
- **init_rank**: How many orthogonal components to start with
- **max_rank**: Maximum number of components allowed
- **orth_method**: Which orthogonalization algorithm to use

## Quick Start Examples

### Example 1: Simple MLP

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
x = torch.randn(4, 10)
output = model(x)

# Clean up when done
handle.remove()
```

### Example 2: Vision Model (ResNet)

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
```

### Example 3: Language Model (HuggingFace)

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

## Finding Layer Names

To see available layer names in your model:

```python
for name, module in model.named_modules():
    print(name, type(module).__name__)
```

## Using the FCA Object Directly

You can also use FCA without attaching it to a model:

```python
import torch
from fca import FunctionalComponentAnalysis

# Create FCA
fca = FunctionalComponentAnalysis(size=128, init_rank=10)

# Create some activations
activations = torch.randn(32, 128)  # Batch of 32, 128-dimensional vectors

# Project activations to FCA space
projected = fca(activations)  # Shape: (32, 10)

# Project back to original space
reconstructed = fca(projected, inverse=True)  # Shape: (32, 128)

# Or do both in one step
projinv = fca.projinv(activations)  # Shape: (32, 128)
```

## Freezing and Adding Components

FCA supports incremental training where you freeze learned components and add new ones:

```python
from fca import FunctionalComponentAnalysis

# Start with 1 component
fca = FunctionalComponentAnalysis(size=64, init_rank=1)

# ... train the first component ...

# Freeze current components
fca.freeze_parameters()

# Add a new trainable component
fca.add_component()

# Now only the new component is trainable
print(f"Total components: {fca.rank}")
print(f"Frozen: {len(fca.frozen_list)}")
print(f"Trainable: {len(fca.train_list)}")
```

## Choosing an Orthogonalization Method

FCA supports multiple orthogonalization methods via the `orth_method` parameter:

```python
# Fast but less stable at high ranks (default)
fca_fast = FunctionalComponentAnalysis(
    size=512, init_rank=20, orth_method="classical"
)

# More stable for high ranks
fca_stable = FunctionalComponentAnalysis(
    size=512, init_rank=100, orth_method="modified"
)

# Most stable, uses QR decomposition
fca_qr = FunctionalComponentAnalysis(
    size=512, init_rank=50, orth_method="householder"
)

# Balanced: fast for frozen, stable for trainable
fca_hybrid = FunctionalComponentAnalysis(
    size=512, init_rank=10, orth_method="hybrid"
)
```

See [Orthogonalization Methods](orthogonalization.md) for detailed explanations.

## Double Precision for Stability

For very high ranks, you may need double precision:

```python
fca = FunctionalComponentAnalysis(
    size=1024,
    init_rank=500,
    orth_with_doubles=True,  # Use float64 internally
)
```

## Next Steps

- [Core Classes](core-classes.md) - Detailed documentation of FCA classes
- [Orthogonalization Methods](orthogonalization.md) - Understanding numerical stability
- [Utilities](utilities.md) - Activation collection and transforms
- [API Reference](api-reference.md) - Complete function reference
