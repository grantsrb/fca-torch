# FCA Documentation

**Functional Component Analysis (FCA)** is a toolkit for identifying functionally sufficient bottleneck circuits in PyTorch-based neural networks. It enables researchers and developers to analyze trained models and isolate representational subspaces that are sufficient to produce the model's behavior.

## Table of Contents

1. [Getting Started](getting-started.md) - Installation and quick examples
2. [Core Classes](core-classes.md) - FCA, PCAFCA, and UnnormedFCA
3. [Orthogonalization Methods](orthogonalization.md) - Gram-Schmidt variants and numerical stability
4. [Utilities](utilities.md) - Activation collection, extractors, and transforms
5. [Projections](projections.md) - PCA and variance analysis
6. [Wrappers](wrappers.md) - Model and data wrappers for training
7. [API Reference](api-reference.md) - Complete function and class reference

## Overview

### What is FCA?

Neural networks often have redundant representations which can be compressed into a lower dimensionality while maintaining the same behavior. FCA provides tools to:

- **Probe network sufficiency** by identifying behaviorally sufficient representations
- **Isolate behavioral activity** to compartmentalize behavioral components of a distributed network
- Support **interpretability**, **sparsification**, and **circuit-level analysis** of black-box models
- Perform efficient **Gram-Schmidt orthogonalization** for methods like [Distributed Alignment Search](https://arxiv.org/abs/2303.02536) and [Model Alignment Search](https://arxiv.org/abs/2501.06164)

### Key Concepts

**Functional Components**: Orthogonal vectors that span a subspace of a neural network's representation space. When a model's activations are projected onto this subspace and back, the resulting behavior is preserved.

**Orthogonalization**: The process of ensuring all component vectors are mutually orthogonal and unit-length. FCA provides multiple methods with different trade-offs between speed and numerical stability.

**Forward Hooks**: PyTorch mechanism that FCA uses to intercept and modify activations during inference without modifying the model itself.

## Quick Example

```python
import torch
import torch.nn as nn
from fca import FunctionalComponentAnalysis, attach_fca

# Create a simple model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 2),
)

# Create FCA for the first layer (64 output features)
fca = FunctionalComponentAnalysis(
    size=64,           # Dimensionality of the representation
    init_rank=5,       # Number of initial components
    orth_method="hybrid",  # Orthogonalization method
)

# Attach FCA to the layer
handle = attach_fca(
    model=model,
    layer_name="0",
    fca_instance=fca,
    output_format="tensor",
)

# Run inference - FCA projects activations
output = model(torch.randn(4, 10))

# Clean up
handle.remove()
```

## Package Structure

```
fca/
├── fca.py          # Core FCA classes and orthogonalization functions
├── utils.py        # Activation collection, extractors, transforms
├── projections.py  # PCA and explained variance utilities
├── wrappers.py     # Model and data wrappers for training
├── schedulers.py   # Learning rate schedulers and plateau detection
└── __init__.py     # Package exports and attach_fca convenience function
```

## Dependencies

- Python 3.7+
- PyTorch >= 1.9
- NumPy
- scikit-learn (for randomized SVD)
- PyYAML (for configuration files)
- tqdm (for progress bars)

## Installation

```bash
# From PyPI
pip install fca

# From source (development)
git clone https://github.com/grantsrb/fca.git
cd fca
pip install -e .
```

## Citation

If you use `fca` in your research, please cite:

> Satchel Grant (2025). *Functional Component Analysis: Understanding Functionally Sufficient Circuits in Neural Networks*.
