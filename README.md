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

```python
from collections import OrderedDict
import torch
import torch.nn as nn
from fca import FunctionalComponentAnalysis
from fca.wrappers import wrap_data, CategoricalModelWrapper


# Load or define your model
in_dim = 2
d_model = 24 # The dimensionality of your model
out_dim = 2
model = CategoricalModelWrapper(
    nn.Sequential(OrderedDict([
        ("layer1", nn.Linear(in_dim, d_model)),
        ("layer2", nn.Linear(d_model, out_dim)),
    ]))
)
for p in model.parameters(): p.requires_grad = False # Freeze weights

# Create the FCA Object
target_layer = "layer1" # The name of the layer you wish to analyze
n_components = 2 # The number of initial fca components
max_rank = 2 # sets a limit on how many components can be used for finding
    # a functionally sufficient circuit
orth_with_doubles = True # Orthogonalize using double precision
fca_object = FunctionalComponentAnalysis(
    size=d_model,
    init_rank=1, # Can argue an intitial number of components here
    max_rank=max_rank,
    orth_with_doubles=orth_with_doubles,
)
# Can add new components using the `add_component` function
for _ in range(n_components-1):
    fca_object.add_component()

# Create the dataset. The keyword "labels" is treated differently, otherwise
# use the keywords that your model's forward function uses. If it does
# not use kwargs as in this example, any keyword will work. "inputs" is arbitrary
n_samples = 100
raw_data = {
    "inputs": torch.randn(n_samples, in_dim),
    "labels": torch.randint(out_dim,(n_samples,)),
}
complement_data = { # Can set to None if don't want to train FCA complement
    "inputs": torch.randn(n_samples, in_dim),
    "labels": torch.randint(out_dim,(n_samples,)),
}
data_loader = wrap_data(
    data=raw_data,
    complement_data=complement_data,
    shuffle=True,
    batch_size=128,
)

# Run FCA to find functionally sufficient components
n_samples = 100
fca_object.find_sufficient_components(
    model=model,
    layer=target_layer,
    data_loader=data_loader,
    n_epochs=100, # set a limit on the number of data iterations
    acc_threshold=0.99, # stop when the FCA object achieves this accuracy
        # on the desired behavior or when epoch limit is reached
)

# Inspect the components
print("Minimal sufficient component vectors:", fca_object.weight)

```

## 🧰 API Highlights

### `FunctionalComponentAnalysis()`
Initializes the FCA analyzer for a given model and layer of interest.

### `find_sufficient_components(input, target, strategy='greedy', threshold=0.05)`
Finds a subset of components (e.g., neurons or features) in the layer that are sufficient to approximate the original model output.

### `ablate_components(indices)`
Zeroes out selected components for functional probing.

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
