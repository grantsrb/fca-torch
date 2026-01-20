# Model and Data Wrappers

The `fca.wrappers` module provides lightweight wrappers for PyTorch models and data to make them compatible with FCA's training utilities.

## Model Wrappers

### CategoricalModelWrapper

Wraps models that produce categorical (classification) outputs. Makes models compatible with `fca.find_sufficient_components()`.

```python
from fca.wrappers import CategoricalModelWrapper
import torch.nn as nn

# Your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 5),  # 5-class classification
)

# Wrap it
wrapped_model = CategoricalModelWrapper(model)

# Now it returns loss and accuracy
outputs = wrapped_model(inputs=x, labels=y)
print(outputs["loss"])  # Cross-entropy loss
print(outputs["acc"])   # Accuracy
```

#### Constructor

```python
CategoricalModelWrapper(
    model,           # Your PyTorch model
    loss_fn=None,    # Custom loss function (default: cross_entropy)
    acc_fn=None,     # Custom accuracy function (default: exact match)
)
```

#### Forward Method

```python
result = wrapped_model(
    fca_ref=None,      # Optional FCA reference for complement training
    **kwargs,          # Model inputs (must include "labels" for loss/acc)
)
```

**Input Formats:**

1. **Simple format** - direct keyword arguments:
   ```python
   result = wrapped_model(inputs=x, labels=y)
   ```

2. **Data/complement format** - for complement training:
   ```python
   result = wrapped_model(
       data={"inputs": x, "labels": y},
       complement={"inputs": x_comp, "labels": y_comp},
       fca_ref=fca,
   )
   ```

#### Custom Loss and Accuracy

```python
def my_loss(logits, labels):
    return F.cross_entropy(logits, labels, label_smoothing=0.1)

def my_acc(preds, labels):
    return (preds == labels).float().mean()

wrapped = CategoricalModelWrapper(
    model,
    loss_fn=my_loss,
    acc_fn=my_acc,
)
```

#### With Masks

Support for masked loss/accuracy computation:

```python
result = wrapped_model(
    inputs=x,
    labels=y,
    mask=attention_mask,  # True = keep, False = ignore
)
```

### ContinuousModelWrapper

Wraps models that produce continuous (regression) outputs.

```python
from fca.wrappers import ContinuousModelWrapper

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1),  # Regression output
)

wrapped = ContinuousModelWrapper(model)

# Uses MSE loss and 1-MSE as "accuracy"
result = wrapped(inputs=x, labels=y)
```

#### Default Loss and Accuracy

- **Loss**: MSE (Mean Squared Error)
- **Accuracy**: `1 - MSE` (higher is better)

## Data Wrappers

### DataWrapper

Wraps dictionaries or datasets into PyTorch Dataset objects.

```python
from fca.wrappers import DataWrapper

# Simple dict of tensors
data = {
    "inputs": torch.randn(100, 10),
    "labels": torch.randint(0, 5, (100,)),
}

dataset = DataWrapper(data)
print(len(dataset))  # 100

# Access item
item = dataset[0]
# item = {"data": {"inputs": tensor, "labels": tensor}}
```

#### With Complement Data

For training FCA on both regular and complement data:

```python
data = {
    "inputs": torch.randn(100, 10),
    "labels": torch.randint(0, 5, (100,)),
}

complement_data = {
    "inputs": torch.randn(100, 10),  # Different inputs
    "labels": torch.randint(0, 5, (100,)),
}

dataset = DataWrapper(data, complement_data=complement_data)
item = dataset[0]
# item = {
#     "data": {"inputs": tensor, "labels": tensor},
#     "complement": {"inputs": tensor, "labels": tensor},
# }
```

### wrap_data()

Convenience function to create a DataLoader from dict data.

```python
from fca.wrappers import wrap_data

data = {
    "inputs": torch.randn(100, 10),
    "labels": torch.randint(0, 5, (100,)),
}

data_loader = wrap_data(
    data,
    complement_data=None,  # Optional complement data
    shuffle=True,
    batch_size=32,
)

for batch in data_loader:
    # batch["data"]["inputs"], batch["data"]["labels"]
    pass
```

## Complete Training Example

Here's a full example of training FCA to find sufficient components:

```python
import torch
import torch.nn as nn
from collections import OrderedDict
from fca import FunctionalComponentAnalysis
from fca.wrappers import CategoricalModelWrapper, wrap_data

# 1. Define model
base_model = nn.Sequential(OrderedDict([
    ("layer1", nn.Linear(10, 64)),
    ("relu", nn.ReLU()),
    ("layer2", nn.Linear(64, 3)),
]))

# Freeze model weights
for p in base_model.parameters():
    p.requires_grad = False

# Wrap for training
model = CategoricalModelWrapper(base_model)

# 2. Create FCA
fca = FunctionalComponentAnalysis(
    size=64,          # Match layer1 output
    init_rank=1,      # Start with 1 component
    max_rank=20,      # Allow up to 20
    orth_method="hybrid",
)

# 3. Prepare data
data = {
    "inputs": torch.randn(500, 10),
    "labels": torch.randint(0, 3, (500,)),
}
data_loader = wrap_data(data, shuffle=True, batch_size=32)

# 4. Attach FCA to layer
handle = fca.hook_model_layer(base_model, "layer1")

# 5. Train
optimizer = torch.optim.Adam(fca.parameters(), lr=0.001)

for epoch in range(50):
    total_loss = 0
    total_acc = 0
    n_batches = 0

    for batch in data_loader:
        optimizer.zero_grad()

        outputs = model(**batch)
        loss = outputs["loss"]
        acc = outputs["acc"]

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()
        n_batches += 1

    print(f"Epoch {epoch}: Loss={total_loss/n_batches:.4f}, Acc={total_acc/n_batches:.2%}")

    # Optionally add components when performance plateaus
    if should_add_component(epoch):
        fca.freeze_parameters()
        fca.add_component()
        optimizer = torch.optim.Adam(fca.parameters(), lr=0.001)

# 6. Clean up
handle.remove()

print(f"Final FCA rank: {fca.rank}")
```

## Complement Training

Train FCA to ensure the complement (what's removed) doesn't contain task-relevant information:

```python
from fca import FunctionalComponentAnalysis
from fca.wrappers import CategoricalModelWrapper, wrap_data

# Prepare data with complement (can be same or different)
data_loader = wrap_data(
    data={"inputs": x, "labels": y},
    complement_data={"inputs": x, "labels": y},  # Same data
    batch_size=32,
)

# Create FCA that removes components in hook
fca = FunctionalComponentAnalysis(size=64, init_rank=5)

# Training loop
for batch in data_loader:
    # Regular forward (FCA keeps components)
    main_output = model(fca_ref=fca, **batch)
    main_loss = main_output["loss"]

    # Complement forward (FCA removes components)
    comp_output = main_output.get("complement", {})
    comp_loss = comp_output.get("loss", 0)

    # Combined objective: maximize main accuracy, minimize complement accuracy
    total_loss = main_loss - 0.1 * comp_loss  # Adjust weight as needed
    total_loss.backward()
    optimizer.step()
```

## Utility Functions

### wrapped_kl_divergence()

Compute KL divergence for probability distributions.

```python
from fca.wrappers import wrapped_kl_divergence

# Compare model outputs
kl_div = wrapped_kl_divergence(
    preds=model_logits,        # Predicted logits
    labels=target_probs,        # Target probabilities
    preds_are_logits=True,      # Apply log_softmax to preds
)
```

Useful for distillation-style training where you want FCA-modified outputs to match original outputs.
