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
