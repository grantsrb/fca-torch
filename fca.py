import torch
import torch.nn as nn

class FunctionalComponentAnalysis(nn.Module):
    """
    Functional Component Analysis (FCA) is a method for learning a set of
    orthogonal vectors that can be used to transform data. This is similar
    to PCA, but the vectors are learned in a more general way. The vectors
    are learned by sampling a random vector and orthogonalizing it to all
    previous vectors. This is done by projecting the new vector onto each
    previous vector and subtracting the projection. The new vector is then
    normalized. The process is repeated for each new vector. The vectors
    are stored as parameters in the model.

    The recommended usage is to first train a single component to
    convergence and then freeze it. Then, individually add new
    components and train each while keeping the previous components frozen.
    This process can be repeated to learn as many components as desired.
    """
    def __init__(self, size, max_rank=None, remove_components=False):
        """
        Args:
            size: int
                The size of the vectors to be learned.
            max_rank: int or None
                The maximum number of components to learn. If None,
                the maximum rank is equal to the size.
            remove_components: bool
                If True, the model will remove components from
                the vectors in the forward hook.
        """
        super(FunctionalComponentAnalysis, self).__init__()
        # Sample a single, initial vector and normalize it
        self.size = size
        self.max_rank = max_rank if max_rank is not None else size
        self.remove_components = remove_components
        initial_vector = torch.randn(size)
        initial_vector = initial_vector / torch.norm(initial_vector)
        self.parameters_list = nn.ParameterList([nn.Parameter(initial_vector)])
        self.is_fixed = False
        self.fixed_weight = None
        self.orthogonalization_vectors = []

    def add_orthogonalization_vectors(self, new_vectors):
        """
        Args:
            new_vectors: list of tensors
        """
        for vec in new_vectors:
            self.orthogonalization_vectors.append( vec.data )
        rank_diff = self.size-len(self.orthogonalization_vectors)
        self.max_rank = min(self.max_rank, rank_diff)
        print("New Max Rank:", self.max_rank)

    def freeze_parameters(self, freeze=True):
        for p in self.parameters_list:
            p.requires_grad = freeze

    def freeze_weights(self, freeze=True):
        self.freeze_parameters(freeze=freeze)

    def set_fixed(self, fixed=True):
        """
        Can fix the weight matrix in order to quit calculating
        orthogonality via gram schmidt.
        """
        if fixed:
            self.fixed_weight = self.weight.data.clone()
        self.is_fixed = fixed

    def orthogonalize_vector(self, new_vector, prev_vectors):
        # Make the new vector orthogonal to all previous vectors using Gram-Schmidt process
        for param in prev_vectors:
            projection = torch.dot(new_vector, param) * param
            new_vector = new_vector - projection
        # Normalize the new vector
        new_vector = new_vector / torch.norm(new_vector, 2)
        # Store the new vector as a parameter
        return new_vector

    def update_parameters(self):
        """
        Orthogonalize all parameters in the list and make a new
        parameter list. Does not track gradients.
        """
        # Sample a new vector and orthogonalize it
        device = self.get_device()
        self.orthogonalization_vectors = [
            v.to(device) for v in self.orthogonalization_vectors
        ]
        params = []
        with torch.no_grad():
            for p in self.parameters_list:
                p = self.orthogonalize_vector(
                    p, prev_vectors=params+self.orthogonalization_vectors)
                params.append(p)
        self.parameters_list = nn.ParameterList([
            nn.Parameter(p.data) for p in params
        ])

    def orthogonalize_parameters(self):
        """
        Only orthogonalize the parameters that require gradients.
        Does track gradients.
        """
        device = self.get_device()
        self.orthogonalization_vectors = [
            v.to(device) for v in self.orthogonalization_vectors
        ]
        params = []
        for p in self.parameters_list:
            if p.requires_grad:
                p = self.orthogonalize_vector(
                    p, prev_vectors=params+self.orthogonalization_vectors)
            params.append(p)
        return params

    def make_matrix(self, components=None):
        """
        Create a matrix from the stored parametersa.

        Args:
            components: None or torch LongTensor.
                If None, all components are used. If a
                LongTensor, only the components specified
                by the tensor are used.
        Returns:
            matrix: torch.Tensor.
                The low rank orthogonal matrix created from
                the parameter list.
        """
        if self.is_fixed and self.fixed_weight is not None:
            matrix = self.fixed_weight.to(self.get_device())
        else:
            params = self.orthogonalize_parameters()
            matrix = torch.vstack(params)
        if components is not None:
            matrix = matrix[components]
        return matrix

    @property
    def weight(self):
        return self.make_matrix()

    @property
    def rank(self):
        return len(self.parameters_list)

    def get_device(self):
        device = self.parameters_list[0].get_device()
        return "cpu" if device<0 else device

    def add_new_axis_parameter(self):
        if len(self.parameters_list) >= self.max_rank:
            return None
        # Sample a new axis and add it to the parameter list
        new_axis = torch.randn(self.parameters_list[0].shape[0])
        p = nn.Parameter(new_axis).to(self.get_device())
        self.parameters_list.append(p)
        return self.parameters_list[-1]

    def load_sd(self, sd):
        """
        Assists in loading a state dict
        """
        n_axes = 0
        for k in sd:
            if "parameters_list" in k:
                ax = int(k.split(".")[-1])
                if ax > n_axes:
                    n_axes = ax
        for _ in range(n_axes+1-self.rank):
            self.add_new_axis_parameter()
        try:
            self.load_state_dict(sd)
        except:
            print("Failed to load sd")
            print("Current sd:")
            for k in self.state_dict():
                print(k, self.state_dict()[k].shape)
            print("Argued sd:")
            for k in sd:
                print(k, sd[k].shape)
            self.load_state_dict(sd)

    def init_from_fca(self, fca, freeze_params=True):
        """
        Simplifies starting the parameters from another fca
        object
        """
        self.load_sd(fca.state_dict())
        self.freeze_parameters(freeze=freeze_params)
        p = self.add_new_axis_parameter()
        return p

    def get_forward_hook(self):
        def hook(module, input, output):
            fca_vec = self.forward(output)
            stripped = self.forward(fca_vec, inverse=True)
            if self.remove_components:
                stripped = output - stripped
            return stripped
        return hook

    def forward(self, x, inverse=False, components=None):
        if inverse:
            return torch.matmul(
                x, self.make_matrix(components=components)
            )
        return torch.matmul(
            x, self.make_matrix(components=components).T
        )

def load_fcas_from_path(file_path):
    fca_checkpoint = torch.load(file_path)
    fca_config = fca_checkpoint["config"]
    state_dicts = fca_checkpoint["fca_state_dicts"]
    fcas = {}
    kwargs = fca_config.get("fca_params", {})
    for layer in fca_config["fca_layers"]:
        sd = state_dicts[layer]
        kwargs["size"] = sd[list(sd.keys())[0]].shape[0]
        fcas[layer] = FunctionalComponentAnalysis( **kwargs )
        fcas[layer].load_sd(sd)
        fcas[layer].update_parameters()
        fcas[layer].freeze_parameters()
        fcas[layer].set_fixed(True)
    return fcas
    
def load_fcas(model, load_path, remove_components=True, verbose=False):
    """
    Simplifies the recursive loading of previous fcas.
    """
    device = "cpu" if model is None else model.get_device()
    if verbose:
        print("Loading:", load_path)
    fca_checkpoint = torch.load(load_path)
    fca_config = fca_checkpoint["config"]
    
    fcas = {}
    handles = {}
    loaded_fcas = []
    loaded_handles = []
    if fca_config.get("fca_load_path", None) is not None:
        loaded_fcas, loaded_handles = load_fcas(
            model=model,
            load_path=fca_config["fca_load_path"],
            remove_components=remove_components,
            verbose=verbose,
        )
    state_dicts = fca_checkpoint["fca_state_dicts"]
    kwargs = fca_config.get("fca_params", {})
    modules = {}
    if model is not None:
        for layer,modu in model.named_modules():
            modules[layer] = modu
    for layer in state_dicts:
        sd = state_dicts[layer]
        kwargs["size"] = sd[list(sd.keys())[0]].shape[0]
        kwargs["remove_components"] = remove_components
        print("Remove Out:", remove_components)
        fcas[layer] = FunctionalComponentAnalysis( **kwargs )
        fcas[layer].load_sd(sd)
        fcas[layer].update_parameters()
        fcas[layer].freeze_parameters()
        fcas[layer].set_fixed(True)
        fcas[layer].to(device)
        if model is not None and layer in modules:
            h = modules[layer].register_forward_hook(
                fcas[layer].get_forward_hook()
            )
            handles[layer] = h
            print("adding handle:")
    loaded_handles.append(handles)
    loaded_fcas.append(fcas)
    return loaded_fcas, loaded_handles

def initialize_fcas(model, config, loaded_fcas=[]):
    device = model.get_device()
    fca_handles = []
    fca_parameters = []
    fcas = {}
    handles = {}
    fca_layers = config["fca_layers"]
    kwargs = config.get("fca_params", {})
    for name,modu in model.named_modules():
        if name in fca_layers:
            kwargs["size"] = modu.weight.shape[0]
            fcas[name] = FunctionalComponentAnalysis(
                **kwargs
            )
            fcas[name].to(device)
            if config.get("ensure_ortho_chain", False):
                if loaded_fcas:
                    for loaded in loaded_fcas:
                        if name in loaded:
                            print("Loading Orthogonalization Vectors", name)
                            fcas[name].add_orthogonalization_vectors(
                                loaded[name].parameters_list)
            h = modu.register_forward_hook(
                fcas[name].get_forward_hook()
            )
            handles[name] = h
            fca_parameters += list(fcas[name].parameters())
    return fcas, handles, fca_parameters
    

# Example usage
if __name__ == "__main__":
    n_dim = 5
    fca = FunctionalComponentAnalysis(size=n_dim)
    for i in range(n_dim):
        fca.add_new_axis_parameter()

    # Assert that the vectors are orthogonal
    fca.update_parameters()
    for i in range(len(fca.parameters_list)):
        for j in range(i):
            assert torch.dot(fca.parameters_list[i], fca.parameters_list[j]) < 1e-6
    print("Rank:", fca.rank)
    vec = torch.randn(1,n_dim)
    mtx = fca.weight[:3]
    rot = torch.matmul(vec, mtx.T)
    new_vec = torch.matmul(rot, mtx)
    diff = vec - new_vec
    zero = torch.matmul(diff, mtx.T)

    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("Diff:", diff)
    print("Zero:", zero)
    print("MSE:", ((vec-new_vec)**2).sum())
    print()

    mtx = fca.weight[3:]
    rot = torch.matmul(vec, mtx.T)
    nnew_vec = torch.matmul(rot, mtx)
    ddiff = vec - nnew_vec
    zzero = torch.matmul(ddiff, mtx.T)

    print("Old Vec:", vec)
    print("New Vec:", nnew_vec)
    print("Diff:", ddiff)
    print("Zero:", zzero)
    print("MSE:", ((vec-new_vec-nnew_vec)**2).sum())
    print()
    
    rot = fca(vec)
    new_vec = fca(rot, inverse=True)
    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("MSE:", ((vec-new_vec)**2).sum())
