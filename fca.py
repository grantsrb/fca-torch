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
        self.max_rank = max_rank if max_rank else size
        self.remove_components = remove_components
        initial_vector = torch.randn(size)
        initial_vector = initial_vector / torch.norm(initial_vector)
        self.parameters_list = nn.ParameterList([nn.Parameter(initial_vector)])

    def freeze_parameters(self):
        for p in self.parameters_list:
            p.requires_grad = False

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
        params = []
        with torch.no_grad():
            for p in self.parameters_list:
                p = self.orthogonalize_vector(p, prev_vectors=params)
                params.append(p)
        self.parameters_list = nn.ParameterList([
            nn.Parameter(p.data) for p in params
        ])

    def orthogonalize_parameters(self):
        """
        Only orthogonalize the parameters that require gradients.
        Does track gradients.
        """
        params = []
        for p in self.parameters_list:
            if p.requires_grad:
                p = self.orthogonalize_vector(p, prev_vectors=params)
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

    def add_new_axis_parameter(self):
        if len(self.parameters_list) >= self.max_rank:
            return None
        # Sample a new axis and add it to the parameter list
        new_axis = torch.randn(self.parameters_list[0].shape[0])
        p = nn.Parameter(new_axis)
        self.parameters_list.append(p)
        return p

    def load_sd(self, sd):
        """
        Assists in loading a state dict
        """
        for k in sd:
            if "parameters_list" in k:
                self.add_new_axis_parameter()
        self.load_state_dict(sd)

    def get_forward_hook(self):
        def hook(module, input, output):
            fca_vec = self.forward(output)
            stripped = self.forward(fca_vec, inverse=True)
            if self.remove_components:
                stripped = output - stripped
                #assert torch.matmul(stripped, self.weight.T).sum()<1e-6
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

# Example usage
if __name__ == "__main__":
    n_dim = 5
    fca = FunctionalComponentAnalysis(size=n_dim)
    print("Params:", fca.parameters_list)
    for i in range(n_dim):
        fca.add_new_axis_parameter()
        print("Params:", fca.parameters_list)

    # Assert that the vectors are orthogonal
    fca.update_parameters()
    for i in range(len(fca.parameters_list)):
        for j in range(i):
            assert torch.dot(fca.parameters_list[i], fca.parameters_list[j]) < 1e-6
    print("Rank:", fca.rank)
    vec = torch.randn(n_dim)
    mtx = fca.make_matrix()
    rot = torch.matmul(vec, mtx)
    new_vec = torch.matmul(rot, mtx.T)
    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("MSE:", ((vec-new_vec)**2).sum())
    
    rot = fca(vec)
    new_vec = fca(rot, inverse=True)
    print("Old Vec:", vec)
    print("New Vec:", new_vec)
    print("MSE:", ((vec-new_vec)**2).sum())