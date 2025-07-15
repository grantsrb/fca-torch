import os
import torch
import torch.nn as nn
import numpy as np
import math
from .schedulers import PlateauTracker
from .projections import perform_pca, explained_variance
from .utils import fca_image_prep, load_json_or_yaml


def orthogonalize_vector(
        new_vector,
        prev_vectors,
        prev_is_mtx_sqr=False,
        norm=True,
        double_precision=False,
    ):
    """
    Orthogonalize a new vector to a list of previous vectors.
    Tracks gradients.
    Args:
        new_vector: tensor (S,)
            The new vector to be orthogonalized.
        prev_vectors: list of tensors [(S,), ...]
            The previous vectors to orthogonalize against. Assumes they
            are all orthogonal to one another.
        prev_is_mtx_sqr: bool
            If True, the previous vectors are assumed to be a covariance
            matrix. Otherwise, they are assumed to be a list of vectors.
            We parallelize the orthogonalization process by using
            matrix multiplication. Thus, we can precompute the mtx.T mtx
            product and use it to compute the projections.
        norm: bool
            If True, the new vector is normalized after orthogonalization.
        double_precision: bool
            if true, will case to double precision before orthogonalizing.
            Helps with numeric underflow issues.
    Returns:
        new_vector: tensor (S,)
            The orthogonalized vector.
    """
    mtx = prev_vectors
    og_dtype = new_vector.dtype
    if mtx is not None and len(mtx)>0:
        if type(mtx)==list:
            # Make matrix of previous vectors
            mtx = torch.vstack(mtx)
        if double_precision:
            new_vector = new_vector.double()
            mtx = mtx.double()
        # Compute projections
        if prev_is_mtx_sqr:
            proj_sum = torch.matmul(mtx, new_vector)
        else:
            proj_sum = torch.matmul(mtx.T, torch.matmul(mtx, new_vector))
        # Subtract projections
        new_vector = new_vector - proj_sum
    if norm:
        # Normalize vector
        new_vector = new_vector / torch.norm(new_vector, 2)
    return new_vector.type(og_dtype)

def gram_schmidt(vectors, old_vectors=None, double_precision=False):
    """
    Only orthogonalize the argued vectors using gram-schmidt.
    Tracks gradients.

    Args:  
        vectors: list of tensors [(S,), ...]
            The vectors to be orthogonalized.
        old_vectors: list of tensors [(S,), ...]
            Additional vectors to orthogonalize against.
            These vectors are not included in the returned
            list, nor are they orthogonalized.
        double_precision: bool
            if true, will case to double precision before orthogonalizing.
            Helps with numeric underflow issues.
    Returns:
        ortho_vecs: list of tensors [(S,), ...]
            The orthogonalized vectors.
    """
    if len(vectors)==0:
        return None
    if old_vectors is None: old_vectors = []
    ortho_vecs = []
    for i,v in enumerate(vectors):
        v = orthogonalize_vector(
            v,
            prev_vectors=ortho_vecs+old_vectors,
            double_precision=double_precision
        )
        ortho_vecs.append(v)
    return ortho_vecs

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
    def __init__(self,
                 size,
                 max_rank=None,
                 use_complement_in_hook=False,
                 component_mask=None,
                 means=None,
                 stds=None,
                 orthogonalization_vectors=None,
                 init_rank=None,
                 init_vectors=None,
                 init_noise=None,
                 orth_with_doubles=False,
                 *args, **kwargs):
        """
        Args:
            size: int
                The size of the vectors to be learned.
            max_rank: int or None
                The maximum number of components to learn. If None,
                the maximum rank is equal to the size.
            init_rank: int or None
                The initial number of components to learn. If None,
                the rank is 1
            use_complement_in_hook: bool
                If True, the model will remove components from
                the vectors in the forward hook.
            component_mask: tensor
                optionally argue a mask or index tensor to select
                specific components when constructing the matrix
            means: None or tensor (S,)
                optionally specify a tensor that will be used to
                center the representations before the FCA
            stds: None or tensor (S,)
                optionally specify a tensor that will be used to
                scale the representations before the FCA
            orthogonalization_vectors: None or list-like of tensors [(S,), ...]
                Adds a list of vectors to the list of vectors that are
                excluded from the functional components but are
                included for orthogonality calculations.
            init_vectors: None or list-like of tensors [(S,), ...]
                Adds a list of vectors to the parameters list
                without orthogonalizing them.
            init_noise: None or float
                Adds noise to the initialization vectors
            orth_with_doubles: bool
                if true, will orthogonalize vectors in double precision.
                This is useful for larger ranks, as the error increases
                with increasing rank.
        """
        super().__init__()
        self.size = size
        self.max_rank = max_rank if max_rank is not None else size
        self.use_complement_in_hook = use_complement_in_hook
        self.component_mask = component_mask
        self.orth_with_doubles = orth_with_doubles
        self.set_means(means)
        self.set_stds(stds)
        self.parameters_list = nn.ParameterList([])
        self.train_list = []
        self.frozen_list = []
        # Orthogonalization matrix defines a matrix for which the FCs
        # need to be orthogonal to.
        self.orthogonalization_mtx = None
        self.init_vectors = init_vectors
        self.init_noise = init_noise if init_noise is not None else 0.1
        if init_vectors is not None and init_rank is None:
            init_rank = len(init_vectors)
        else:
            init_rank = 1 if init_rank is None else init_rank
        for _ in range(init_rank):
            self.add_component()
        # If is_cached, the components are frozen in their orthogonal
        # normalized state. Can still track gradients.
        self.is_cached = False
        # cached weight is the weight matrix that is made up of cached
        # components.
        self.cached_weight = None
        # List of vectors that the FCs need to be orthogonal to.
        self.excl_ortho_list = []
        if orthogonalization_vectors is not None:
            self.add_excl_ortho_vectors(orthogonalization_vectors)

    def set_means(self, means):
        """
        Means are used to center the representations before extracting
        functional components.
        """
        if means is not None:
            means = means.squeeze()
        self.register_buffer("means", means)

    def set_stds(self, stds):
        """
        Standard deviations are used to scale the representations
        before extracting functional components.
        """
        if stds is not None:
            stds = stds.squeeze()
        self.register_buffer("stds", stds)

    def add_new_axis_parameter(self, init_vector=None):
        """
        Adds a new functional component vector. This increases the rank
        of the FCA matrix by 1. Does not allow the rank to exceed the
        max_rank property.

        Args:
            init_vector: torch Tensor (S,)
                optionally argue a new axis vector to initialize from.
                It will be orthogonalized to all other components and
                orthogonalization vectors.
        """
        if len(self.parameters_list) >= self.max_rank:
            return None
        # Sample a new axis and add it to the parameter list
        if init_vector is not None:
            new_axis = init_vector.clone()
        elif self.init_vectors is not None and self.rank<len(self.init_vectors):
            new_axis = self.init_vectors[self.rank].clone()
        else:
            new_axis = torch.randn(self.size)
        if self.init_noise is not None and self.init_noise>0:
            new_axis += torch.randn(self.size) * self.init_noise
        p = nn.Parameter(new_axis).to(self.get_device())
        self.parameters_list.append(p)
        self.train_list.append(p)
        return self.parameters_list[-1]

    def add_component(self, init_vector=None):
        """
        Args:
            init_vector: torch Tensor (S,)
                optionally argue a vector from which to initialize the
                new component from. It will be orthogonalized to all
                other components and orthogonalization vectors.
        """
        self.add_new_axis_parameter(init_vector=init_vector)

    def remove_component(self, idx=None):
        if idx is None: idx = -1
        was_cached = self.is_cached
        if self.is_cached: self.set_cached(False)
        del_p = self.parameters_list[idx]
        new_list = [p for p in self.parameters_list if p is not del_p]
        self.parameters_list = nn.ParameterList(new_list)
        self.train_list = [p for p in self.train_list if p is not del_p]
        self.frozen_list = [p for p in self.frozen_list if p is not del_p]
        if was_cached: self.set_cached(True)

    def remove_all_components(self):
        """
        Removes all functional components from the FCA.
        """
        for _ in range(self.rank-1):
            self.remove_component()

    def update_orthogonalization_mtx(
            self,
            orthogonalize=False,
            double_precision=False,
    ):
        """
        Creates and updates the main orthogonal matrix.
        This function concatenates the excluded orthogonalization
        parameters with the cached parameters (that are possibly
        being trained).
        """
        if len(self.excl_ortho_list)==0 and len(self.frozen_list)==0:
            self.orthogonalization_mtx = []
            self.ortho_mtx_sqr = []
            return
        device = self.get_device()
        vecs = [p.data.to(device) for p in self.excl_ortho_list] +\
               [p.data.to(device) for p in self.frozen_list]
        with torch.no_grad():
            if orthogonalize:
                vecs = gram_schmidt(
                    vecs, double_precision=self.orth_with_doubles)
        self.orthogonalization_mtx = torch.vstack( vecs )
        # This can be used for efficient orthogonalization
        # because we orthogonalize by first projecting into the
        # orthogonal components and then subtracting the orthogonal
        # vectors from the new vector, scaling each ortho vector
        # by the new vector's projection.
        self.ortho_mtx_sqr = torch.matmul(
            self.orthogonalization_mtx.T, self.orthogonalization_mtx
        )

    def add_excl_ortho_vectors(self, new_vectors):
        """
        Adds a list of vectors to the list of vectors that are
        excluded from the functional components but are
        included for orthogonality calculations.

        Args:
            new_vectors: list of tensors
        """
        new_vectors = gram_schmidt(
            new_vectors,
            self.excl_ortho_list,
            double_precision=self.orth_with_doubles,
        )
        for v in new_vectors:
            self.excl_ortho_list.append(v)
        rank_diff = self.size-len(self.excl_ortho_list)
        self.max_rank = min(self.max_rank, rank_diff)
        print("New Max Rank:", self.max_rank)
        self.update_orthogonalization_mtx()

    def add_orthogonalization_vectors(self, new_vectors):
        """
        Duplicate function with more intuitive name.
        """
        self.add_excl_ortho_vectors(new_vectors)

    def freeze_parameters(self, freeze=True):
        frozen_list = []
        train_list = []
        for p in self.parameters_list:
            p.requires_grad = not freeze
            if freeze:
                frozen_list.append(p)
            else:
                train_list.append(p)
        self.train_list = train_list
        self.frozen_list = frozen_list
        self.update_orthogonalization_mtx(orthogonalize=True)

    def freeze_weights(self, freeze=True):
        """
        Potentially better method name
        """
        self.freeze_parameters(freeze=freeze)

    def set_cached(self, cached=True):
        """
        Can fix the weight matrix in order to quit calculating
        orthogonality via gram schmidt.
        """
        if cached:
            self.cached_weight = self.weight
        self.is_cached = cached

    def reset_cached_weight(self,):
        """
        Reorthogonalizes the parameters and stores them as a matrix in
        cached_weight.
        """
        params = self.orthogonalize_parameters_with_grad()
        matrix = torch.vstack(params)
        self.cached_weight = matrix

    def orthogonalize_vector(self,
        new_vector,
        prev_vectors,
        prev_is_mtx_sqr=False,
        norm=True,
    ):
        """
        Orthogonalize a new vector to a list of previous vectors.
        Tracks gradients.

        Args:
            new_vector: tensor (S,)
                The new vector to be orthogonalized.
            prev_vectors: list of tensors [(S,), ...]
                The previous vectors to orthogonalize against. Assumes they
                are all orthogonal to one another.
            prev_is_mtx_sqr: bool
                if true, assumes the prev_vectors argument is equal
                to M^TM where M is a matrix of prev_vectors rows. This
                saves computation.
            norm: bool
                If True, the new vector is normalized after orthogonalization.
        Returns:
            new_vector: tensor (S,)
                The orthogonalized vector.
        """
        return orthogonalize_vector(
            new_vector,
            prev_vectors=prev_vectors,
            prev_is_mtx_sqr=prev_is_mtx_sqr,
            norm=norm,
            double_precision=self.orth_with_doubles,
        )

    def update_parameters_no_grad(self):
        """
        Orthogonalize all parameters in the list and make a new
        parameter list. Does not track gradients and detaches from
        previous Parameter objects.
        """
        device = self.get_device()
        orth = self.excl_ortho_list
        if len(orth)>0: orth = [o.to(device) for o in orth]
        params = []
        with torch.no_grad():
            for p in self.parameters_list:
                p = self.orthogonalize_vector(
                    p, prev_vectors=orth+params)
                params.append(p)
        self.parameters_list = nn.ParameterList([
            nn.Parameter(p.data) for p in params
        ])
        self.train_list = [p for p in self.parameters_list if p.requires_grad]
        self.frozen_list = [p for p in self.parameters_list if not p.requires_grad]
        self.update_orthogonalization_mtx()

    def orthogonalize_parameters_with_grad(self):
        """
        Only orthogonalize the parameters that require gradients.
        Does track gradients. Assumes frozen_list and
        orthogonalization_mtx are already orthogonalized.
        """
        device = self.get_device()
        if self.orthogonalization_mtx is None:
            self.update_orthogonalization_mtx()
        # cached vectors and excluded vectors are both included
        # in the orthogonalization matrix
        if len(self.orthogonalization_mtx)>0:
            orth = self.orthogonalization_mtx.to(device)
            mtx_sqr = self.ortho_mtx_sqr.to(device)
        else:
            orth = []
            mtx_sqr = []
        params = []
        for i,p in enumerate(self.parameters_list):
            if p.requires_grad==True:
                if len(params)==0 and len(mtx_sqr)>0:
                    p = self.orthogonalize_vector(
                        p,
                        prev_vectors=mtx_sqr,
                        prev_is_mtx_sqr=True,
                    )
                elif len(orth)>0:
                    p = self.orthogonalize_vector(
                        p, prev_vectors=[orth] + params
                    )
                else:
                    p = self.orthogonalize_vector(
                        p, prev_vectors=params
                    )
                params.append(p)
        return self.frozen_list + params

    def make_fca_matrix(self, components=None):
        """
        Create the low rank, FCA matrix from the stored parameters. Does
        not include the orthogonalization matrix, but is orthogonal to
        the orthogonalization matrix.

        Args:
            components: None or torch LongTensor.
                If None, all components are used. If a
                LongTensor, only the components specified
                by the tensor are used.
        Returns:
            matrix: torch.Tensor (R, D)
                The low rank orthogonal matrix created from
                the parameter list. Does not include the orthogonalization
                matrix in the resulting matrix. Each row is an orthogonal
                component, whereas the number of columns is equal to the
                dimensionality of the representations. 
        """
        if self.is_cached and self.cached_weight is not None:
            matrix = self.cached_weight.to(self.get_device())
        else:
            params = self.orthogonalize_parameters_with_grad()
            matrix = torch.vstack(params)
        if components is not None:
            matrix = matrix[components]
        elif self.component_mask is not None:
            matrix = matrix[self.component_mask]
        return matrix

    @property
    def weight(self):
        """
        Will construct the matrix by either using the cached_weight
        or will use gram-schmidt to orthogonalize the functional
        components against each other and the orthogonalization matrix.

        Returns:
            matrix: torch.Tensor (R, D)
                The low rank orthogonal matrix created from
                the parameter list. Does not include the orthogonalization
                matrix in the resulting matrix. Each row is an orthogonal
                component, whereas the number of columns is equal to the
                dimensionality of the representations. 
        """
        return self.make_fca_matrix()

    @property
    def rank(self):
        """
        Returns the number of functional components in this object

        Returns:
            rank: int
                the number of functional components in the object
        """
        return len(self.parameters_list)

    @property
    def n_components(self):
        """
        Returns the number of functional components in this object

        Returns:
            rank: int
                the number of functional components in the object
        """
        return self.rank

    def get_device(self):
        try:
            device = self.parameters_list[0].get_device()
        except: device = -1
        return "cpu" if device<0 else device

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
            if "means" in sd and not hasattr(self, "means"):
                self.add_means(sd["means"])
            if "stds" in sd and not hasattr(self, "stds"):
                self.add_means(sd["stds"])
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
                assert False

    def add_params_from_vector_list(self, vec_list, overwrite=True):
        """
        Adds each of the vectors in the vec_list to the parameters without
        orthogonalizing them. Will overwrite the values of the existing
        parameters' data if overwrite is true. Will not delete the
        extra remaining vectors.
        
        Args:
            vec_list: list of torch tensors [(S,), ...]
            overwrite: bool
                if true, will overwrite existing parameters before adding new
                ones. Otherwise only initializes new vectors.
        """
        device = self.get_device()
        for i,vec in enumerate(vec_list):
            if overwrite and i<len(self.parameters_list):
                p = self.parameters_list[i]
            else:
                p = self.add_new_axis_parameter()
            p.data = vec.data.clone().to(device)
        self.update_orthogonalization_mtx()

    def get_forward_hook(self, comms_dict=None):
        """
        Returns a forward hook function to perform FCA on a desired
        module's outputs. Possible to use the `use_complement_in_hook`
        field to subtract the projected functional components from the
        representation as opposed to using only the projected functional
        components.

        Args:
            comms_dict: dict or None
                dict to collect activations before applying fca.
        Returns:
            hook: python function
        """
        def hook(module, input, output):
            if type(output)!=torch.Tensor:
                reps = output["hidden_states"]
            else:
                reps = output
            if comms_dict is not None:
                comms_dict[self] = reps

            stripped = self.projinv(reps)
            if self.use_complement_in_hook:
                stripped = reps - stripped
            if type(output)!=torch.Tensor:
                output["hidden_states"] = stripped
            else:
                output = stripped
            return output
        return hook

    def get_image_forward_hook(self, comms_dict=None):
        """
        Returns a forward hook function to perform FCA on a desired
        module's outputs. Possible to use the `use_complement_in_hook`
        field to subtract the projected functional components from the
        representation as opposed to using only the projected functional
        components.

        Args:
            comms_dict: dict or None
                dict to collect activations before applying fca.
        Returns:
            hook: python function
        """
        def hook(module, input, output):
            if type(output)!=torch.Tensor:
                reps = output["hidden_states"]
            else:
                reps = output
            if comms_dict is not None:
                comms_dict[self] = reps

            og_shape = reps.shape
            reps = fca_image_prep(reps)

            stripped = self.projinv(reps)
            if self.use_complement_in_hook:
                stripped = reps - stripped

            stripped = fca_image_prep(stripped, og_shape=og_shape, inverse=True)

            if type(output)!=torch.Tensor:
                output["hidden_states"] = stripped
            else:
                output = stripped
            return output
        return hook
    
    def hook_model_layer(self, model, layer, comms_dict=None, rep_type="language"):
        """
        Helper function to register the forward hook produced from
        `get_forward_hook` to the argued model's argued layer.

        Args:
            model: torch Module
            layer: str
                the layer to register the forward hook.
            comms_dict: dict
                dict to collect activations before applying fca
            rep_type: str
                the representation type. Valid options are
                - "images": assumes latent shape of (B,C,H,W)
                - "language": assumes latent shape of (B,S,D)
                - "mlp": assumes latent shape of (B,D)
        """
        for mlayer,modu in model.named_modules():
            if layer==mlayer:
                if rep_type in {"image", "images"}:
                    return modu.register_forward_hook(
                        self.get_image_forward_hook(comms_dict=comms_dict)
                    )
                else:
                    return modu.register_forward_hook(
                        self.get_forward_hook(comms_dict=comms_dict)
                    )
        return None

    def forward(self, x, inverse=False, components=None):
        """
        Args:
            x: torch Tensor (...,S)
            inverse: bool
                if true, will use the transpose of the FCA matrix to
                project back into the original neural space. To be clear,
                we actually store the FCA matrix as a row matrix, but
                in theory it is a column matrix. So, we actually end up
                using the transpose of the matrix in the forward pass
                and using the untransposed version in the inverse.
            components: torch Long Tensor
                optionally argue a tensor indicating the components to
                use based on their index in the fca matrix.
        """
        if inverse:
            x = torch.matmul(
                x, self.make_fca_matrix(components=components)
            )
            if self.stds is not None:
                x = x*self.stds
            if self.means is not None:
                x = x+self.means
            return x
        if self.means is not None:
            x = x-self.means
        if self.stds is not None:
            x = x/self.stds
        return torch.matmul(
            x, self.make_fca_matrix(components=components).T
        )
    
    def projinv(self, x, components=None):
        """
        Projects the input x into the functional component space
        and returns the inverse projection back into the original
        neural space.

        Args:
            x: torch Tensor (...,S)
            components: torch Long Tensor
                optionally argue a tensor indicating the components to
                use based on their index in the fca matrix.
        """
        return self(
            self(x, components=components),
            inverse=True,
            components=components
        )
    
    def interchange_intervention(self, trg, src):
        """
        Performs a Distributed Alignment Search interchange intervention
        using the fca matrix as defined in https://arxiv.org/abs/2501.06164.
        This cannot be used with Model Alignment Search!!
        """
        return trg-self.projinv(trg)+self.projinv(src)

    def find_sufficient_components(
        self,
        model,
        layer,
        data_loader,
        val_data=None,
        acc_threshold=0.99,
        lr=0.001,
        n_epochs=None,
        verbose=True,
    ):
        """
        This method trains the functionally sufficient components to
        acheive the accuracy threshold on the outputs.
        You can optionally include "labels" in the data as a way
        to specify the objective outputs, otherwise will produce output
        labels from the model's outputs on the input data. 

        Args:
            model: torch module
            layer: str
                the name of the model module to perform FCA on
            data_loader: data iterable
                This iterable should return batches of data that can be
                fed to the model's forward function as kwargs using the
                double star notation (**kwargs).
            val_data: (optional) Dataset
                optionally argue validation data for tracking loss plateaus
            lr: float
                learning rate
            n_epochs: (optional) int or None
                the number of training epochs. Can also ignore this
                argument and train till convergence.
        """
        plateau_tracker = PlateauTracker()
        self.set_cached(True) # Will make fewer calls to gram-schmidt
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        handle = self.hook_model_layer(model=model, layer=layer)
        if handle is None:
            layer = "model." + layer
            handle = self.hook_model_layer(model=model, layer=layer)
        assert handle is not None, "Failed to find layer "+str(layer)
        loss = math.inf
        acc = 0
        metrics = {
            "train_loss": [],
            "train_acc": [],
            "valid_loss": [],
            "valid_acc": [],
        }
        epoch = 0
        if n_epochs is None: n_epochs = math.inf

        # Begin Training Process
        while epoch<n_epochs:
            epoch += 1
            if verbose:
                print("\nBeginning Epoch", epoch)
            
            # Train
            losses, accs = [],[]
            for step, batch in enumerate(data_loader):
                outputs = model(**batch)
                loss, acc = outputs["loss"], outputs["acc"]
                loss.backward()
                optimizer.step()
                self.reset_cached_weight()
                losses.append(loss.item())
                accs.append(acc.item())
            metrics["train_loss"].append(np.mean(losses))
            metrics["train_acc"].append(np.mean(accs))
            loss,acc = metrics["train_loss"][-1],metrics["train_acc"][-1]

            if verbose:
                print("Train Loss:", loss.item(), "-- Acc:", acc.item())

            # Validation
            if val_data is not None:
                self.use_complement_in_hook = False
                losses, accs = [],[]
                for step, batch in enumerate(val_data):
                    with torch.no_grad():
                        outputs = model(**batch)
                    loss, acc = outputs["loss"], outputs["acc"]
                    losses.append(loss.item())
                    accs.append(accs.item())
                metrics["valid_loss"].append(np.mean(losses))
                metrics["valid_acc"].append(np.mean(accs))
                loss,acc = metrics["valid_loss"][-1],metrics["valid_acc"][-1]
                if verbose:
                    print("Valid Loss:", loss.item(), "-- Acc:", acc.item())

            # If we achieve our accuracy threshold, we stop the process
            if acc>=acc_threshold:
                print("Reached accuracy threshold, stopping process")
                break

            # Add a new component when performance plateaus
            if plateau_tracker.update(loss=loss, acc=acc):
                new_component = self.add_component()
                if verbose: print("Adding new component! New Rank", self.rank)
                # If at max_rank, the new_component will not be added
                if new_component is None:
                    print("Reached max components, stopping process")
                    break

                # Need to add new component to optimizer
                optimizer = torch.optim.Adam(self.parameters(), lr=lr)
            
        if epoch==n_epochs and verbose:
            print("Stopping due to epoch limit")
        
        # Remove the hook
        handle.remove()
        return metrics


class UnnormedFCA(FunctionalComponentAnalysis):
    """
    A version of FCA that does not normalize the vectors.
    This is useful for debugging and testing.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def orthogonalize_vector(self,
                            new_vector,
                            prev_vectors,
                            prev_is_mtx_sqr=False,
                            norm=False):
        """
        Orthogonalize a new vector to a list of previous vectors.
        Tracks gradients.
        Args:
            new_vector: tensor (S,)
                The new vector to be orthogonalized.
            prev_vectors: list of tensors [(S,), ...]
                The previous vectors to orthogonalize against. Assumes they
                are all orthogonal to one another.
            prev_is_mtx_sqr: bool
                If True, the previous vectors are assumed to be a covariance
                matrix. Otherwise, they are assumed to be a list of vectors.
            norm: bool
                If True, the new vector is normalized after orthogonalization.
        Returns:
            new_vector: tensor (S,)
                The orthogonalized vector.
        """
        return orthogonalize_vector(
            new_vector,
            prev_vectors=prev_vectors,
            prev_is_mtx_sqr=prev_is_mtx_sqr,
            norm=norm,
            double_precision=self.orth_with_doubles,
        )

    def orthogonalize_parameters_with_grad(self):
        """
        Only orthogonalize the parameters that require gradients.
        Does track gradients. Assumes frozen_list and
        orthogonalization_mtx are already orthogonalized.
        """
        device = self.get_device()
        if self.orthogonalization_mtx is None:
            self.update_orthogonalization_mtx()
        # cached vectors and excluded vectors are both included
        # in the orthogonalization matrix
        if len(self.orthogonalization_mtx)>0:
            orth = self.orthogonalization_mtx.to(device)
            mtx_sqr = self.ortho_mtx_sqr.to(device)
        else:
            orth = []
            mtx_sqr = []
        params = []
        for i,p in enumerate(self.parameters_list):
            if p.requires_grad==True:
                if len(params)==0 and len(mtx_sqr)>0:
                    p = self.orthogonalize_vector(
                        p,
                        prev_vectors=mtx_sqr,
                        prev_is_mtx_sqr=True
                    )
                elif len(orth)>0:
                    p = self.orthogonalize_vector(
                        p, prev_vectors=[orth] + params
                    )
                else:
                    p = self.orthogonalize_vector(
                        p, prev_vectors=params
                    )
                params.append(p)
        return self.frozen_list + params

class PCAFunctionalComponentAnalysis(FunctionalComponentAnalysis):
    """
    A version of FCA that uses PCA as the functional components.
    """
    def __init__(self, X, scale=True, center=True, *args, **kwargs):
        """
        Args:
            X: torch Tensor (N, S)
                The input data to perform PCA on. This is used to
                initialize the functional components.
            scale: bool
                If True, scales the data before performing PCA.
            center: bool
                If True, centers the data before performing PCA.
            *args, **kwargs: additional arguments
                These are passed to the FunctionalComponentAnalysis
                constructor.
        """
        self.pca_info = self.perform_pca(
            X=X, scale=scale, center=center, **kwargs, )
        if "size" not in kwargs: kwargs["size"] = X.shape[-1]
        super().__init__(
            **kwargs,
            means=self.pca_info.get("means", None),
            stds=self.pca_info.get("stds", None),
        )
        self.set_cached(True)
        self.update_parameters_with_pca(*args, **kwargs)

    def proportion_expl_var(self, rank=None, actvs=None):
        """
        Returns the proportion of explained variance by the PCA components.
        """
        if rank is None: rank = self.max_rank
        if actvs is None:
            return self.pca_info["proportion_expl_var"][:rank].sum()
        projinvs = self.projinv(actvs)
        return explained_variance( preds=projinvs, labels=actvs, )

    def update_parameters_with_pca(self, *args, **kwargs):
        """ Updates the parameters of the FCA with PCA components. """
        if self.rank > 1: self.remove_all_components()
        vecs = self.pca_info["components"][:self.max_rank]
        self.add_params_from_vector_list( vecs, overwrite=True )
        self.cached_weight = torch.vstack(
            [p for p in self.parameters_list]).to(self.get_device())

    def set_max_rank(self, max_rank):
        """
        Updates the max rank of the PCAFunctionalComponentAnalysis.
        This will reinitialize the components and orthogonalization vectors.
        """
        self.max_rank = max_rank
        self.update_parameters_with_pca()

    def perform_pca(self,
        X,
        scale=True,
        center=True,
        *args, **kwargs,
    ):
        """
        Performs PCA on the input data and returns the principal components.

        Args:
            X: torch Tensor (N, S)
                The input data to perform PCA on.
            scale: bool
                If True, scales the data before performing PCA.
            center: bool
                If True, centers the data before performing PCA.
        Returns:
            ret_dict: list of tensors [(S,), ...]
                The principal components of the input data.
        """
        return perform_pca(
            X=X,
            n_components=None,
            scale=scale,
            center=center,
            transform_data=False,
            use_eigen=True,
        )

def load_fca_from_path(save_dir):
    """
    Loads a single FCA from a file path.

    Args:
        save_dir: str
            The path to the FCA checkpoint save directory.
    Returns:
        fca: FunctionalComponentAnalysis
            The loaded FCA object.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"File not found: {save_dir}")
    if not os.path.isdir(save_dir):
        save_dir = "/".join(save_dir.split("/")[:-1])  # Remove the last part if it's a file
    
    # Load the FCA checkpoint
    checkpt_path = os.path.join(save_dir, "fca_best.pt")
    fca_state_dict = torch.load(checkpt_path, map_location="cpu")
    config_path = os.path.join(save_dir, "fca_config.yaml")
    fca_config = load_json_or_yaml(config_path)
    
    # Initialize the FCA object
    kwargs = fca_config.get("fca_params", {})
    fca = FunctionalComponentAnalysis(**kwargs)
    fca.load_sd(fca_state_dict)
    fca.update_parameters_no_grad()
    fca.freeze_parameters()
    fca.set_cached(True)
    
    return fca

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
        fcas[layer].update_parameters_no_grad()
        fcas[layer].freeze_parameters()
        fcas[layer].set_cached(True)
    return fcas

def load_fcas(
        model,
        load_path,
        use_complement_in_hook=True,
        ret_paths=False,
        verbose=False):
    """
    Simplifies the recursive loading of previous, chained fcas.

    Args:
        use_complement_in_hook: bool
            generally want this to be true when chaining fca objects
            during an fca training. We want to train the new fca on the
            complement subspace of the previous fcas.
    """
    device = "cpu" if model is None else model.get_device()

    # Load Checkpoint
    fca_checkpoint = torch.load(load_path)
    fca_config = fca_checkpoint["config"]

    # Initialize Variables
    fcas = {}
    handles = {}
    loaded_fcas = []
    loaded_handles = []
    loaded_paths = []

    # Recursively Load Previous FCAs
    if fca_config.get("fca_load_path", None) is not None:
        ret = load_fcas(
            model=model,
            load_path=fca_config["fca_load_path"],
            use_complement_in_hook=use_complement_in_hook,
            ret_paths=ret_paths,
            verbose=verbose,
        )
        if ret_paths:
            loaded_fcas, loaded_handles, loaded_paths = ret
        else:
            loaded_fcas, loaded_handles = ret
    loaded_paths.append(load_path)

    # Create the FCAs and Load the SDs
    if verbose:
        print("Loading:", load_path)
    state_dicts = fca_checkpoint["fca_state_dicts"]
    kwargs = fca_config.get("fca_params", {})
    modules = {}
    # Attach FCA if model is argued
    if model is not None:
        for layer,modu in model.named_modules():
            modules[layer] = modu
    # Create FCA for each layer
    for layer in state_dicts:
        sd = state_dicts[layer]
        kwargs["size"] = sd[list(sd.keys())[0]].shape[0]
        kwargs["use_complement_in_hook"] = use_complement_in_hook
        fcas[layer] = FunctionalComponentAnalysis( **kwargs )
        fcas[layer].load_sd(sd)
        fcas[layer].freeze_parameters()
        fcas[layer].set_cached(True)
        fcas[layer].to(device)
        if model is not None and layer in modules:
            h = modules[layer].register_forward_hook(
                fcas[layer].get_forward_hook()
            )
            handles[layer] = h

    loaded_handles.append(handles)
    loaded_fcas.append(fcas)
    if ret_paths:
        return loaded_fcas, loaded_handles, loaded_paths
    return loaded_fcas, loaded_handles

def initialize_fcas(
        model,
        config,
        loaded_fcas=[],
        means=None,
        stds=None):
    """
    Args:
        model: torch module
        config: dict
        loaded_fcas: list of FCA objects
        means: dict
            keys: str
                the layer names
            vals: torch tensor (S,)
                the means for that layer
        stds: dict
            keys: str
                the layer names
            vals: torch tensor (S,)
                the stds for that layer
    """
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
            kwargs["means"] = means[name]
            kwargs["stds"] = stds[name]
            fcas[name] = FunctionalComponentAnalysis(
                **kwargs
            )
            fcas[name].to(device)
            if config.get("ensure_ortho_chain", False):
                if loaded_fcas:
                    for loaded in loaded_fcas:
                        if name in loaded:
                            print("Loading Orthogonalization Vectors", name)
                            fcas[name].add_excl_ortho_vectors(
                                loaded[name].parameters_list)
            h = modu.register_forward_hook(
                fcas[name].get_forward_hook()
            )
            handles[name] = h
            fca_parameters += list(fcas[name].parameters())
    return fcas, handles, fca_parameters
    
def load_ortho_fcas(fca, fca_save_list):
    """
    Loads the orthogonalization vectors from a list of FCA objects.
    This is useful for loading the orthogonalization vectors from a
    previous or multiple FCA objects.

    Args:
        fca: FunctionalComponentAnalysis object
            The FCA object to load the orthogonalization vectors into.
        fca_save_list: list of FunctionalComponentAnalysis objects or save paths
            The list of FCA objects to load the orthogonalization vectors from.
    """
    for prev_fca in fca_save_list:
        if type(prev_fca) is str:
            prev_fca = load_fca_from_path(prev_fca)
        fca.add_excl_ortho_vectors(prev_fca.parameters_list)

__all__ = [
    "FunctionalComponentAnalysis", "gram_schmidt",
    "orthogonalize_vector", "load_ortho_fcas",
    "load_fca_from_path", "load_fcas_from_path",
]

# Example usage
if __name__ == "__main__":

    n_dim = 1024
    orth_with_doubles = True
    fca = FunctionalComponentAnalysis(
        size=n_dim,
        init_rank=n_dim-1,
        orth_with_doubles=orth_with_doubles,
    )
    if torch.cuda.is_available():
        fca.cuda()
    # Demontrates how to add components after initialization. Now at full rank
    fca.add_component()
    print("NComponents:", fca.rank)

    import time

    prev_list = fca.orthogonalize_parameters_with_grad()
    prev_list = [p.data for p in prev_list[:-1]]

    print("Base:")
    start = time.time()
    for _ in range(1000):
        base_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=prev_list)
        base_params.append(p)
    print("Time:", time.time()-start)

    mtx = torch.vstack([v.data for v in prev_list])
    print("Fast:")
    start = time.time()
    for _ in range(1000):
        fast_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=mtx)
        fast_params.append(p)
    print("Time:", time.time()-start)

    cov = torch.matmul(mtx.T, mtx)
    print("Cov:")
    start = time.time()
    for _ in range(1000):
        cov_params = []
        p = fca.parameters_list[-1]
        p = fca.orthogonalize_vector(p, prev_vectors=cov, prev_is_mtx_sqr=True)
        cov_params.append(p)
    print("Time:", time.time()-start)

    for bp,fp,cv in zip(base_params, fast_params, cov_params):
        mse = ((bp-fp)**2).mean()
        if mse>1e-6:
            print("BP:", bp[:5])
            print("FP:", fp[:5])
            print("MSE:", mse)
            print()
        mse = ((bp-cv)**2).mean()
        if mse>1e-6:
            print("BP:", bp[:5])
            print("Cv:", cv[:5])
            print("MSE:", mse)
            print()
        break

    print("End Speed Comparison")


    # Determine the extent to which the vectors are orthogonal
    plist = fca.parameters_list
    lin = torch.nn.Linear(n_dim,n_dim)
    plist = torch.nn.utils.parametrizations.orthogonal(lin).weight
    errors = []
    fca.update_parameters_no_grad()
    for i in range(len(plist)):
        for j in range(i):
            errors.append(
                torch.dot(plist[i], plist[j]).item()
            )
    errors = np.asarray(errors)
    print("Rank:", fca.rank)
    print("FCA Orthogonalization Error:")
    print("\tMean:", np.mean(errors))
    print("\tAbsMean:", np.mean(np.abs(errors)))
    print("\tMax:", np.max(errors))
    print("\tMin:", np.min(errors))
    print()

    # lin = torch.nn.Linear(n_dim,n_dim)
    # plist = torch.nn.utils.parametrizations.orthogonal(lin).weight
    # errors = []
    # fca.update_parameters_no_grad()
    # for i in range(len(plist)):
    #     for j in range(i):
    #         errors.append(
    #             torch.dot(plist[i], plist[j]).item()
    #         )
    # errors = np.asarray(errors)
    # print("Exponential Orthogonalization Error:")
    # print("\tMean:", np.mean(errors))
    # print("\tAbsMean:", np.mean(np.abs(errors)))
    # print("\tMax:", np.max(errors))
    # print("\tMin:", np.min(errors))
    # print()

    vec = torch.randn(1,n_dim)
    if torch.cuda.is_available(): vec = vec.cuda()
    mtx = fca.weight[:3]
    rot = torch.matmul(vec, mtx.T)
    new_vec = torch.matmul(rot, mtx)
    diff = vec - new_vec
    zero = torch.matmul(diff, mtx.T)
    assert zero.mean()<1e-5

    mtx = fca.weight[3:]
    rot = torch.matmul(vec, mtx.T)
    nnew_vec = torch.matmul(rot, mtx)
    ddiff = vec - nnew_vec
    zzero = torch.matmul(ddiff, mtx.T)

    assert ((vec-new_vec-nnew_vec)**2).sum() < 1e-5
    
    rot = fca(vec)
    new_vec = fca(rot, inverse=True)
    assert ((vec-new_vec)**2).sum()<1e-7
