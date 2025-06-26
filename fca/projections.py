import torch
from sklearn.utils.extmath import randomized_svd
import numpy as np

def matrix_projinv(x, W):
    """
    This function projects the activations into the weight space and then
    inverts the projection, returning the inverted vectors.
    
    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        torch tensor (B,D)
            the projected activations returned to their
            original space.
    """
    return torch.matmul(torch.matmul(x, W), torch.linalg.pinv(W))

def explained_variance(
    preds: torch.Tensor,
    labels:torch.Tensor,
    eps: float = 1e-8,
    mean_over_dims=False,
) -> torch.Tensor:
    """
    Caculates the explained variance of the reps on the
    target reps.
    
    Args:
        preds: torch tensor (B,D)
        labels: torch tensor (B,D)
        eps: float
            small constant to prevent division by zero
        mean_over_dims: bool
            if true, will return the mean explained variance over the
            feature dimensions
    Returns:
        expl_var: torch tensor (D,) or (1,)
            we get an explained variance value for each dimension, but
            we can average over this value if mean_over_dims is true.
    """
    assert preds.shape == labels.shape, "Shapes of preds and labels must match"
    
    diff = labels - preds
    var_diff = torch.var(diff, dim=0, unbiased=True)    # shape (D,)
    var_labels = torch.var(labels, dim=0, unbiased=True)# shape (D,)

    expl_var = 1 - var_diff / (var_labels + eps)         # shape (D,)
    if mean_over_dims:
        return expl_var.mean()
    return expl_var

def projinv_expl_variance(x,W):
    """
    Projects x into W and inverts the projection to create z. Then
    returns the explained variance of x using z.

    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        explained_variance: torch tensor (1,)
            the explained variance of the activations projected into W
            and then returned to their original space.
    """
    preds = matrix_projinv(x,W)
    return explained_variance(preds, x)

def lost_variance(x,W):
    """
    Returns the variance lost when x is right multiplied by W.

    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        lost_var: torch tensor (1,)
            1 minus the explained variance of x projected into W
            and then returned to their original space.
    """
    return 1-projinv_expl_variance(x,W)

def component_wise_expl_var(actvs, weight, eps=1e-6):
    """
    For each component in U, will determine its projected
    explained variance.
    
    Args:
        actvs: torch tensor (B,D)
        weight: torch tensor (D,P)
        eps: float
            a value for which components will be removed
    Returns:
        expl_vars: tensor (P,)
            the explained variance for each component
        cumu_expl_vars: tensor (P,)
            the explained variance for the cumulation of the components
    """
    U, S, Vt = torch.linalg.svd(weight)
    n_components = (S>=eps).long().sum()
    expl_vars = []
    cumu_sum = 0
    cumu_expl_vars = []
    for comp in range(n_components):
        W = S[comp]*torch.matmul(U[:,comp:comp+1], Vt[comp:comp+1, :])
        preds = matrix_projinv(actvs, W=W)
        expl_var = explained_variance(preds, actvs)
        expl_vars.append(expl_var)

        cumu_sum += preds
        expl_var =  explained_variance(cumu_sum, actvs)
        cumu_expl_vars.append(expl_var)
    for _ in range(max(*weight.shape)-n_components):
        expl_vars.append(torch.zeros_like(expl_vars[-1]))
        cumu_expl_vars.append(torch.zeros_like(cumu_expl_vars[-1]))
    return torch.stack(expl_vars), torch.stack(cumu_expl_vars)

def perform_pca(
        X,
        n_components=None,
        scale=True,
        center=True,
        transform_data=False,
        full_matrices=False,
        randomized=False):
    """
    Perform PCA on the data matrix X

    Args:
        X: tensor (M,N)
        n_components: int
            optionally specify the number of components
        scale: bool
            if true, will scale the data along each column
        transform_data: bool
            if true, will compute and return the transformed
            data
        full_matrices: bool
            determines if U will be returned as a square.
        randomized: bool
            if true, will use randomized svd for faster
            computations
    Returns:
        ret_dict: dict
            A dictionary containing the following keys:
            - "components": tensor (N, n_components)
                The principal components (eigenvectors) of the data.
            - "explained_variance": tensor (n_components,)
                The explained variance for each principal component.
            - "prop_explained_variance": tensor (n_components,)
                The proportion of explained variance for each principal component.
            - "means": tensor (N,)
                The mean of each feature (column) in the data.
            - "stds": tensor (N,)
                The standard deviation of each feature (column) in the data.
            - "transformed_X": tensor (M, n_components)
                The data projected onto the principal components, if transform_data is True.
    """
    if n_components is None:
        n_components = X.shape[-1]
        
    svd_kwargs = {}
    if type(X)==torch.Tensor:
        if randomized:
            svd_kwargs["q"] = n_components
            svd = torch.svd_lowrank
        else:
            svd_kwargs["full_matrices"] = full_matrices
            svd = torch.linalg.svd
    elif type(X)==np.ndarray:
        if randomized:
            svd_kwargs["n_components"] = n_components
            svd = randomized_svd
        else:
            svd_kwargs["n_components"] = n_components
            svd_kwargs["compute_uv"] = True
            svd = np.linalg.svd
    assert not n_components or X.shape[-1]>=n_components
    # Center the data by subtracting the mean along each feature (column)
    means = torch.zeros_like(X[0])
    if center:
        means = X.mean(dim=0, keepdim=True)
        X = X - means
    stds = torch.ones_like(X[0])
    if scale:
        stds = (X.std(0)+1e-6)
        X = X/stds
    
    
    # Compute the SVD of the centered data
    # X = U @ diag(S) @ Vh, where Vh contains the principal components as its rows
    U, S, Vh = svd(X, **svd_kwargs)
    
    # The principal components (eigenvectors) are the first n_components rows of Vh
    components = Vh[:n_components]
    
    # Explained variance for each component can be computed from the singular values
    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)
    prop_explained_variance = explained_variance/explained_variance.sum()
    
    ret_dict = {
        "components": components,
        "explained_variance": explained_variance,
        "prop_explained_variance": prop_explained_variance,
        "means": means,
        "stds": stds,
    }
    if transform_data:
        # Project the data onto the principal components
        # Note: components.T has shape (features, n_components)
        ret_dict["transformed_X"] = X @ components.T

    return ret_dict

__all__ = [
    "matrix_projinv", "projinv_expl_variance", "lost_variance", "explained_variance",
    "component_wise_expl_var", "perform_pca",
]

if __name__=="__main__":
    n_dims = 4
    U,_,Vt = torch.linalg.svd(torch.randn(n_dims,n_dims), full_matrices=True)
    indys, cumu = component_wise_expl_var(U,U)
    print("Indy Components:", indys.mean(-1))
    print("\tSum:", indys.mean(-1).sum(0))
    print("Cumu Components:", cumu.mean(-1))
    preds = matrix_projinv(U,U)
    assert cumu.mean(-1)[-1]==explained_variance(preds, U).mean()
    assert explained_variance(preds, U).mean()==projinv_expl_variance(U,U).mean()
    assert lost_variance(U, U).mean()==(1-projinv_expl_variance(U,U).mean())