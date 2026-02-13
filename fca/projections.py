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
    Caculates the explained variance of the reps on the target reps.
    
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

def get_cor_mtx(X, Y, batch_size=500, to_numpy=False, zscore=True, device=None):
    """
    Creates a correlation matrix for X and Y using the GPU

    X: torch tensor or ndarray (T, C) or (T, C, H, W)
    Y: torch tensor or ndarray (T, K) or (T, K, H1, W1)
    batch_size: int
        batches the calculation if this is not None
    to_numpy: bool
        if true, returns matrix as ndarray
    zscore: bool
        if true, both X and Y are normalized over the T dimension
    device: int
        optionally argue a device to use for the matrix multiplications

    Returns:
        cor_mtx: (C,K) or (C*H*W, K*H1*W1)
            the correlation matrix
    """
    if len(X.shape) < 2:
        X = X[:,None]
    if len(Y.shape) < 2:
        Y = Y[:,None]
    if len(X.shape) > 2:
        X = X.reshape(len(X), -1)
    if len(Y.shape) > 2:
        Y = Y.reshape(len(Y), -1)
    if type(X) == type(np.array([])):
        to_numpy = True
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    if device is None:
        device = X.get_device()
        if device<0: device = "cpu"
    if zscore:
        xmean = X.mean(0)
        xstd = torch.sqrt(((X-xmean)**2).mean(0))
        ymean = Y.mean(0)
        ystd = torch.sqrt(((Y-ymean)**2).mean(0))
        xstd[xstd<=0] = 1
        X = (X-xmean)/(xstd+1e-5)
        ystd[ystd<=0] = 1
        Y = (Y-ymean)/(ystd+1e-5)
    X = X.permute(1,0)

    with torch.no_grad():
        if batch_size is None:
            X = X.to(device)
            Y = Y.to(device)
            cor_mtx = torch.einsum("it,tj->ij", X, Y).detach().cpu()
        else:
            cor_mtx = []
            for i in range(0,len(X),batch_size): # loop over x neurons
                sub_mtx = []
                x = X[i:i+batch_size].to(device)

                # Loop over y neurons
                for j in range(0,Y.shape[1], batch_size):
                    y = Y[:,j:j+batch_size].to(device)
                    cor_block = torch.einsum("it,tj->ij",x,y)
                    cor_block = cor_block.detach().cpu()
                    sub_mtx.append(cor_block)
                cor_mtx.append(torch.cat(sub_mtx,dim=1))
            cor_mtx = torch.cat(cor_mtx, dim=0)
    cor_mtx = cor_mtx/len(Y)
    if to_numpy:
        return cor_mtx.numpy()
    return cor_mtx

def perform_pca(
        X,
        n_components=None,
        scale=True,
        center=True,
        transform_data=False,
        full_matrices=False,
        randomized=False,
        use_eigen=True,
        batch_size=None,
        verbose=True,
):
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
        use_eigen: bool
            if true, will use an eigen decomposition on the
            covariance matrix of X to save compute
        batch_size: int or None
            optionally argue a batch size. only applies if use_eigen
            is true.
    Returns:
        ret_dict: dict
            A dictionary containing the following keys:
            - "components": tensor (N, n_components)
                The principal components (eigenvectors) of the data.
            - "explained_variance": tensor (n_components,)
                The explained variance for each principal component.
            - "proportion_expl_var": tensor (n_components,)
                The proportion of explained variance for each principal component.
            - "means": tensor (N,)
                The mean of each feature (column) in the data.
            - "stds": tensor (N,)
                The standard deviation of each feature (column) in the data.
            - "transformed_X": tensor (M, n_components)
                The data projected onto the principal components, if
                transform_data is True.
    """
    if use_eigen:
        return perform_eigen_pca(
            X=X,
            n_components=n_components,
            scale=scale,
            center=center,
            transform_data=transform_data,
            batch_size=batch_size,
            verbose=verbose,
        )
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
    if verbose: print("Performing SVD")
    U, S, Vh = svd(X, **svd_kwargs)
    
    # The principal components (eigenvectors) are the first n_components rows of Vh
    components = Vh[:n_components]
    
    # Explained variance for each component can be computed from the singular values
    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)
    proportion_expl_var = explained_variance/explained_variance.sum()
    
    ret_dict = {
        "components": components,
        "explained_variance": explained_variance,
        "proportion_expl_var": proportion_expl_var,
        "means": means,
        "stds": stds,
    }
    if transform_data:
        # Project the data onto the principal components
        # Note: components.T has shape (features, n_components)
        ret_dict["transformed_X"] = X @ components.T

    return ret_dict

def perform_eigen_pca(
        X,
        n_components=None,
        scale=True,
        center=True,
        transform_data=False,
        batch_size=None,
        device=None,
        verbose=True,
):
    """
    Perform PCA on the data matrix X by using an eigen decomp on
    the covariance matrix

    Args:
        X: tensor (M,N)
        n_components: int
            optionally specify the number of components
        scale: bool
            if true, will scale the data along each column
        transform_data: bool
            if true, will compute and return the transformed
            data
    Returns:
        ret_dict: dict
            A dictionary containing the following keys:
            - "components": tensor (N, n_components)
                The principal components (eigenvectors) of the data.
            - "explained_variance": tensor (n_components,)
                The explained variance for each principal component.
            - "proportion_expl_var": tensor (n_components,)
                The proportion of explained variance for each principal component.
            - "means": tensor (N,)
                The mean of each feature (column) in the data.
            - "stds": tensor (N,)
                The standard deviation of each feature (column) in the data.
            - "transformed_X": tensor (M, n_components)
                The data projected onto the principal components, if
                transform_data is True.
    """
    if n_components is None:
        n_components = X.shape[-1]
        
    if type(X)==torch.Tensor:
        eigen_fn = torch.linalg.eigh
    elif type(X)==np.ndarray:
        eigen_fn = np.linalg.eigh
    assert not n_components or X.shape[-1]>=n_components

    # Center the data by subtracting the mean along each feature (column)
    means = 0
    if center:
        means = X.mean(dim=0, keepdim=True)
        X = X - means
    stds = 1
    if scale:
        stds = (X.std(0)+1e-6)
        X = X/stds
    
    cov = get_cor_mtx(
        X,X,
        zscore=False,
        batch_size=batch_size,
        device=device,
        to_cpu=X.device.type == "cpu",
    )
    ## Use eigendecomposition of the covariance matrix for efficiency
    ## Cov = (1 / (M - 1)) * X^T X
    #cov = X.T @ X / (X.shape[0] - 1)  # shape (N, N)

    # Force symmetry and add regularization for numerical stability
    if type(X) == torch.Tensor:
        cov = (cov + cov.T) / 2
        cov = cov + torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype) * 1e-6
    else:
        cov = (cov + cov.T) / 2
        cov = cov + np.eye(cov.shape[0]) * 1e-6

    # Compute eigenvalues and eigenvectors
    # Try original dtype first, then float64 if that fails (float32 eigh can be unstable)
    if type(X) == torch.Tensor:
        orig_dtype = cov.dtype
        try:
            eigvals, eigvecs = eigen_fn(cov)
        except RuntimeError:
            # float32 eigh can fail on large matrices; use float64
            cov64 = cov.double()
            eigvals, eigvecs = eigen_fn(cov64)
            eigvals = eigvals.to(orig_dtype)
            eigvecs = eigvecs.to(orig_dtype)
        # Select top n_components in descending order
        eigvals = eigvals[-n_components:].flip(0)
        eigvecs = eigvecs[:, -n_components:].flip(1)
    else:
        eigvals, eigvecs = eigen_fn(cov)
        eigvals = eigvals[-n_components:][::-1]
        eigvecs = eigvecs[:, -n_components:][:, ::-1]

    explained_variance = eigvals
    proportion_expl_var = explained_variance / explained_variance.sum()
    components = eigvecs.T  # shape (n_components, N)

    ret_dict = {
        "components": components,
        "explained_variance": explained_variance,
        "proportion_expl_var": proportion_expl_var,
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
