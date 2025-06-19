import torch

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
            dimensions
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

__all__ = [
    "matrix_projinv", "projinv_expl_variance", "lost_variance", "explained_variance", "component_wise_expl_var",
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