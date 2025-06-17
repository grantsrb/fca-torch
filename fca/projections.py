import torch

def matrix_proj(x, W):
    """
    This function projects the activations into the
    weight space and then inverts the projection to
    return the inverted vectors.
    
    Args:
        x: torch tensor (B,D)
        W: torch tensor (D,P)
    Returns:
        torch tensor (B,D)
            the projected activations returned to their
            original space.
    """
    return torch.matmul(torch.matmul(x, W), torch.linalg.pinv(W))

def project_out(
        actvs,
        weight=None,
        n_components=None,
        U=None, S=None, Vt=None,
        eps=1e-6,
):
    """
    This function projects the activations into the
    weight space and then inverts the projection to
    return both the projected and inverted vectors.
    
    Args:
        actvs: torch tensor (B,D)
        weight: torch tensor (D,P)
        n_components: (optional) int
            optionally specify the number of components
            to project into.
        U: torch tensor (D,P)
            for computational efficiency
        S: torch tensor (D,P)
            for computational efficiency
        Vt: torch tensor (D,P)
            for computational efficiency
        eps: float
            a value for which components will be removed
    Returns:
        proj_actvs: torch tensor (B,s)
            the activations projected into the s components
            that explain 100% of the weight matrix.
        ret_actvs: torch tensor (B,D)
            the projected activations returned to their
            original space.
    """
    if U is None:
        U, S, Vt = torch.linalg.svd(weight)
    if n_components is None:
        if S is not None:
            n_components = (S>=eps).long().sum()
        else:
            n_components = U.shape[-1]
    proj_actvs = torch.matmul(actvs,U[:,:n_components])
    ret_actvs = torch.matmul(proj_actvs,U[:,:n_components].T)
    return proj_actvs, ret_actvs

def explained_variance(
    preds: torch.Tensor,
    labels:
    torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Caculates the explained variance of the reps on the
    target reps.
    
    Args:
        preds: torch tensor (B,D)
        labels: torch tensor (B,D)
        eps: float
            small constant to prevent division by zero
    Returns:
        expl_var: torch tensor (D,)
    """
    assert preds.shape == labels.shape, "Shapes of preds and labels must match"
    
    diff = labels - preds
    var_diff = torch.var(diff, dim=0, unbiased=False)    # shape (D,)
    var_labels = torch.var(labels, dim=0, unbiased=False)# shape (D,)

    expl_var = 1 - var_diff / (var_labels + eps)               # shape (D,)
    return expl_var

def component_expl_var(actvs, weight, eps=1e-6):
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
    """
    U, S, _ = torch.linalg.svd(weight)
    n_components = (S>=eps).long().sum()
    expl_vars = []
    cumu_sum = 0
    cumu_expl_vars = []
    for comp in range(n_components):
        _, preds = project_out(actvs, U=U[:,comp:comp+1], eps=eps)
        cumu_sum += preds
        expl_var = explained_variance(preds, actvs)
        expl_vars.append(expl_var)
        expl_var =  explained_variance(cumu_sum, actvs)
        cumu_expl_vars.append(expl_var)
    for _ in range(max(*weight.shape)-n_components):
        expl_vars.append(torch.zeros_like(expl_vars[-1]))
        cumu_expl_vars.append(torch.zeros_like(cumu_expl_vars[-1]))
    return torch.stack(expl_vars), torch.stack(cumu_expl_vars)

def projected_expl_var(
    actvs,
    weight=None,
    U=None,
    n_components=None
):
    """
    This function finds the explained variance of a projection
    into the left svd vectors from the argued weight.
    
    Args:
        actvs: torch tensor (B,D)
        weight: torch tensor (D,P)
        U: torch tensor (D,P)
            for computational efficiency
        n_components: (optional) int
            optionally specify the number of components
            to project into.
    Returns:
        expl_var: torch tensor (D,)
    """
    _, preds = project_out(actvs, weight=weight, U=U,n_components=n_components)
    return explained_variance(preds, actvs)

if __name__=="__main__":
    U,_,Vt = torch.linalg.svd(torch.randn(10,10), full_matrices=True)
    proj, pred = project_out(U,U)
    assert explained_variance(pred, U).mean().item()==1
    assert projected_expl_var(U,U).mean().item()==1
    indys, cumu = component_expl_var(U,U)
    print("Indy Components:", indys.mean(-1))
    print("\tSum:", indys.sum(0).mean(-1))
    print("Cumu Components:", cumu.mean(-1))
    preds = matrix_proj(U,U)
    assert cumu.mean(-1)[-1]==explained_variance(preds, U).mean()