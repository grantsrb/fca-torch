# TODO:
#   - Test Wrappers
#       - regular pytorch models
#       - huggingface models
#   - build easy way to make complement data for text models
#

import torch
import torch.nn.functional as F

class CategoricalModelWrapper(torch.nn.Module):
    """
    A lightweight wrapper class for basic pytorch models that produce
    categorical predictions. This wrapper makes such models compatible
    with the FCA find_sufficient_components function.
    """
    def __init__(self, model, loss_fn=None, acc_fn=None):
        """
        Args:
            model: nn.Module
                The model to wrap.
            loss_fn: callable(preds, labels) → loss
                A loss function
            acc_fn: callable(preds, labels) → loss
                An accuracy function
        """
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn or self.default_loss_fn
        self.acc_fn = acc_fn or self.default_acc_fn

    def default_loss_fn(self, logits, labels):
        """
        Args:
            logits: torch Tensor (B, ..., P)
            labels: torch Tensor (B, ...)
        Returns:
            acc: torch Tensor (1,)
        """
        return F.cross_entropy(logits, labels)

    def masked_loss_fn(self, logits, labels, mask=None):
        """
        Function that returns the accuracy of the predictions

        Args:
            logits: torch Tensor (B, ..., P)
            labels: torch Tensor (B, ...)
            mask: torch BoolTensor (B, ...)
        Returns:
            acc: torch Tensor (1,)
        """
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        if mask is not None:
            logits = logits[mask.reshape(-1)]
            labels = labels[mask.reshape(-1)]
        return self.loss_fn(logits, labels)

    def default_acc_fn(self, preds, labels):
        """
        Args:
            preds: torch Tensor (B, ...)
            labels: torch Tensor (B, ...)
        Returns:
            acc: torch Tensor (1,)
        """
        return (preds == labels).float().mean()

    def masked_acc_fn(self, logits, labels, mask=None):
        """
        Function that returns the accuracy of the predictions

        Args:
            logits: torch Tensor (B, ..., P)
            labels: torch Tensor (B, ...)
            mask: torch BoolTensor (B, ...)
        Returns:
            acc: torch Tensor (1,)
        """
        logits = logits.reshape(-1, logits.shape[-1])
        labels = labels.reshape(-1)
        if mask is not None:
            logits = logits[mask.reshape(-1)]
            labels = labels[mask.reshape(-1)]
        preds = torch.argmax(logits, dim=-1)
        return self.acc_fn(preds, labels)

    def get_device(self):
        device = next(self.parameters()).get_device()
        return device if device>=0 else "cpu"

    def subforward(self, **kwargs):
        """
        Assumes the model returns logits and optionally uses 'labels' to compute loss.
        """
        device = self.get_device()
        kwargs = {k: v.to(device) for k,v in kwargs.items()}
        labels = kwargs.pop("labels", None)
        mask = kwargs.pop("mask", None) # true values denote
            # values to keep in the loss and accuracy calculations
        try:
            outputs = self.model(**kwargs)
        except:
            k = list(kwargs.keys())[0]
            outputs = self.model(kwargs[k])

        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        else:
            logits = outputs  # assume raw tensor
        
        result = {}
        if labels is not None:
            result["loss"] = self.masked_loss_fn(logits, labels, mask=mask)
            result["acc"]  = self.masked_acc_fn(logits, labels, mask=mask)
        else:
            result["loss"] = torch.tensor(0.0, device=logits.device)
            result["acc"] = torch.tensor(0.0, device=logits.device)
        
        return result

    def forward(self, fca_ref=None, **kwargs):
        """
        Args:
            fca_ref: (optional) FunctionalComponentAnalysis
                optionally argue an fca reference for complement training
            kwargs: dict
                the data argued using the double star notation. If training
                the complement, must argue the normal data under the
                keyword "data" and the complement data under "complement" 
        """
        result = dict()        
        if "data" not in kwargs and "complement" not in kwargs:
            result = self.subforward(**kwargs)
        elif "data" in kwargs:
            result = self.subforward(**kwargs["data"])
        if "complement" in kwargs and fca_ref is not None:
            prior_state = fca_ref.use_complement_in_hook
            fca_ref.use_complement_in_hook = True
            result["complement"] = self.subforward(**kwargs["complement"])
            fca_ref.use_complement_in_hook = prior_state
        return result

class ContinuousModelWrapper(CategoricalModelWrapper):
    def __init__(self,*args,**kwargs):
        super().__init__(*args, **kwargs)

    def default_loss_fn(self, preds, labels):
        """
        Function that returns the loss of the predictions

        Args:
            preds: torch Tensor (B, ...)
            labels: torch Tensor (B, ...)
        Returns:
            loss: torch Tensor (1,)
        """
        return F.mse_loss(preds, labels)

    def default_acc_fn(self, preds, labels):
        """
        Function that returns the accuracy of the predictions

        Args:
            preds: torch Tensor (B, ...)
            labels: torch Tensor (B, ...)
        Returns:
            acc: torch Tensor (1,)
        """
        return 1-F.mse_loss(preds, labels)

class DataWrapper(torch.utils.data.Dataset):
    """
    Lightweight wrapper to create torch data loader object.
    
    Args:
        data: dict
        complement_data: (optional) dict 
            if you want to train the complement of the fca object, you
            can argue a separate complement dataset here.
    """
    def __init__(self, data, complement_data=None, collate_fn=None):
        self.data = data
        self.complement_data = complement_data
        if collate_fn is not None: self.collate_fn = collate_fn
        if type(self.data)==dict:
            self.size = len(self.data[list(self.data.keys())[0]])
        else:
            self.size = len(self.data)

    def collate_fn(self, idx, data, complement=None):
        """
        Optionally argue a complement dataset.
        """
        data_dict = {
            "data": {
                k: data[k][idx] for k in data.keys()
            },
        }
        if complement is not None:
            data_dict["complement"] = {
                k: complement[k][idx] for k in complement.keys()
            }
        return data_dict
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.collate_fn(idx, self.data, complement=self.complement_data)

def wrap_data(data, complement_data=None, shuffle=True, batch_size=128):
    """
    Wraps your data with a lightweight wrapper in order to create a torch
    DataLoader object.

    Args:
        data: Arrow Dataset or dict
        complement_data: Arrow Dataset or dict
        data_type: str
            valid options: text, images
    Returns:
        data_loader: DataLoader
    """
    wrapped = DataWrapper(data=data, complement_data=complement_data)
    data_loader = torch.utils.data.DataLoader(
        wrapped, shuffle=shuffle, batch_size=batch_size,
    )
    return data_loader

def wrapped_kl_divergence(preds, labels, preds_arg_logits=True):
    """
    Computes the KL divergence between two categorical distributions.
    
    Args:
        preds: torch Tensor (B, P)
            Predicted probabilities.
        labels: torch Tensor (B, P)
            True probabilities.
        preds_arg_logits: bool
            If True, assumes preds are logits and applies log softmax.
            If False, assumes preds are probabilities and applies log.
    Returns:
        kl_div: torch Tensor (1,)
            The KL divergence value.
    """
    if preds_arg_logits:
        preds = F.log_softmax(preds, dim=-1)
    else:
        preds = preds.log()
    return F.kl_div(preds, labels, reduction='batchmean')

__all__ = [
    "wrap_data", "DataWrapper", "ContinuousModelWrapper",
    "CategoricalModelWrapper"
]