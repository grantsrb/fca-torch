__version__ = '0.1.0'
from . import fca, projections, schedulers, wrappers, utils
from .fca import *
from .projections import *
from .schedulers import *
from .wrappers import *
from .utils import *


def attach_fca(
    model,
    layer_name,
    fca_instance,
    output_format="auto",
    use_complement=None,
    output_extractor=None,
    shape_transform=None,
    inverse_transform=None,
    comms_dict=None,
):
    """
    Convenience function to attach FCA to any PyTorch model layer.

    This is a high-level API for attaching FCA hooks to models. It handles
    common output formats automatically and provides sensible defaults.

    Args:
        model: torch.nn.Module
            Any PyTorch model.
        layer_name: str
            Name of the layer to attach to. Use `model.named_modules()` to
            find available layer names.
        fca_instance: FunctionalComponentAnalysis
            An initialized FCA object with the appropriate size for the layer.
        output_format: str
            How the layer outputs data. Options:
            - "auto": Attempts automatic detection (default)
            - "tensor": Layer outputs a raw tensor
            - "image": Layer outputs (B, C, H, W) image tensors
            - "sequence": Layer outputs (B, S, D) sequence tensors
            - "tuple": Layer outputs (tensor, ...) tuple
            - "dict": Layer outputs dict with 'hidden_states' key
            - "dict:key_name": Layer outputs dict with custom key
        use_complement: bool or None
            If True, the hook removes FCA components (keeps complement).
            If False, the hook keeps only FCA components.
            If None, uses the FCA instance's `use_complement_in_hook` setting.
        output_extractor: callable or None
            Custom function to extract tensor from output.
            Signature: output_extractor(output) -> tensor
        shape_transform: callable or None
            Custom function to transform tensor shape for FCA.
            Signature: shape_transform(tensor) -> (flat_tensor, original_shape)
        inverse_transform: callable or None
            Custom function to restore original tensor shape.
            Signature: inverse_transform(tensor, original_shape) -> tensor
        comms_dict: dict or None
            Optional dict to collect original activations before FCA.

    Returns:
        handle: torch hook handle
            Call `handle.remove()` to detach the FCA hook.

    Raises:
        ValueError: If layer_name is not found in model.

    Examples:
        Basic usage with a ResNet:
        >>> import torchvision.models as models
        >>> from fca import FunctionalComponentAnalysis, attach_fca
        >>>
        >>> model = models.resnet18(pretrained=True).eval()
        >>> fca = FunctionalComponentAnalysis(size=64, init_rank=10)
        >>>
        >>> handle = attach_fca(
        ...     model=model,
        ...     layer_name="layer1.0.conv2",
        ...     fca_instance=fca,
        ...     output_format="image",
        ... )
        >>> output = model(torch.randn(1, 3, 224, 224))
        >>> handle.remove()  # Clean up when done

        Usage with HuggingFace BERT:
        >>> from transformers import AutoModel
        >>> model = AutoModel.from_pretrained("bert-base-uncased").eval()
        >>> fca = FunctionalComponentAnalysis(size=768, init_rank=10)
        >>>
        >>> handle = attach_fca(
        ...     model=model,
        ...     layer_name="encoder.layer.5.output.dense",
        ...     fca_instance=fca,
        ...     output_format="sequence",
        ... )
    """
    from .utils import (
        identity_extractor,
        first_element_extractor,
        dict_extractor,
        image_to_flat,
        flat_to_image,
        sequence_to_flat,
        flat_to_sequence,
    )

    # Set complement behavior if specified
    if use_complement is not None:
        fca_instance.use_complement_in_hook = use_complement

    # Determine rep_type and output_extractor from output_format
    rep_type = "auto"
    if output_format == "image":
        rep_type = "image"
    elif output_format == "sequence":
        rep_type = "sequence"
    elif output_format == "tensor":
        rep_type = "flat"
        if output_extractor is None:
            output_extractor = identity_extractor
    elif output_format == "tuple":
        rep_type = "auto"
        if output_extractor is None:
            output_extractor = first_element_extractor
    elif output_format == "dict":
        rep_type = "auto"
        if output_extractor is None:
            output_extractor = dict_extractor("hidden_states")
    elif output_format.startswith("dict:"):
        rep_type = "auto"
        key = output_format.split(":", 1)[1]
        if output_extractor is None:
            output_extractor = dict_extractor(key)

    # Attach the hook using the FCA's hook_model_layer method
    handle = fca_instance.hook_model_layer(
        model=model,
        layer=layer_name,
        comms_dict=comms_dict,
        rep_type=rep_type,
        output_extractor=output_extractor,
        shape_transform=shape_transform,
        inverse_transform=inverse_transform,
    )

    if handle is None:
        available_layers = [name for name, _ in model.named_modules()]
        raise ValueError(
            f"Layer '{layer_name}' not found in model. "
            f"Available layers: {available_layers[:10]}... "
            f"(use model.named_modules() to see all)"
        )

    return handle


def get_layer_output_size(model, layer_name, sample_input=None):
    """
    Helper function to determine the output size of a layer.

    Useful for initializing an FCA object with the correct size.

    Args:
        model: torch.nn.Module
        layer_name: str
            Name of the layer.
        sample_input: torch.Tensor or None
            Optional sample input to run through the model.
            If provided, extracts size from actual output.
            If None, tries to infer from layer attributes.

    Returns:
        int: The output feature dimension of the layer.

    Raises:
        ValueError: If layer not found or size cannot be determined.

    Example:
        >>> size = get_layer_output_size(model, "layer1.0.conv2")
        >>> fca = FunctionalComponentAnalysis(size=size, init_rank=10)
    """
    import torch

    # Find the layer
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found in model.")

    # Try to get size from module attributes
    if hasattr(target_module, 'out_features'):
        return target_module.out_features
    if hasattr(target_module, 'out_channels'):
        return target_module.out_channels
    if hasattr(target_module, 'hidden_size'):
        return target_module.hidden_size
    if hasattr(target_module, 'embed_dim'):
        return target_module.embed_dim
    if hasattr(target_module, 'weight'):
        weight = target_module.weight
        if len(weight.shape) >= 1:
            return weight.shape[0]

    # If we have sample input, run it through to get the size
    if sample_input is not None:
        output_size = [None]

        def capture_hook(module, input, output):
            if isinstance(output, torch.Tensor):
                output_size[0] = output.shape[-1]
            elif isinstance(output, tuple) and len(output) > 0:
                output_size[0] = output[0].shape[-1]
            elif hasattr(output, 'last_hidden_state'):
                output_size[0] = output.last_hidden_state.shape[-1]

        handle = target_module.register_forward_hook(capture_hook)
        with torch.no_grad():
            model(sample_input)
        handle.remove()

        if output_size[0] is not None:
            return output_size[0]

    raise ValueError(
        f"Could not determine output size for layer '{layer_name}'. "
        f"Please provide a sample_input or specify size manually."
    )


__all__ = [
    # Core classes
    "FunctionalComponentAnalysis",
    "PCAFunctionalComponentAnalysis",
    "UnnormedFCA",
    "OrthogonalProjection",
    # Convenience functions
    "attach_fca",
    "get_layer_output_size",
    # Loading functions
    "load_fca_from_path",
    "load_fcas_from_path",
    "load_fcas",
    "load_ortho_fcas",
    # Utilities
    "gram_schmidt",
    "orthogonalize_vector",
    "orthogonalize_vector_mgs",
    "orthogonalize_batch_qr",
    # Extractors
    "identity_extractor",
    "first_element_extractor",
    "dict_extractor",
    "last_hidden_state_extractor",
    # Transforms
    "image_to_flat",
    "flat_to_image",
    "sequence_to_flat",
    "flat_to_sequence",
    "vit_to_flat",
    "flat_to_vit",
]

