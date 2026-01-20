"""
Orthogonal Steering Example: Using OrthogonalProjection for LLM Activation Steering

This example demonstrates how to use the OrthogonalProjection class to learn
steering vectors that are constrained to be orthogonal to a set of fixed
reference directions.

Use Case:
    When steering LLM activations, you may want to modify the model's behavior
    while avoiding certain known directions (e.g., directions that cause
    harmful outputs, or directions that encode specific concepts you want to
    preserve). OrthogonalProjection lets you learn steering vectors that are
    guaranteed to be orthogonal to these "forbidden" directions.

The script:
1. Loads a pretrained GPT-2 model
2. Collects activations from a specified layer to create fixed reference vectors
3. Creates an OrthogonalProjection with trainable steering parameters
4. Applies steering via a forward hook that adds the orthogonal vectors to activations
5. Demonstrates generation with and without steering

Requirements:
    pip install transformers torch

Usage:
    # Basic usage with defaults
    python orthogonal_steering.py

    # Specify layer and number of vectors
    python orthogonal_steering.py layer=transformer.h.6 n_fixed=10 n_params=2

    # Adjust steering strength
    python orthogonal_steering.py steering_strength=0.5

Configuration Options:
    model_name       : str   - HuggingFace model name (default: "gpt2")
    layer            : str   - Layer to apply steering (default: "transformer.h.5")
    n_fixed          : int   - Number of fixed vectors to be orthogonal to (default: 5)
    n_params         : int   - Number of steering parameters (default: 1)
    steering_strength: float - Multiplier for steering vector (default: 1.0)
    prompt           : str   - Text prompt for generation
    max_new_tokens   : int   - Maximum tokens to generate (default: 50)
    device           : str   - Device to use (default: "cuda" if available)
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import OrthogonalProjection and utilities
from fca import OrthogonalProjection
from fca.utils import get_command_line_args


def get_layer_module(model, layer_name):
    """
    Retrieve a module from a model by its dot-separated name.

    Args:
        model: nn.Module - The model to search
        layer_name: str - Dot-separated path to the module (e.g., "transformer.h.5")

    Returns:
        nn.Module or None
    """
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def collect_reference_vectors(model, tokenizer, layer_name, n_vectors, device):
    """
    Collect activation vectors from the model to use as fixed reference directions.

    These could represent directions you want to avoid when steering (e.g.,
    directions associated with certain behaviors or concepts).

    Args:
        model: The language model
        tokenizer: The tokenizer
        layer_name: Layer to collect from
        n_vectors: Number of reference vectors to collect
        device: Torch device

    Returns:
        Tensor of shape (n_vectors, hidden_size)
    """
    # Sample prompts to collect diverse activations
    sample_prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Science is the systematic study of the natural world.",
        "Music has the power to evoke strong emotions.",
        "The economy is influenced by many complex factors.",
        "Technology continues to advance at a rapid pace.",
        "Nature provides countless examples of beauty and complexity.",
        "History teaches us valuable lessons about human behavior.",
        "Art reflects the culture and values of its time.",
        "Mathematics is the language of the universe.",
        "Philosophy explores fundamental questions about existence.",
    ]

    collected = []
    layer_module = get_layer_module(model, layer_name)

    def capture_hook(module, input, output):
        # Handle different output formats
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output
        # Take the last token's activation (most contextual)
        collected.append(hidden[0, -1, :].detach().clone())

    handle = layer_module.register_forward_hook(capture_hook)

    with torch.no_grad():
        for prompt in sample_prompts[:n_vectors]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)

    handle.remove()

    return torch.stack(collected)


def create_steering_hook(ortho_proj, steering_strength=1.0):
    """
    Create a forward hook that applies orthogonal steering to activations.

    The hook adds the steering vector(s) to all token positions in the sequence.

    Args:
        ortho_proj: OrthogonalProjection instance
        steering_strength: Scalar multiplier for the steering effect

    Returns:
        Callable hook function
    """
    def steering_hook(module, input, output):
        # Handle tuple outputs (common in transformer layers)
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Get the orthogonalized steering vector(s)
        # Shape: (n_params, hidden_size)
        steering_vectors = ortho_proj()

        # Sum steering vectors if multiple, then scale
        # Shape: (hidden_size,)
        combined_steering = steering_vectors.sum(dim=0) * steering_strength

        # Add steering to all positions in the sequence
        # hidden_states shape: (batch, seq_len, hidden_size)
        steered = hidden_states + combined_steering.unsqueeze(0).unsqueeze(0)

        if rest is not None:
            return (steered,) + rest
        return steered

    return steering_hook


def generate_text(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    # Default configuration
    config = {
        "model_name": "gpt2",
        "layer": "transformer.h.5",
        "n_fixed": 5,
        "n_params": 1,
        "steering_strength": 1.0,
        "prompt": "The future of artificial intelligence is",
        "max_new_tokens": 50,
        "dtype": "bfloat16",  # Model dtype (bfloat16, float16, float32)
    }

    # Override with command line arguments
    config = {**config, **get_command_line_args()}

    # Ensure numeric types
    config["n_fixed"] = int(config["n_fixed"])
    config["n_params"] = int(config["n_params"])
    config["steering_strength"] = float(config["steering_strength"])
    config["max_new_tokens"] = int(config["max_new_tokens"])

    print("=" * 60)
    print("Orthogonal Steering Example")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Layer: {config['layer']}")
    print(f"Fixed vectors: {config['n_fixed']}")
    print(f"Steering parameters: {config['n_params']}")
    print(f"Steering strength: {config['steering_strength']}")
    print(f"Dtype: {config['dtype']}")
    print()

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config["dtype"], torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    model.eval()

    # Get model device (for inputs)
    device = next(model.parameters()).device
    print(f"Device: {device}")

    # Get hidden size from model config
    hidden_size = model.config.n_embd
    print(f"Hidden size: {hidden_size}")

    # Collect reference vectors (directions to avoid)
    print(f"\nCollecting {config['n_fixed']} reference vectors from layer...")
    fixed_vectors = collect_reference_vectors(
        model, tokenizer, config["layer"], config["n_fixed"], device
    )
    print(f"Fixed vectors shape: {fixed_vectors.shape}")

    # Create OrthogonalProjection
    print(f"\nCreating OrthogonalProjection with {config['n_params']} steering parameter(s)...")
    ortho_proj = OrthogonalProjection(
        size=hidden_size,
        n_params=config["n_params"],
        fixed_vectors=fixed_vectors,
        normalize=True,  # Normalize steering vectors to unit length
        init_noise=0.1,
    ).to(device)

    # Verify orthogonality
    orth_check = ortho_proj.check_orthogonality()
    print(f"Orthogonality check - max dot product with fixed vectors: {orth_check['max_dot_product']:.2e}")
    print(f"Is orthogonal (tol=1e-5): {orth_check['is_orthogonal']}")

    # Get the layer module for hooking
    layer_module = get_layer_module(model, config["layer"])
    if layer_module is None:
        available = [n for n, _ in model.named_modules() if n]
        raise ValueError(
            f"Layer '{config['layer']}' not found. "
            f"Available layers include: {available[:10]}..."
        )

    # Generate without steering (baseline)
    print("\n" + "=" * 60)
    print("Generation WITHOUT steering:")
    print("=" * 60)
    print(f"Prompt: {config['prompt']}")
    baseline_output = generate_text(
        model, tokenizer, config["prompt"], config["max_new_tokens"]
    )
    print(f"Output: {baseline_output}")

    # Apply steering hook
    steering_hook = create_steering_hook(ortho_proj, config["steering_strength"])
    handle = layer_module.register_forward_hook(steering_hook)

    # Generate with steering
    print("\n" + "=" * 60)
    print("Generation WITH orthogonal steering:")
    print("=" * 60)
    print(f"Prompt: {config['prompt']}")
    steered_output = generate_text(
        model, tokenizer, config["prompt"], config["max_new_tokens"]
    )
    print(f"Output: {steered_output}")

    # Clean up
    handle.remove()

    # Show steering vector statistics
    print("\n" + "=" * 60)
    print("Steering Vector Statistics:")
    print("=" * 60)
    steering_vecs = ortho_proj()
    print(f"Steering vector shape: {steering_vecs.shape}")
    print(f"Steering vector norm: {steering_vecs.norm(dim=-1)}")
    print(f"Steering vector mean: {steering_vecs.mean():.4f}")
    print(f"Steering vector std: {steering_vecs.std():.4f}")

    # Demonstrate that steering is orthogonal to fixed vectors
    print("\n" + "=" * 60)
    print("Orthogonality Verification:")
    print("=" * 60)
    dot_products = torch.matmul(steering_vecs, fixed_vectors.T)
    print(f"Dot products with fixed vectors:\n{dot_products}")
    print(f"Max absolute dot product: {dot_products.abs().max():.2e}")

    print("\n" + "=" * 60)
    print("Example Training Loop (not executed):")
    print("=" * 60)
    print("""
    # The steering parameters can be trained while maintaining orthogonality:

    optimizer = torch.optim.Adam(ortho_proj.parameters(), lr=0.01)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Get orthogonalized steering vectors (differentiable)
        steering_vecs = ortho_proj()

        # Your loss function here (e.g., maximize some behavior)
        loss = compute_steering_loss(model, steering_vecs, ...)

        loss.backward()  # Gradients flow through orthogonalization
        optimizer.step()

        # Orthogonality is maintained automatically!
    """)


if __name__ == "__main__":
    main()
