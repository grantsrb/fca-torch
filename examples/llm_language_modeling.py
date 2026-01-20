"""
FCA Example: Language Modeling with a HuggingFace LLM

This example demonstrates how to use Functional Component Analysis (FCA) to find
sufficient representational subspaces in a language model. We train FCA components
to preserve next-token prediction performance while minimizing the dimensionality.

Usage:
    python llm_language_modeling.py
    python llm_language_modeling.py model_name=gpt2-medium layer=transformer.h.8

Results are saved to the `save_files/` directory.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add parent directory to path for local development
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fca import FunctionalComponentAnalysis
from fca.utils import get_output_size, save_json


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "model_name": "gpt2",                    # HuggingFace model name
    "layer": "transformer.h.5",              # Layer to apply FCA to
    "dataset": "wikitext",                   # Dataset name
    "dataset_config": "wikitext-2-raw-v1",   # Dataset configuration
    "max_length": 128,                       # Maximum sequence length
    "batch_size": 8,                         # Batch size
    "num_samples": 1000,                     # Number of training samples
    "init_rank": 1,                          # Initial FCA rank
    "max_rank": 50,                          # Maximum FCA rank
    "lr": 1e-3,                              # Learning rate
    "epochs": 20,                            # Training epochs per rank
    "patience": 3,                           # Epochs without improvement before adding rank
    "target_acc": 0.95,                      # Target accuracy (fraction of original)
    "save_dir": "save_files",                # Directory to save results
    "dtype": "bfloat16",                     # Model dtype (bfloat16, float16, float32)
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


def parse_args():
    """Parse command line arguments in key=value format."""
    config = DEFAULT_CONFIG.copy()
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Type conversion
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").replace("-", "").isdigit():
                value = float(value)
            config[key] = value
    return config


# =============================================================================
# Data Preparation
# =============================================================================

def prepare_data(config, tokenizer):
    """Load and prepare the language modeling dataset."""
    print(f"Loading dataset: {config['dataset']}/{config['dataset_config']}")

    dataset = load_dataset(
        config["dataset"],
        config["dataset_config"],
        split="train"
    )

    # Filter empty texts
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    # Limit samples
    if len(dataset) > config["num_samples"]:
        dataset = dataset.select(range(config["num_samples"]))

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
            return_tensors="pt",
        )

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        encodings = tokenizer(
            texts,
            truncation=True,
            max_length=config["max_length"],
            padding="max_length",
            return_tensors="pt",
        )
        return encodings

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    return dataloader


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataloader, device):
    """Evaluate language modeling performance (next-token prediction accuracy)."""
    model.eval()
    total_correct = 0
    total_tokens = 0
    total_loss = 0
    n_batches = 0

    criterion = nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Shift for language modeling: predict next token
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Flatten for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # Mask padding tokens
            mask = labels_flat != -100
            if attention_mask is not None:
                mask = mask & (attention_mask.view(-1) == 1)

            valid_logits = logits_flat[mask]
            valid_labels = labels_flat[mask]

            if len(valid_labels) > 0:
                loss = criterion(valid_logits, valid_labels)
                total_loss += loss.item()

                preds = valid_logits.argmax(dim=-1)
                total_correct += (preds == valid_labels).sum().item()
                total_tokens += len(valid_labels)

            n_batches += 1

    accuracy = total_correct / max(1, total_tokens)
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "accuracy": accuracy,
        "loss": avg_loss,
        "perplexity": perplexity,
    }


# =============================================================================
# Training
# =============================================================================

def train_fca(model, fca, dataloader, optimizer, device, config):
    """Train FCA for one epoch."""
    model.train()  # Enable hooks but model weights are frozen
    total_loss = 0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Shift for language modeling
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        attention_mask = attention_mask[:, :-1].contiguous()

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Compute loss
        logits_flat = logits.view(-1, logits.size(-1))
        labels_flat = labels.view(-1)

        # Mask padding
        mask = (attention_mask.view(-1) == 1)
        valid_logits = logits_flat[mask]
        valid_labels = labels_flat[mask]

        if len(valid_labels) > 0:
            loss = criterion(valid_logits, valid_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Reset cached weight after optimizer step
        fca.reset_cached_weight()
        n_batches += 1

    return total_loss / max(1, n_batches)


# =============================================================================
# Main
# =============================================================================

def main():
    config = parse_args()

    print("=" * 60)
    print("FCA Language Modeling Example")
    print("=" * 60)
    print(f"Model: {config['model_name']}")
    print(f"Layer: {config['layer']}")
    print(f"Device: {config['device']}")
    print()

    # Create save directory
    os.makedirs(config["save_dir"], exist_ok=True)

    # Load model and tokenizer
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Get model device (for inputs)
    device = next(model.parameters()).device

    # Prepare data
    dataloader = prepare_data(config, tokenizer)
    print(f"Loaded {len(dataloader.dataset)} samples")

    # Get baseline performance
    print("\nEvaluating baseline model...")
    baseline = evaluate(model, dataloader, device)
    print(f"Baseline - Accuracy: {baseline['accuracy']:.4f}, "
          f"Perplexity: {baseline['perplexity']:.2f}")

    # Determine layer output size
    sample_batch = next(iter(dataloader))
    sample_input = {k: v.to(device) for k, v in sample_batch.items()}

    size = get_output_size(model, config["layer"], data_sample=sample_input)
    print(f"\nLayer '{config['layer']}' output size: {size}")

    # Create FCA
    fca = FunctionalComponentAnalysis(
        size=size,
        init_rank=config["init_rank"],
        max_rank=config["max_rank"],
        orth_method="hybrid",
    )
    fca.to(device)

    # Match model dtype
    if torch_dtype != torch.float32:
        fca = fca.to(torch_dtype)

    # Attach FCA to layer
    handle = fca.hook_model_layer(model, config["layer"])
    fca.set_cached(True)

    print(f"\nFCA attached to '{config['layer']}' with {fca.rank} components")

    # Training loop
    optimizer = torch.optim.Adam(fca.parameters(), lr=config["lr"])

    best_acc = 0
    patience_counter = 0
    history = []

    print("\n" + "=" * 60)
    print("Training FCA")
    print("=" * 60)

    target_acc = baseline["accuracy"] * config["target_acc"]
    print(f"Target accuracy: {target_acc:.4f} ({config['target_acc']*100:.0f}% of baseline)")
    print()

    for epoch in range(config["epochs"] * config["max_rank"]):
        # Train
        train_loss = train_fca(model, fca, dataloader, optimizer, device, config)

        # Evaluate
        metrics = evaluate(model, dataloader, device)

        history.append({
            "epoch": epoch,
            "rank": fca.rank,
            "train_loss": train_loss,
            "accuracy": metrics["accuracy"],
            "perplexity": metrics["perplexity"],
        })

        print(f"Epoch {epoch+1:3d} | Rank {fca.rank:2d} | "
              f"Loss: {train_loss:.4f} | Acc: {metrics['accuracy']:.4f} | "
              f"PPL: {metrics['perplexity']:.2f}")

        # Check for improvement
        if metrics["accuracy"] > best_acc + 0.001:
            best_acc = metrics["accuracy"]
            patience_counter = 0

            # Save best model
            torch.save({
                "fca_state_dict": fca.state_dict(),
                "rank": fca.rank,
                "accuracy": metrics["accuracy"],
            }, os.path.join(config["save_dir"], "fca_best.pt"))
        else:
            patience_counter += 1

        # Check if target reached
        if metrics["accuracy"] >= target_acc:
            print(f"\n✓ Target accuracy reached with {fca.rank} components!")
            break

        # Add component if plateaued
        if patience_counter >= config["patience"]:
            if fca.rank >= config["max_rank"]:
                print("\nMax rank reached, stopping.")
                break

            print(f"  → Adding component (rank {fca.rank} -> {fca.rank + 1})")
            fca.freeze_parameters()
            fca.add_component()
            optimizer = torch.optim.Adam(fca.parameters(), lr=config["lr"])
            patience_counter = 0

    # Remove hook
    handle.remove()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    # Reattach for final eval
    handle = fca.hook_model_layer(model, config["layer"])
    final_metrics = evaluate(model, dataloader, device)
    handle.remove()

    print(f"Baseline Accuracy:  {baseline['accuracy']:.4f}")
    print(f"FCA Accuracy:       {final_metrics['accuracy']:.4f}")
    print(f"Accuracy Ratio:     {final_metrics['accuracy']/baseline['accuracy']:.2%}")
    print(f"Final Rank:         {fca.rank}")
    print(f"Compression:        {size} → {fca.rank} ({fca.rank/size:.1%})")

    # Save final results
    results = {
        "config": config,
        "baseline": baseline,
        "final": final_metrics,
        "final_rank": fca.rank,
        "history": history,
    }
    save_json(results, os.path.join(config["save_dir"], "results.json"))

    torch.save({
        "fca_state_dict": fca.state_dict(),
        "rank": fca.rank,
        "config": config,
    }, os.path.join(config["save_dir"], "fca_final.pt"))

    print(f"\nResults saved to '{config['save_dir']}/'")


if __name__ == "__main__":
    main()
