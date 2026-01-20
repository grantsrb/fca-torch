"""
FCA Example: Image Classification with ResNet18 on CIFAR-10

This example demonstrates how to use Functional Component Analysis (FCA) to find
sufficient representational subspaces in a vision model. We train FCA components
to preserve classification accuracy while minimizing the dimensionality.

Usage:
    python resnet_cifar10.py
    python resnet_cifar10.py layer=layer2.1.conv2 init_rank=5

Results are saved to the `save_files/` directory.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# Add parent directory to path for local development
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fca import FunctionalComponentAnalysis, attach_fca
from fca.utils import save_json


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_CONFIG = {
    "layer": "layer3.1.conv2",           # Layer to apply FCA to
    "batch_size": 128,                   # Batch size
    "num_train": 10000,                  # Number of training samples
    "num_test": 2000,                    # Number of test samples
    "init_rank": 1,                      # Initial FCA rank
    "max_rank": 64,                      # Maximum FCA rank
    "lr": 1e-3,                          # Learning rate
    "epochs": 15,                        # Training epochs per rank
    "patience": 3,                       # Epochs without improvement before adding rank
    "target_acc": 0.95,                  # Target accuracy (fraction of original)
    "save_dir": "save_files",            # Directory to save results
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

def prepare_data(config):
    """Load and prepare CIFAR-10 dataset."""
    print("Loading CIFAR-10 dataset...")

    # Standard CIFAR-10 transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    # Limit samples if specified
    if config["num_train"] and config["num_train"] < len(train_dataset):
        train_dataset = torch.utils.data.Subset(
            train_dataset, range(config["num_train"])
        )
    if config["num_test"] and config["num_test"] < len(test_dataset):
        test_dataset = torch.utils.data.Subset(
            test_dataset, range(config["num_test"])
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=2,
    )

    return train_loader, test_loader


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(model, dataloader, device):
    """Evaluate classification accuracy."""
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return {"accuracy": accuracy, "loss": avg_loss}


# =============================================================================
# Training
# =============================================================================

def train_fca(model, fca, dataloader, optimizer, device):
    """Train FCA for one epoch."""
    model.eval()  # Model in eval mode, but hooks still active
    total_loss = 0
    total_correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        # Reset cached weight after optimizer step
        fca.reset_cached_weight()

        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    accuracy = total_correct / total_samples
    avg_loss = total_loss / total_samples

    return {"loss": avg_loss, "accuracy": accuracy}


# =============================================================================
# Utility
# =============================================================================

def get_layer_channels(model, layer_name, device):
    """Determine the number of output channels for a conv layer."""
    # Register a hook to capture output shape
    output_shape = [None]

    def hook(module, input, output):
        output_shape[0] = output.shape

    # Find the layer
    target_module = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_module = module
            break

    if target_module is None:
        raise ValueError(f"Layer '{layer_name}' not found")

    handle = target_module.register_forward_hook(hook)

    # Run a forward pass
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        model(dummy_input)

    handle.remove()

    # For conv layers, output is (B, C, H, W), return C
    return output_shape[0][1]


def print_available_layers(model):
    """Print available layer names for reference."""
    print("\nAvailable layers:")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"  {name}: {type(module).__name__}")


# =============================================================================
# Main
# =============================================================================

def main():
    config = parse_args()

    print("=" * 60)
    print("FCA ResNet18 CIFAR-10 Example")
    print("=" * 60)
    print(f"Layer: {config['layer']}")
    print(f"Device: {config['device']}")
    print()

    # Create save directory
    os.makedirs(config["save_dir"], exist_ok=True)

    # Load pretrained ResNet18
    print("Loading pretrained ResNet18...")
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

    # Modify for CIFAR-10 (32x32 images, 10 classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    model.fc = nn.Linear(model.fc.in_features, 10)

    model.to(config["device"])

    # Freeze model weights
    for param in model.parameters():
        param.requires_grad = False

    # Show available layers
    print_available_layers(model)

    # Prepare data
    train_loader, test_loader = prepare_data(config)
    print(f"\nTrain samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Get baseline performance
    print("\nEvaluating baseline model...")
    baseline = evaluate(model, test_loader, config["device"])
    print(f"Baseline - Accuracy: {baseline['accuracy']:.4f}, Loss: {baseline['loss']:.4f}")

    # Determine layer output channels
    num_channels = get_layer_channels(model, config["layer"], config["device"])
    print(f"\nLayer '{config['layer']}' has {num_channels} channels")

    # Create FCA
    fca = FunctionalComponentAnalysis(
        size=num_channels,
        init_rank=config["init_rank"],
        max_rank=config["max_rank"],
        orth_method="hybrid",
    )
    fca.to(config["device"])

    # Attach FCA to layer with image format
    handle = attach_fca(
        model=model,
        layer_name=config["layer"],
        fca_instance=fca,
        output_format="image",  # Handles (B, C, H, W) -> (B*H*W, C) automatically
    )
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

    epoch = 0
    while True:
        # Train
        train_metrics = train_fca(
            model, fca, train_loader, optimizer, config["device"]
        )

        # Evaluate on test set
        test_metrics = evaluate(model, test_loader, config["device"])

        history.append({
            "epoch": epoch,
            "rank": fca.rank,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
        })

        print(f"Epoch {epoch+1:3d} | Rank {fca.rank:2d} | "
              f"Train Acc: {train_metrics['accuracy']:.4f} | "
              f"Test Acc: {test_metrics['accuracy']:.4f}")

        # Check for improvement
        if test_metrics["accuracy"] > best_acc + 0.005:
            best_acc = test_metrics["accuracy"]
            patience_counter = 0

            # Save best model
            torch.save({
                "fca_state_dict": fca.state_dict(),
                "rank": fca.rank,
                "test_accuracy": test_metrics["accuracy"],
            }, os.path.join(config["save_dir"], "fca_best.pt"))
        else:
            patience_counter += 1

        # Check if target reached
        if test_metrics["accuracy"] >= target_acc:
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

        epoch += 1

        # Safety limit
        if epoch >= config["epochs"] * config["max_rank"]:
            print("\nMax epochs reached, stopping.")
            break

    # Remove hook
    handle.remove()

    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Results")
    print("=" * 60)

    # Reattach for final eval
    handle = attach_fca(
        model=model,
        layer_name=config["layer"],
        fca_instance=fca,
        output_format="image",
    )
    final_metrics = evaluate(model, test_loader, config["device"])
    handle.remove()

    print(f"Baseline Accuracy:  {baseline['accuracy']:.4f}")
    print(f"FCA Accuracy:       {final_metrics['accuracy']:.4f}")
    print(f"Accuracy Ratio:     {final_metrics['accuracy']/baseline['accuracy']:.2%}")
    print(f"Final Rank:         {fca.rank}")
    print(f"Compression:        {num_channels} → {fca.rank} ({fca.rank/num_channels:.1%})")

    # Save final results
    results = {
        "config": config,
        "baseline": baseline,
        "final": final_metrics,
        "final_rank": fca.rank,
        "num_channels": num_channels,
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
