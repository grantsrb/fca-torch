import os
import csv
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import matplotlib.pyplot as plt
from fca.fca import FunctionalComponentsAnalysis, load_ortho_fcas  # Assuming you have a custom FCA module
from fca.utils import get_command_line_args, get_output_size
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

# --------- Configuration ---------
ROOT_DIR = os.getcwd()
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 16
TARGET_LAYER_NAME = "transformer.layer.5.output"
TOLERANCE = 0.01
PATIENCE = 3
LEARNING_RATE = 1e-3
LOG_EVERY = 2
MAX_EPOCHS = 100
DATASET_NAME = "toxicity"  # replace with 'jigsaw-toxic-comment-classification' if using a local version
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == "__main__":
    print("Running Hugging Face Toxicity Example...")
    config = {
        "root_dir": ROOT_DIR,
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "target_layer": TARGET_LAYER_NAME,
        "tolerance": TOLERANCE,
        "patience": PATIENCE,
        "lr": LEARNING_RATE,
        "dataset": DATASET_NAME,
        "ortho_fcas": None,  # Path to orthogonal FCA components if any
    }
    config.update(get_command_line_args())

    # --------- Logging Setup ---------
    ROOT_DIR = config["root_dir"]
    MODEL_NAME = config["model_name"]
    LOG_DIR = f"{ROOT_DIR}/{MODEL_NAME}/run_{RUN_ID}"
    config["log_dir"] = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    CSV_PATH = os.path.join(LOG_DIR, "results.csv")
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "avg_loss", "behavior_match", "num_components"])

    # --------- Dataset & Tokenizer ---------
    dataset = load_dataset(config["dataset"], split="train[:5000]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def collate_fn(batch):
        texts = [item["text"] for item in batch]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        return inputs

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # --------- Model Setup ---------
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,)
    base_model.eval()
    for param in base_model.parameters():
        param.requires_grad = False

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2)
    model.eval()
    size = get_output_size(
        model=model,
        layer_name=TARGET_LAYER_NAME,
        data_sample=next(iter(dataloader))
    )
    config["fca_params"] = {
        "size": size,
        "max_components": size,
    }

    # --------- Initial Model Performance ---------

    with torch.no_grad():
        total_loss, total_match = 0.0, 0.0
        for batch in dataloader:
            output = base_model(**batch)
            loss = F.cross_entropy(output.logits, batch["labels"])
            total_loss += loss.item()
            acc = (output.logits.argmax(dim=-1) == batch["labels"]).float().mean().item()
            total_match += acc
        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
    print(f"Initial Model Performance: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
    config["initial_loss"] = avg_loss
    config["initial_match"] = avg_match
        
    with open(os.path.join(LOG_DIR, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # --------- Functional Components Analysis (FCA) Setup ---------

    fca = FunctionalComponentsAnalysis(**config["fca_params"])
    if "ortho_fcas" in config and config["ortho_fcas"]:
        fca = load_ortho_fcas(fca, config["ortho_fcas"])

    comms_dict = dict()
    hook = fca.hook_model_layer(
        model=model,
        layer=TARGET_LAYER_NAME,
        comms_dict=comms_dict)

    optimizer = torch.optim.Adam(fca.parameters(), lr=config["lr"])
    criterion = wrapped_kl_divergence

    # --------- Training Loop ---------
    loss_history = []
    match_history = []
    best_loss = float('inf')
    best_match = 0.0

    for epoch in range(MAX_EPOCHS):
        total_loss, total_match = 0.0, 0.0
        for batch in dataloader:
            with torch.no_grad():
                base_output = F.softmax(base_model(**batch).logits)

            logits = model(**batch).logits
            loss = criterion(logits, base_output)
            total_loss += loss.item()

            match = (logits.argmax(dim=1) == base_output.argmax(dim=1)).float().mean().item()
            total_match += match

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            fca.reset()

        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        loss_history.append(avg_loss)
        match_history.append(avg_match)

        with open(CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch, avg_loss, avg_match, fca.num_components
            ])

        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Match={avg_match:.4f}, Components={fca.num_components}")

        if len(loss_history) > config["patience"]:
            if max(loss_history[-config["patience"]:]) -\
                    min(loss_history[-config["patience"]:]) < 1e-4:
                if fca.num_components >= fca.max_components:
                    print("Stopping early due to convergence.")
                    break
                fca.add_component()

        if avg_match >= (1 - config["tolerance"]):
            print("✅ Stopping: Match >= 99%")
            break

        if epoch % LOG_EVERY == 0:
            torch.save(
                fca.state_dict(),
                os.path.join(LOG_DIR, f"fca.pt"),
            )
            if avg_loss < best_loss and avg_match > best_match:
                best_loss = avg_loss
                best_match = avg_match
                torch.save(
                    fca.state_dict(),
                    os.path.join(LOG_DIR, "fca_best.pt")
                )

    torch.save(
        fca.state_dict(),
        os.path.join(LOG_DIR, "fca_final.pt")
    )
    hook.remove()

    # --------- Subspace Visualization ---------
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(match_history)), [fca.num_components for _ in match_history], marker='o')
    plt.title("Subspace Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "subspace_evolution.png"))
