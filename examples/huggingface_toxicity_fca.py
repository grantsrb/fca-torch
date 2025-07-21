import os
import csv
import yaml
import math
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from datasets import load_dataset, concatenate_datasets

from fca.fca import FunctionalComponentAnalysis, load_ortho_fcas  # Assuming you have a custom FCA module
from fca.utils import get_command_line_args, get_output_size, save_json, arglast
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

from toxicity_constants import TOXIC_TOKEN, NONTOXIC_TOKEN, PROMPT_TEMPLATE

# --------- Configuration ---------
ROOT_DIR = "/data2/grantsrb/fca_saves/" #os.getcwd()
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" #"Qwen/Qwen3-14B" #"openai-community/gpt2" #"distilbert/distilbert-base-uncased" #
BATCH_SIZE = 16
TARGET_LAYER_NAME = "model.layers.15"
TOLERANCE = 0.01
PATIENCE = 3
LEARNING_RATE = 1e-4
LOG_EVERY = 2
MAX_EPOCHS = 100
DATASET_NAME = 'anitamaxvim/jigsaw-toxic-comments' #"Johnesss/Jigsaw-Toxic-Comment-Classification"  # replace with 'jigsaw-toxic-comment-classification' if using a local version
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
LABEL_KEYS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult",
    "identity_hate"
]
available_devices = [i for i in range(torch.cuda.device_count())]
DEVICE = available_devices[0] if torch.cuda.is_available() else "cpu"
DEVICE2 = available_devices[-1] if torch.cuda.is_available() else "cpu"

def balance_dataset(dataset, seed=42):
    # Count the distribution of labels
    toxic_count = sum(
        sum(item[key] for key in LABEL_KEYS) > 0 for item in dataset
    )
    nontoxic_count = len(dataset) - toxic_count
    if toxic_count > nontoxic_count:
        nontoxic = dataset.filter(
            lambda item: sum(item[key] for key in LABEL_KEYS) == 0
        )
        toxic = dataset.filter(
            lambda item: sum(item[key] for key in LABEL_KEYS) > 0
        ).shuffle(seed=seed).select(range(nontoxic_count))
        dataset = concatenate_datasets([ toxic, nontoxic ])
    elif nontoxic_count > toxic_count:
        nontoxic = dataset.filter(
            lambda item: sum(item[key] for key in LABEL_KEYS) == 0
        ).shuffle(seed=seed).select(range(toxic_count))
        toxic = dataset.filter(
            lambda item: sum(item[key] for key in LABEL_KEYS) > 0
        )
        dataset = concatenate_datasets([ toxic, nontoxic ])
    return dataset.shuffle(seed=seed)

if __name__ == "__main__":
    print("Running Hugging Face Toxicity Example...")
    config = {
        "root_dir": ROOT_DIR,
        "seed": 42,  # Random seed for reproducibility, also the meaning of life, the universe, and everything
        "model_name": MODEL_NAME,
        "batch_size": BATCH_SIZE,
        "target_layer": TARGET_LAYER_NAME,
        "tolerance": TOLERANCE,
        "patience": PATIENCE,
        "lr": LEARNING_RATE,
        "dataset": DATASET_NAME,
        "ortho_fcas": None,  # Path to orthogonal FCA components if any
        "debugging": False,
        "reduce_init_eval_data": True, # if true, will reduce the amount
            # of data used to initially evaluate the model
        "use_model_labels": False,  # Use model predictions as labels
        "compl_eps": 0.0,  # Weight for complement loss, set to 0.0 to disable
        "max_components": math.inf,  # Maximum number of components to learn
    }
    config.update(get_command_line_args())

    # --------- Logging Setup ---------
    ROOT_DIR = config["root_dir"]
    MODEL_NAME = config["model_name"]
    LOG_DIR = f"{ROOT_DIR}/fca_{MODEL_NAME}/run_{RUN_ID}"
    config["log_dir"] = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    CSV_PATH = os.path.join(LOG_DIR, "results.csv")
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "avg_loss", "behavior_match", "n_components"])

    # --------- Dataset & Tokenizer ---------
    dataset = load_dataset(config["dataset"], split="train")
    print("Dataset loaded:", config["dataset"])
    dataset = dataset.rename_column("comment_text", "text")

    # Filter dataset to have balanced classes
    print("Starting dataset size:", len(dataset))
    dataset = balance_dataset(dataset, seed=config["seed"])
    toxic_count = sum(
        sum(item[key] for key in LABEL_KEYS) > 0 for item in dataset
    )
    nontoxic_count = len(dataset) - toxic_count
    assert toxic_count == nontoxic_count, \
        f"Dataset is not balanced: {toxic_count} toxic, {nontoxic_count} nontoxic"
    print(f"Dataset balanced: {toxic_count} toxic, {nontoxic_count} nontoxic, total {len(dataset)}")

    print("Model Name:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    toxic_id = tokenizer.encode(TOXIC_TOKEN, add_special_tokens=False)[0]
    nontoxic_id = tokenizer.encode(NONTOXIC_TOKEN, add_special_tokens=False)[0]
    config["tokenizer_info"] = {
        "toxic_id": toxic_id,
        "nontoxic_id": nontoxic_id,
    }

    def collate_fn(batch):
        labels = [
            sum([item[k] for k in LABEL_KEYS])>0 for item in batch
        ]
        labels = torch.tensor(labels).bool()
        texts = [
            PROMPT_TEMPLATE.format(
                comment=item["text"],
                label=TOXIC_TOKEN if item["toxic"] else NONTOXIC_TOKEN
            )
            for item in batch
        ]
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        ans_idx = (inputs["input_ids"]==toxic_id)
        ans_idx = ans_idx&(labels==True)[:,None]
        ans_idx1 = (labels==False)[:,None]&(inputs["input_ids"]==nontoxic_id)
        ans_idx = ans_idx | ans_idx1
        ans_idx = arglast(ans_idx.long(), dim=-1)-1 # subtract 1 to get predicted token index
        return {
            "ans_idx": ans_idx,
            "targets": labels,
            **inputs,
        }

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )

    # --------- Model Setup ---------

    data_sample = {k: v.to(DEVICE) for k,v in next(iter(dataloader)).items()}
    ans_idx = data_sample["ans_idx"]
    inpt_ids = data_sample["input_ids"]
    # print(" STOP", tokenizer.encode(" STOP", add_special_tokens=False)[0])
    # print(TOXIC_TOKEN, tokenizer.encode(TOXIC_TOKEN, add_special_tokens=False)[0])
    # print(NONTOXIC_TOKEN, tokenizer.encode(NONTOXIC_TOKEN, add_special_tokens=False)[0])
    # print("data_sample:", data_sample["input_ids"][0][:300])
    # print("Answer Index:", data_sample["input_ids"][0][ans_idx[0]])
    # print(data_sample["input_ids"][0], "tox", toxic_id, "nont", nontoxic_id)
    for i in range(len(inpt_ids)):
        assert inpt_ids[i,ans_idx[i]+1] == toxic_id or inpt_ids[i,ans_idx[i]+1] == nontoxic_id, \
        "Input IDs do not match expected toxic or nontoxic tokens."

    def prep_logits(logits, ans_idx):
        """
        Prepares logits for binary classification.
        """
        row_idx = torch.arange(logits.shape[0], device=logits.device).long()
        logits = torch.cat([
            logits[row_idx, ans_idx, toxic_id].unsqueeze(-1),
            logits[row_idx, ans_idx, nontoxic_id].unsqueeze(-1)
        ], dim=-1)
        return logits

    if config.get("use_model_labels", False):
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
        )
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        #base_model.to(DEVICE2)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    #model.to(DEVICE)
    print(model)

    # --------- Initial Model Performance ---------

    criterion = nn.CrossEntropyLoss()
    toxic_proportion = 0
    with torch.no_grad():
        total_loss, total_match, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0
        recall = 0.0
        precision = 0.0
        for i,batch in enumerate(dataloader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            model_kwargs = {"input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"]}
            logits = model(**model_kwargs).logits
            logits = prep_logits( logits, inputs["ans_idx"] )

            labels = inputs["targets"].long()
            loss = criterion(logits, labels)
            total_loss += loss.item()
            match = (logits.argmax(-1) == labels)
            acc = match.float().mean().item()
            total_match += acc

            labels = inputs["targets"].long()
            precision +=  match[logits.argmax(-1)==1].float().sum().item()
            total_precision += (logits.argmax(-1)==1).float().sum().item()
            recall +=  match[labels==1].float().sum().item()
            total_recall += (labels==1).float().sum().item()

            toxic_proportion += inputs["targets"].long().sum().item()
            print(f"Batch {i+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}", end=" "*20+"\r")
            if config.get("debugging", False) and i >= 10:
                break
            if config.get("reduce_init_eval_data", False) and i >= 30:
                break
        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision / max(1,total_precision)
        avg_recall = recall / max(1,total_recall)
    print(f"Initial Model Performance: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
    print(f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")
    print("Label Histogram:", toxic_proportion/len(dataloader.dataset))
    config["initial_loss"] = avg_loss
    config["initial_match"] = avg_match

    torch.cuda.empty_cache()
        
    config_path = os.path.join(LOG_DIR, "config.json")
    save_json(config, config_path)

    # --------- Functional Component Analysis (FCA) Setup ---------

    size = get_output_size(
        model=model,
        layer_name=TARGET_LAYER_NAME,
        data_sample=data_sample,
    )
    config["fca_params"] = {
        "size": size,
        "max_components": min(size, config["max_components"]),
    }

    fca = FunctionalComponentAnalysis(**config["fca_params"])
    if "ortho_fcas" in config and config["ortho_fcas"]:
        fca = load_ortho_fcas(fca, config["ortho_fcas"])
    fca.to(DEVICE)

    if next(fca.parameters()).dtype != model.dtype:
        print("Warning: FCA parameters dtype does not match model dtype. "
              "Converting FCA parameters to model dtype.")
        fca = fca.to(model.dtype)

    comms_dict = dict()
    hook = fca.hook_model_layer(
        model=model,
        layer=TARGET_LAYER_NAME,
        comms_dict=comms_dict)

    if config.get("use_model_labels", False):
        criterion = wrapped_kl_divergence
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(fca.parameters(), lr=config["lr"])

    # reduces number of computations by caching the weight
    fca.set_cached(True)
    fca.reset_cached_weight()

    # --------- Training Loop ---------
    loss_history = []
    match_history = []
    best_loss = float('inf')
    best_match = 0.0

    for epoch in range(MAX_EPOCHS):
        print("Beginning epoch", epoch + 1)
        total_loss, total_match = 0.0, 0.0
        total_recall, total_precision = 0.0, 0.0
        recall, precision = 0, 0
        for bi,batch in enumerate(dataloader):
            fca.use_complement_in_hook = False
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            
            # Collect model outputs as labels
            if config.get("use_model_labels", False):
                with torch.no_grad():
                    base_output = base_model(**{
                        k: v.to(DEVICE) for k,v in model_kwargs.items()
                    }).logits
                    targets = prep_logits(
                        base_output, batch["ans_idx"].to(DEVICE2)
                    ).to(DEVICE)
                    labels = targets.argmax(dim=-1)
                torch.cuda.empty_cache()
            else:
                targets = batch["targets"].long()
                labels = targets

            logits = model(**model_kwargs).logits
            if config.get("debugging", False):
                print("Pre Logit Preparation:")
                print("Logits:", logits[0, batch["ans_idx"][0], :].float().cpu().detach().numpy())
                print("Targets:", targets[0], "Labels:", labels[0])
                print("Input IDs:", batch["input_ids"][0][:300])
                print("Answer Index:", batch["ans_idx"][0].item())
                print("Tox:", logits[0, batch["ans_idx"][0], toxic_id].item(),
                      "Nontox:", logits[0, batch["ans_idx"][0], nontoxic_id].item())
            logits = prep_logits( logits, batch["ans_idx"] )

            loss = criterion(logits, targets)

            if config.get("debugging", False):
                print("Post Logit Preparation:")
                print("Logits:", logits[0, :].float().cpu().detach().numpy())
                print("Targets:", targets[0], "Labels:", labels[0])
                print("Loss:", loss.item())

            total_loss += loss.item()

            match = (logits.argmax(-1) == labels)
            acc = match.float().mean().item()
            total_match += acc

            precision +=  match[logits.argmax(-1)==1].float().sum().item()
            total_precision += (logits.argmax(-1)==1).float().sum().item()
            recall +=  match[labels==1].float().sum().item()
            total_recall += (labels==1).float().sum().item()

            # Track the complement loss
            compl_loss = torch.tensor(0.0, device=logits.device)
            if config.get("compl_eps", 0) > 0:
                fca.use_complement_in_hook = True
                compl_logits = prep_logits(
                    model(**model_kwargs).logits,
                    ans_idx=batch["ans_idx"]
                )
                fca.use_complement_in_hook = False
                compl_labels = 0.5*torch.ones_like(compl_logits)
                compl_loss = wrapped_kl_divergence(
                    compl_logits, compl_labels, preds_are_logits=True)

            eps = config.get("compl_eps", 0)
            loss = eps*compl_loss + (1-eps)*loss
            try:
                loss.backward()
            except RuntimeError as e:
                print("Failed to backpropagate due to:", e)
            try:
                optimizer.step()
            except RuntimeError as e:
                print("Failed to optimize due to:", e)
            optimizer.zero_grad()
            fca.reset_cached_weight()

            print(f"Batch {bi+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}", end=" "*20+"\r")
            
            if config.get("debugging", False) and bi >= 5:
                break

        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision / total_precision
        avg_recall = recall / total_recall
        loss_history.append(avg_loss)
        match_history.append(avg_match)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_match:.4f}, "
              f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Components={fca.n_components}")

        with open(CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch, avg_loss, avg_match, fca.n_components
            ])

        if len(loss_history) > config["patience"]:
            if max(loss_history[-config["patience"]:]) -\
                    min(loss_history[-config["patience"]:]) < 1e-4:
                if fca.n_components >= fca.max_components:
                    print("Stopping early due to convergence.")
                    break
                fca.add_component()

        if avg_match >= (1 - config["tolerance"]):
            print("✅ Stopping: Match >= 99%")
            break

        if epoch % LOG_EVERY == 0:
            if not config.get("debugging", False):
                torch.save(
                    fca.state_dict(),
                    os.path.join(LOG_DIR, f"fca.pt"),
                )
            if avg_loss < best_loss and avg_match > best_match:
                best_loss = avg_loss
                best_match = avg_match
                if not config.get("debugging", False):
                    torch.save(
                        fca.state_dict(),
                        os.path.join(LOG_DIR, "fca_best.pt")
                    )

    if not config.get("debugging", False):
        torch.save(
            fca.state_dict(),
            os.path.join(LOG_DIR, "fca_last.pt")
        )
    hook.remove()

    # --------- Subspace Visualization ---------
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(match_history)), [fca.n_components for _ in match_history], marker='o')
    plt.title("Subspace Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Components")
    plt.grid(True)
    plt.tight_layout()
    if not config.get("debugging", False):
        plt.savefig(os.path.join(LOG_DIR, "subspace_evolution.png"))
