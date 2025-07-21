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

from fca.fca import PCAFunctionalComponentAnalysis 
from fca.utils import (
    collect_activations_using_loader, get_command_line_args,
    get_command_line_args, get_output_size, save_json, arglast,
    get_activations_hook, register_activation_hooks
)
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

import pandas as pd

from toxicity_constants import TOXIC_TOKEN, NONTOXIC_TOKEN, PROMPT_TEMPLATE


# --------- Configuration ---------
ROOT_DIR = "/data2/grantsrb/fca_saves/" #os.getcwd()
MODEL_NAME = "openai-community/gpt2" #"Qwen/Qwen3-14B" #"distilbert/distilbert-base-uncased" #
BATCH_SIZE = 8
TARGET_LAYER_NAME = "transformer.h.5"
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
        "debugging": False,
        "use_model_labels": False,  # Use model predictions as labels
        "max_components": math.inf,  # Maximum number of components to learn
        "overwrite": False,
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

    # --------- Dataset ---------

    dataset = load_dataset(config["dataset"], split="train")
    val_dataset = load_dataset(config["dataset"], split="test")
    print("Dataset loaded:", config["dataset"])
    dataset = dataset.rename_column("comment_text", "text")
    val_dataset = val_dataset.rename_column("comment_text", "text")

    # Filter dataset to have balanced classes
    print("Starting dataset size:", len(dataset))
    dataset = balance_dataset(dataset, seed=config["seed"])
    toxic_count = sum(
        sum(item[key] for key in LABEL_KEYS) > 0 for item in dataset
    )
    nontoxic_count = len(dataset) - toxic_count
    assert toxic_count == nontoxic_count, \
        f"Dataset is not balanced: {toxic_count} toxic, {nontoxic_count} nontoxic"
    print(f"Train set balanced: {toxic_count} toxic, {nontoxic_count} nontoxic, total {len(dataset)}")


    # --------- Tokenizer ---------

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

    # --------- Data Loaders ---------

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

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --------- Model Setup ---------

    data_sample = {k: v.to(DEVICE) for k,v in next(iter(dataloader)).items()}
    ans_idx = data_sample["ans_idx"]
    inpt_ids = data_sample["input_ids"]
    
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    print(model)

    # --------- Initial Model Representations and Performance ---------

    # Collect train representations from the specified layers
    print("Collecting Train Activations")
    with torch.no_grad():
        trn_outputs = collect_activations_using_loader(
            model=model,
            data_loader=dataloader,
            layers=[config["target_layer"]],
            to_cpu=True,
            verbose=True,
        )
    X = trn_outputs[config["target_layer"]]
    print("Initial Model Representation Shape:", X.shape) 

    criterion = nn.CrossEntropyLoss()
    toxic_proportion = 0
    model_labels = []
    with torch.no_grad():
        total_loss, total_match, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0
        recall = 0.0
        precision = 0.0
        for i,batch in enumerate(val_dataloader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            model_kwargs = {"input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"]}
            logits = model(**model_kwargs).logits
            logits = prep_logits( logits, inputs["ans_idx"] )
            model_labels.append(logits.detach().cpu())

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

        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision / max(1,total_precision)
        avg_recall = recall / max(1,total_recall)

    print(f"Initial Model Performance: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
    print(f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}")
    print("Label Histogram:", toxic_proportion/len(dataloader.dataset))
    config["initial_loss"] = avg_loss
    config["initial_match"] = avg_match
    config["initial_precision"] = avg_precision
    config["initial_recall"] = avg_recall
    model_labels = torch.cat(model_labels, dim=0)

    loss_floor = avg_loss
    match_ceiling = avg_match
    precision_ceiling = avg_precision
    recall_ceiling = avg_recall

    torch.cuda.empty_cache()
        
    config_path = os.path.join(LOG_DIR, "config.json")
    save_json(config, config_path)

    # --------- PCA Functional Component Analysis (FCA) Setup ---------

    pc_fca = PCAFunctionalComponentAnalysis(
        X=X, center=True, scale=True,
    )
    pc_fca.to(DEVICE)

    comms_dict = dict()
    handle = pc_fca.hook_model_layer(
        model=model,
        layer=config["target_layer"],
        comms_dict=comms_dict)

    if config.get("use_model_labels", False):
        criterion = wrapped_kl_divergence
    else:
        criterion = nn.CrossEntropyLoss()

    # --------- Training Loop ---------
    metrics = {
        "rank": [],
        "layer": [],
        "all_layers": [],
        "trn_expl_var": [],
        "val_expl_var": [],
        "accuracy": [],
        "recovered_accuracy": [],
        "precision": [],
        "recovered_precision": [],
        "recall": [],
        "recovered_recall": [],
    }

    max_rank = X.shape[-1]
    loss_history = []
    match_history = []
    best_loss = float('inf')
    best_match = 0.0
    layer = config["target_layer"]

    for rank in range(1, max_rank+1):
        print(f"Rank: {rank}/{max_rank}")
        pc_fca.set_max_rank(rank)
        print(f"Layer: {layer}")
        samp = torch.randperm(trn_outputs[layer].shape[0])[:1000].long()
        expl_var = pc_fca.proportion_expl_var(
            rank=rank,
            actvs=trn_outputs[layer][samp].to(DEVICE),
        ).mean().item()
        print(f"Explained Variance: {expl_var}")

        actvs = []
        total_loss, total_match = 0.0, 0.0
        total_recall, total_precision = 0.0, 0.0
        recall, precision = 0, 0
        for bi,batch in enumerate(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            model_kwargs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }

            # Collect model outputs as labels
            if config.get("use_model_labels", False):
                targets = model_labels.to(DEVICE)
                labels = targets.argmax(dim=-1)
            else:
                targets = batch["targets"].long()
                labels = targets

            with torch.no_grad():
                logits = model(**model_kwargs).logits
            actvs.append(comms_dict[pc_fca])

            if config.get("debugging", False):
                print("Pre Logit Preparation:")
                print("Logits:", logits[0, batch["ans_idx"][0], :].cpu().detach().numpy())
                print("Targets:", targets[0], "Labels:", labels[0])
                print("Input IDs:", batch["input_ids"][0][:300])
                print("Answer Index:", batch["ans_idx"][0].item())
                print("Tox:", logits[0, batch["ans_idx"][0], toxic_id].item(),
                      "Nontox:", logits[0, batch["ans_idx"][0], nontoxic_id].item())

            logits = prep_logits( logits, batch["ans_idx"] )
            loss = criterion(logits, targets)

            if config.get("debugging", False):
                print("Post Logit Preparation:")
                print("Logits:", logits[0, :].cpu().detach().numpy())
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

            loss = loss

            print(f"Batch {bi+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}", end=" "*20+"\r")
            
            if config.get("debugging", False) and bi >= 5:
                break

        avg_loss =      total_loss / len(dataloader)
        avg_match =     total_match / len(dataloader)
        avg_precision = precision / total_precision
        avg_recall =    recall / total_recall

        recovered_loss = (avg_loss-loss_floor) / loss_floor
        recovered_match = avg_match / match_ceiling
        recovered_precision = avg_precision / precision_ceiling
        recovered_recall = avg_recall / recall_ceiling

        metrics["loss"].append(avg_loss)
        metrics["acc"].append(avg_match)

        print(f"Rank {rank}: Loss={avg_loss:.4f}, Acc={avg_match:.4f}, "
              f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Components={pc_fca.n_components}")

        with torch.no_grad():
            trn_expl_var = pc_fca.proportion_expl_var().mean().item()
            pc_fca.cpu()
            val_expl_var = pc_fca.proportion_expl_var(
                rank=rank,
                actvs=torch.vstack(actvs[pc_fca]),
            ).mean().item()

        metrics["trn_expl_var"].append(trn_expl_var)
        metrics["val_expl_var"].append(val_expl_var)

        metrics["loss"].append(loss)
        metrics["match"].append(match)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)

        metrics["recovered_loss"].append(recovered_loss)
        metrics["recovered_match"].append(recovered_match)
        metrics["recovered_precision"].append(recovered_precision)
        metrics["recovered_recall"].append(recovered_recall)

        metrics["rank"].append(rank)
        metrics["layer"].append(layer)

        print(f"Layer: {layer}, Rank: {rank}, "
              f"Train Expl Var: {trn_expl_var:.4f}, "
              f"Val Expl Var: {val_expl_var:.4f}, "
              f"Accuracy: {avg_match:.4f}, "
              f"Recovered Acc: {recovered_match:.4f}\n"
              f"Precis: {avg_precision:.4f}, "
              f"Recovered Prec: {recovered_precision:.4f}, "
              f"Recall: {avg_recall:.4f}, "
              f"Recovered Recl: {recovered_recall:.4f}, "
        )

    df = pd.DataFrame(metrics)
    if config.get("debugging", False):
        print(df.head())
    else:
        save_folder = config.get("save_folder", "results")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_name = os.path.join(
            save_folder,
            config.get("save_name", "resnet_fca_metrics.csv"),
        )
        if os.path.exists(save_name) and not config["overwrite"]:
            prev_df = pd.read_csv(save_name)
            df = pd.concat([prev_df, df], ignore_index=True)
        else:
            print(f"Saving metrics to {save_name}")
        df.to_csv(save_name, index=False, header=True)
        print("Metrics saved to {save_name}")


    if not config.get("debugging", False):
        torch.save(
            pc_fca.state_dict(),
            os.path.join(LOG_DIR, "fca_last.pt")
        )
    handle.remove()