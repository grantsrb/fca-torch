import os
import csv
import yaml
import math
from datetime import datetime
import numpy as np
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
    get_activations_hook, register_activation_hooks,load_model_config,
)
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

import pandas as pd

from toxicity_constants import TOXIC_TOKEN, NONTOXIC_TOKEN, PROMPT_TEMPLATE


# --------- Configuration ---------
ROOT_DIR = "/data2/grantsrb/fca_saves/" #os.getcwd()
MODEL_NAME = "openai-community/gpt2" #"Qwen/Qwen3-14B" #"distilbert/distilbert-base-uncased" #
BATCH_SIZE = 64
TARGET_LAYER_NAME = "transformer.h.5"
DATASET_NAME = 'anitamaxvim/jigsaw-toxic-comments' #"Johnesss/Jigsaw-Toxic-Comment-Classification"  # replace with 'jigsaw-toxic-comment-classification' if using a local version
RUN_ID = datetime.now().strftime("d%Y-%m-%d_t%H-%M-%S")
LABEL_KEYS = [
    "toxic", "severe_toxic", "obscene", "threat", "insult",
    "identity_hate"
]
available_devices = [i for i in range(torch.cuda.device_count())]
DEVICE = available_devices[0] if torch.cuda.is_available() else "cpu"

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
        "tokenizer_name": None,
        "batch_size": BATCH_SIZE,
        "pca_sample_size": 10000,
        "pca_batch_size": None,
        "target_layer": TARGET_LAYER_NAME,
        "dataset": DATASET_NAME,
        "debugging": False,
        "small_data": False, # used for debugging purposes
        "use_model_labels": False,  # Use model predictions as labels

        "initial_rank": 300,
        "components_per_incr": 10, # how many components to increase the
            # FCA by with every performance plateau
        "max_components": math.inf,  # Maximum number of components to consider
    }
    config.update(get_command_line_args())

    # --------- Logging and Config Setup ---------
    ROOT_DIR = config["root_dir"]
    MODEL_NAME = config["model_name"]
    TOKENIZER_NAME = config.get("tokenizer_name", None)
    if os.path.isdir(MODEL_NAME) or os.path.exists(MODEL_NAME):
        mconfig = load_model_config(MODEL_NAME)
        TOKENIZER_NAME = mconfig.get("tokenizer_name", mconfig["model_name"])
    if TOKENIZER_NAME is None:
        TOKENIZER_NAME = MODEL_NAME

    dir_model_name = MODEL_NAME
    if ROOT_DIR in dir_model_name:
        dir_model_name = dir_model_name.split(ROOT_DIR)[-1]
    dir_model_name = dir_model_name.replace("/", "-")
    BASE_DIR = os.path.join(ROOT_DIR, f"fpca_{dir_model_name}")
    config["base_dir"] = BASE_DIR
    LOG_DIR = os.path.join(BASE_DIR, f"run_{RUN_ID}")
    config["log_dir"] = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    config["small_data"] = config.get("small_data", False) or config.get("debugging", False)

    # --------- Dataset ---------

    if config["small_data"]:
        dataset = load_dataset(config["dataset"], split="train[:1000]")
        val_dataset = load_dataset(config["dataset"], split="test[:1000]")
    else:
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

    val_toxic_count = sum(
        sum(item[key] for key in LABEL_KEYS) > 0 for item in val_dataset
    )
    val_nontoxic_count = len(val_dataset) - val_toxic_count
    print(f"Valid set balanced: {val_toxic_count} toxic, {val_nontoxic_count} nontoxic, total {len(val_dataset)}")


    # --------- Tokenizer ---------

    print("Model Name:", MODEL_NAME, "-- Tokenizer Name:", TOKENIZER_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    toxic_id = tokenizer.encode(TOXIC_TOKEN, add_special_tokens=False)[0]
    nontoxic_id = tokenizer.encode(NONTOXIC_TOKEN, add_special_tokens=False)[0]
    config["tokenizer_info"] = {
        "toxic_id": toxic_id,
        "nontoxic_id": nontoxic_id,
    }

    # --------- Data Loaders ---------

    dataset = dataset.map(
        lambda x:
            {"seq_len": tokenizer(
                x["text"],
                padding=True,
                truncation=True,
                return_tensors="pt")["input_ids"].shape[-1]}
    )
    val_dataset = val_dataset.map(
        lambda x:
            {"seq_len": tokenizer(
                x["text"],
                padding=True,
                truncation=True,
                return_tensors="pt")["input_ids"].shape[-1]}
    )
    pad_len = max(
        np.max(val_dataset["seq_len"]),
        np.max(dataset["seq_len"])
    )
    print("Pad Len:", pad_len)

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
        inputs = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=pad_len,
        )
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
        shuffle=False,
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
    dset_name = config["dataset"].replace("/", "-")
    actvs_path = os.path.join(BASE_DIR, f"{dset_name}_trn_actvs.pt")
    if os.path.exists(actvs_path):
        pt = torch.load(actvs_path)
        X = pt["X"]
        mask = pt["attention_mask"]
    else:
        with torch.no_grad():
            trn_outputs = collect_activations_using_loader(
                model=model,
                data_loader=dataloader,
                layers=[config["target_layer"]],
                to_cpu=True,
                verbose=True,
            )
            mask = torch.cat(
                [batch["attention_mask"] for batch in dataloader], axis=0
            ).bool()
            X = trn_outputs[config["target_layer"]]
            if not config.get("debugging", False):
                torch.save({"X": X, "attention_mask": mask}, actvs_path)
    X = X.reshape(-1, X.shape[-1])
    X = X[mask.reshape(-1).bool()]
    print("Initial Model Representation Shape:", X.shape) 

    print("Collecting initial validation performance")
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
            print(f"Batch {i+1}/{len(val_dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}", end=" "*20+"\r")
            if config.get("debugging", False) and i >= 5:
                break

        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision / max(1,total_precision)
        avg_recall = recall / max(1,total_recall)

    print()
    print(f"Initial Model Performance: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
    print(f"\tPrecision={avg_precision:.4f}, Recall={avg_recall:.4f}")
    print("Toxicity Proportion:", toxic_proportion/len(dataloader.dataset))
    config["initial_loss"] = avg_loss
    config["initial_match"] = avg_match
    config["initial_precision"] = avg_precision
    config["initial_recall"] = avg_recall
    model_labels = torch.cat(model_labels, dim=0)

    loss_floor = avg_loss
    match_ceiling = max(avg_match, 0.001)
    precision_ceiling = max(avg_precision, 0.001)
    recall_ceiling = max(avg_recall, 0.001)

    torch.cuda.empty_cache()
        
    config_path = os.path.join(LOG_DIR, "config.json")
    save_json(config, config_path)

    # --------- PCA Functional Component Analysis (FCA) Setup ---------

    print("Building Functional PCA")
    fpca = PCAFunctionalComponentAnalysis(
        X=X,
        center=True, scale=True,
        batch_size=config.get("pca_batch_size", None),
        max_sample_size=config.get("pca_sample_size", 10000),
        max_rank=config["max_components"],
        init_rank=config["initial_rank"],
    )
    fpca.to(DEVICE)

    # Comms Dict is used to collect activations from the specified layer
    comms_dict = dict()
    handle = fpca.hook_model_layer(
        model=model,
        layer=config["target_layer"],
        comms_dict=comms_dict)

    if config.get("use_model_labels", False):
        criterion = wrapped_kl_divergence
    else:
        criterion = nn.CrossEntropyLoss()

    for p in fpca.parameters():
        p.requires_grad = False

    # --------- Training Loop ---------
    metrics = {
        "rank": [],
        "layer": [],
        "trn_expl_var": [],
        "val_expl_var": [],
        "loss": [],
        "recovered_loss": [],
        "match": [],
        "recovered_match": [],
        "precision": [],
        "recovered_precision": [],
        "recall": [],
        "recovered_recall": [],
    }

    max_rank = config["max_components"]
    init_rank = config["initial_rank"]
    loss_history = []
    match_history = []
    best_loss = float('inf')
    best_match = 0.0
    layer = config["target_layer"]

    with torch.no_grad():
        for rank in range(init_rank, max_rank+1, config["components_per_incr"]):
            print(f"Rank: {rank}/{max_rank}")
            fpca.set_max_rank(rank)
            print(f"Layer: {layer}")

            samp_size = 1000
            samp_idx = torch.randperm(X.shape[0])[:samp_size].long()
            fpca.cpu()
            expl_var = fpca.proportion_expl_var(
                rank=rank,
                actvs=X[samp_idx].cpu(),
            ).mean().item()

            print(f"Rank {rank} -- Train Expl Var: {expl_var}")
            fpca.to(DEVICE)

            actvs = []
            total_loss, total_match = 0.0, 0.0
            total_recall, total_precision = 0.0, 0.0
            recall, precision = 0, 0
            for bi,batch in enumerate(val_dataloader):
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

                logits = model(**model_kwargs).logits
                actvs.append(comms_dict[fpca].cpu()) # stores the activations post
                    # processing by the fca module so we can evaluate the explained
                    # variance of the representations later

                #if config.get("debugging", False):
                #    print("Pre Logit Preparation:")
                #    print("Logits:", logits[0, batch["ans_idx"][0], :].cpu().detach().numpy())
                #    print("Targets:", targets[0], "Labels:", labels[0])
                #    print("Input IDs:", batch["input_ids"][0][:300])
                #    print("Answer Index:", batch["ans_idx"][0].item())
                #    print("Tox:", logits[0, batch["ans_idx"][0], toxic_id].item(),
                #          "Nontox:", logits[0, batch["ans_idx"][0], nontoxic_id].item())

                logits = prep_logits( logits, batch["ans_idx"] )
                loss = criterion(logits, targets)

                #if config.get("debugging", False):
                #    print("Post Logit Preparation:")
                #    print("Logits:", logits[0, :].cpu().detach().numpy())
                #    print("Targets:", targets[0], "Labels:", labels[0])
                #    print("Loss:", loss.item())

                total_loss += loss.item()

                match = (logits.argmax(-1) == labels)
                acc = match.float().mean().item()
                total_match += acc

                precision +=  match[logits.argmax(-1)==1].float().sum().item()
                total_precision += (logits.argmax(-1)==1).float().sum().item()
                recall +=  match[labels==1].float().sum().item()
                total_recall += (labels==1).float().sum().item()

                loss = loss

                print(f"Batch {bi+1}/{len(val_dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                      f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}", end=" "*20+"\r")
                
                if config.get("small_data", False) and bi >= 5:
                    break

            avg_loss =      total_loss / len(val_dataloader)
            avg_match =     total_match / len(val_dataloader)
            avg_precision = 0 if total_precision<=0 else precision / total_precision
            avg_recall =    0 if total_recall<=0 else recall / total_recall

            recovered_loss = (avg_loss-loss_floor) / loss_floor
            recovered_match = avg_match / match_ceiling
            recovered_precision = avg_precision / precision_ceiling
            recovered_recall = avg_recall / recall_ceiling

            print(f"Rank {rank}: Loss={avg_loss:.4f}, Acc={avg_match:.4f}, "
                  f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Components={fpca.n_components}")

            with torch.no_grad():
                trn_expl_var = fpca.proportion_expl_var().mean().item()
                fpca.cpu()
                actvs = torch.vstack(actvs)
                samp_idx = torch.randperm(len(actvs)).long()
                actvs = actvs[samp_idx[:samp_size]]
                val_expl_var = fpca.proportion_expl_var(
                    rank=rank, actvs=actvs,
                ).mean().item()
                print(f"Val Expl Var: {val_expl_var:.4f}")

            metrics["trn_expl_var"].append(trn_expl_var)
            metrics["val_expl_var"].append(val_expl_var)

            metrics["loss"].append(avg_loss)
            metrics["match"].append(avg_match)
            metrics["precision"].append(avg_precision)
            metrics["recall"].append(avg_recall)

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
            if not config.get("debugging", False):
                save_folder = config.get("save_folder", "results")
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                save_name = os.path.join(
                    save_folder,
                    config.get("save_name", "resnet_fca_metrics.csv"),
                )
                print(f"Saving metrics to {save_name}")
                df.to_csv(save_name, index=False, header=True)
                print("Metrics saved to {save_name}")

            print()
        
    if not config.get("debugging", False):
        torch.save(
            fpca.state_dict(),
            os.path.join(LOG_DIR, "fca_last.pt")
        )
    else:
        df = pd.DataFrame(metrics)
        print(df.head())
    handle.remove()
