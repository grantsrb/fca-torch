import os
import csv
import yaml
import time
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
from fca.utils import (
    get_command_line_args, get_output_size, save_json, arglast,
    load_model_config,
)
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

from toxicity_constants import TOXIC_TOKEN, NONTOXIC_TOKEN, PROMPT_TEMPLATE

import pandas as pd

# --------- Configuration ---------
ROOT_DIR = "/data2/grantsrb/fca_saves/" #os.getcwd()
MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" #"Qwen/Qwen3-14B" #"openai-community/gpt2" #"distilbert/distilbert-base-uncased" #
BATCH_SIZE = 16
TARGET_LAYER_NAME = "transformer.h.5" # "model.layers.15"
TOLERANCE = 0.01
PATIENCE = 3
LEARNING_RATE = 1e-4
LOG_EVERY = 2
MAX_EPOCHS = 100
DATASET_NAME = 'anitamaxvim/jigsaw-toxic-comments' #"Johnesss/Jigsaw-Toxic-Comment-Classification"  # replace with 'jigsaw-toxic-comment-classification' if using a local version
RUN_ID = datetime.now().strftime("d%Y-%m-%d_t%H-%M-%S")
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
        (sum(item[key] for key in LABEL_KEYS) > 0) for item in dataset
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

def forward_pass(
    model,
    dataloader,
    config,
    criterion=nn.CrossEntropyLoss(),
    fca=None,
    optimizer=None,
    is_validation=False,
):
    prev_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(not is_validation)
    total_loss, total_match, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0
    toxic_count, total_count = 0, 0
    recall = 0.0
    precision = 0.0
    for i,batch in enumerate(dataloader):
        start_time = time.time()
        if fca is not None:
            fca.use_complement_in_hook = False
        inputs = {k: v.to(DEVICE) for k, v in batch.items()}
        model_kwargs = {"input_ids": inputs["input_ids"],
                        "attention_mask": inputs["attention_mask"]}
        logits = model(**model_kwargs).logits
        logits = prep_logits( logits, inputs["ans_idx"] )

        labels = inputs["targets"].long()
        loss = criterion(logits, labels)/config.get("grad_accumulation_steps", 1)

        # Track the complement loss
        compl_loss = torch.tensor(0.0, device=logits.device)
        eps = config.get("compl_eps", 0)
        if eps > 0 and fca is not None:
            fca.use_complement_in_hook = True
            compl_logits = prep_logits(
                model(**model_kwargs).logits,
                ans_idx=batch["ans_idx"]
            )
            fca.use_complement_in_hook = False
            compl_labels = 0.5*torch.ones_like(compl_logits)
            compl_loss = wrapped_kl_divergence(
                compl_logits, compl_labels, preds_are_logits=True)

        loss = eps*compl_loss + (1-eps)*loss

        if not is_validation:
            try:
                loss.backward()
            except RuntimeError as e:
                print("Failed to backpropagate due to:", e)
            if i % config.get("grad_accumulation_steps", 1) == 0:
                if optimizer is not None:
                    optimizer.step()
                    optimizer.zero_grad()
                if fca is not None:
                    fca.reset_cached_weight()

        total_loss += loss.item()
        match = (logits.argmax(-1) == labels)
        acc = match.float().mean().item()
        total_match += acc

        labels = inputs["targets"].long()
        precision +=  match[logits.argmax(-1)==1].float().sum().item()
        total_precision += (logits.argmax(-1)==1).float().sum().item()
        recall +=  match[labels==1].float().sum().item()
        total_recall += (labels==1).float().sum().item()

        toxic_count += inputs["targets"].long().sum().item()
        total_count += len(inputs["targets"])
        run_time = time.time()-start_time
        print(f"Batch {i+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
              f"Precision={precision/max(1,total_precision):.4f}, Recall={recall/max(1,total_recall):.4f}, "
              f"Exec Time: {run_time:.4f}",
              end=" "*20+"\r")
        if config.get("small_data", False) and i >= 10:
            break
    print()

    avg_loss = total_loss / len(dataloader)
    avg_match = total_match / len(dataloader)
    avg_precision = precision / max(1,total_precision)
    avg_recall = recall / max(1,total_recall)
    toxic_proportion = toxic_count/total_count
    torch.set_grad_enabled(prev_grad_state)

    return {
        "avg_loss": avg_loss,
        "avg_match":  avg_match,
        "avg_precision":  avg_precision,
        "avg_recall":  avg_recall,
        "toxic_proportion": toxic_proportion,
    }

if __name__ == "__main__":
    print("Running Hugging Face Toxicity Example...")
    config = {
        "root_dir": ROOT_DIR,
        "seed": 42,  # Random seed for reproducibility, also the meaning of life, the universe, and everything
        "model_name": MODEL_NAME,
        "tokenizer_name": None,
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

        "initial_rank": 300,
        "components_per_incr": 10, # how many components to increase the
            # FCA by with every performance plateau
        "max_components": 384, # how many components to increase the
            # FCA by with every performance plateau
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
            TOKENIZER_NAME = mconfig["model_name"]
    if TOKENIZER_NAME is None:
        TOKENIZER_NAME = MODEL_NAME

    dir_model_name = MODEL_NAME
    if ROOT_DIR in dir_model_name:
        dir_model_name = dir_model_name.split(ROOT_DIR)[-1]
    dir_model_name = dir_model_name.replace("/", "-")
    BASE_DIR = os.path.join(ROOT_DIR, f"fca_{dir_model_name}")
    config["base_dir"] = BASE_DIR
    LOG_DIR = os.path.join(BASE_DIR, f"run_{RUN_ID}")
    config["log_dir"] = LOG_DIR
    os.makedirs(LOG_DIR, exist_ok=True)

    CSV_PATH = os.path.join(LOG_DIR, "results.csv")
    with open(CSV_PATH, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["epoch", "avg_loss", "behavior_match", "n_components"])

    config["small_data"] = config.get("small_data", False) or config.get("debugging", False)


    # --------- Dataset & Tokenizer ---------

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

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype="auto",
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            os.path.join(MODEL_NAME, "model_checkpt"),
            device_map="auto",
            torch_dtype="auto",
        )
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    #model.to(DEVICE)
    print(model)

    # --------- Initial Model Performance ---------

    print("Collecting initial performance")
    criterion = nn.CrossEntropyLoss()
    toxic_proportion = 0
    with torch.no_grad():
        model.eval()
        print("Training Data...")
        outputs = forward_pass(
            model=model,
            dataloader=dataloader,
            config=config,
            criterion=criterion,
            is_validation=True,
        )
        avg_loss =      outputs["avg_loss"]
        avg_match =     outputs["avg_match"]
        avg_precision = outputs["avg_precision"]
        avg_recall =    outputs["avg_recall"]
        toxic_proportion = outputs["toxic_proportion"]

        print(f"Initial Train: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
        print(f"\tPrecision={avg_precision:.4f}, Recall={avg_recall:.4f}")
        print("Toxic Proportion:", toxic_proportion/len(dataloader.dataset))

        print("Validation Data...")
        outputs = forward_pass(
            model=model,
            dataloader=val_dataloader,
            config=config,
            criterion=criterion,
            is_validation=True,
        )
        val_avg_loss =      outputs["avg_loss"]
        val_avg_match =     outputs["avg_match"]
        val_avg_precision = outputs["avg_precision"]
        val_avg_recall =    outputs["avg_recall"]
        toxic_proportion = outputs["toxic_proportion"]

        print(f"Initial Valid: Loss={val_avg_loss:.4f}, Match={val_avg_match:.4f}")
        print(f"\tPrecision={val_avg_precision:.4f}, Recall={val_avg_recall:.4f}")
        print("Toxic Proportion:", toxic_proportion/len(val_dataloader.dataset))
    config["initial_loss"] = avg_loss
    config["initial_match"] = avg_match
    config["initial_val_loss"] = val_avg_loss
    config["initial_val_match"] = val_avg_match

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
        "init_rank": config["initial_rank"],
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
    metrics = {
        "epoch": [],
        "train_avg_loss": [],
        "train_avg_match": [],
        "train_avg_precision": [],
        "train_avg_recall": [],
        "valid_avg_loss": [],
        "valid_avg_match": [],
        "valid_avg_precision": [],
        "valid_avg_recall": [],
    }

    loss_history = []
    match_history = []
    best_loss = float('inf')
    best_match = 0.0

    for epoch in range(MAX_EPOCHS):
        print("Beginning epoch", epoch + 1, "-- Rank", fca.rank)

        fca.use_complement_in_hook = False
        model.train()
        outputs = forward_pass(
            model=model,
            fca=fca,
            dataloader=dataloader,
            optimizer=optimizer,
            config=config,
            criterion=criterion,
            is_validation=False,
        )
        avg_loss =      outputs["avg_loss"]
        avg_match =     outputs["avg_match"]
        avg_precision = outputs["avg_precision"]
        avg_recall =    outputs["avg_recall"]

        model.eval()
        with torch.no_grad():
            outputs = forward_pass(
                model=model,
                fca=fca,
                dataloader=dataloader,
                optimizer=optimizer,
                config=config,
                criterion=criterion,
                is_validation=True,
            )
        val_avg_loss =      outputs["avg_loss"]
        val_avg_match =     outputs["avg_match"]
        val_avg_precision = outputs["avg_precision"]
        val_avg_recall =    outputs["avg_recall"]

        metrics["epoch"].append(epoch)
        metrics["train_avg_loss"].append(avg_loss)
        metrics["train_avg_match"].append(avg_match)
        metrics["train_avg_precision"].append(avg_precision)
        metrics["train_avg_recall"].append(avg_recall)
        metrics["valid_avg_loss"].append(val_avg_loss)
        metrics["valid_avg_match"].append(val_avg_match)
        metrics["valid_avg_precision"].append(val_avg_precision)
        metrics["valid_avg_recall"].append(val_avg_recall)

        print(f"Epoch: {epoch+1} --", LOG_DIR)
        print(f"Train Performance: Loss={avg_loss:.4f}, Match={avg_match:.4f}")
        print(f"\tPrecision={avg_precision:.4f}, Recall={avg_recall:.4f}")
        print(f"Valid Performance: Loss={val_avg_loss:.4f}, Match={val_avg_match:.4f}")
        print(f"\tPrecision={val_avg_precision:.4f}, Recall={val_avg_recall:.4f}")

        if not config.get("debugging", False):
            df = pd.DataFrame(metrics)
            df.to_csv(CSV_PATH, header=True)

        if len(loss_history) > config["patience"]:
            if max(loss_history[-config["patience"]:]) -\
                    min(loss_history[-config["patience"]:]) < 1e-4:
                if fca.n_components >= fca.max_rank:
                    print("Stopping early due to convergence.")
                    break
                for _ in range(config.get("components_per_incr", 1)):
                    fca.add_component()

        high_acc = avg_match >= (1 - config["tolerance"]) and\
                   avg_precision >= (1 - config["tolerance"])
        if high_acc:
            print("✅ Stopping: Match: {avg_match} -- Prec: {avg_precision}")
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
        print()

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
