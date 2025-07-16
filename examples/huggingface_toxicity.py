import os
import csv
import yaml
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
from datasets import load_dataset

from fca.fca import FunctionalComponentAnalysis, load_ortho_fcas  # Assuming you have a custom FCA module
from fca.utils import get_command_line_args, get_output_size, save_json, arglast
from fca.wrappers import wrapped_kl_divergence  # Assuming you have a custom wrapper for KL divergence

TOXIC_TOKEN = " yes"
NONTOXIC_TOKEN = " no"
PROMPT_TEMPLATE = (
    "You have been tasked with analyzing the toxicity of comments on "
    "wikipedia articles. You are a nuanced thinker who values integrity, "
    "fairness, and the well-being of all individuals. Your goal is to "
    "identify toxic comments while minimizing false positives, ensuring "
    "that you do not label non-toxic comments as toxic. You will be "
    "evaluated based on your ability to accurately classify comments "
    "as toxic or non-toxic, with a focus on maintaining a high level of "
    "precision and recall. Your responses should reflect a deep understanding "
    "of the complexities of human language and the subtleties of online "
    "communication. You will be given one comment at a time, and your "
    "task is to classify each comment as toxic or nontoxic based on the "
    "guidelines provided. "
    "Here is an example of the evaluation format: \n\n"
    "Comment: \"I hate you, you are the worst person ever!\"\n"
    "Is this toxic: yes STOP\n\n"
    "Please only respond with either the word "
    "\"yes\" or \"no\" followed by the word STOP."
    "Do not provide any additional explanations "
    "or justifications for your classification, and do not place quotes arround your answers. Your responses should be "
    "just one word followed by STOP. "
    "You will be evaluated on your performance, so strive for excellence in "
    "your classifications. Good luck!\n\n\n"
    "Comment: \"{comment}\"\nIs this toxic:{label}"
)

# --------- Configuration ---------
ROOT_DIR = "/data2/grantsrb/fca_saves/" #os.getcwd()
MODEL_NAME = "openai-community/gpt2" #"distilbert/distilbert-base-uncased" #
BATCH_SIZE = 16
TARGET_LAYER_NAME = "transformer.h.5"
TOLERANCE = 0.01
PATIENCE = 3
LEARNING_RATE = 1e-3
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
        "debugging": False,
        "use_all_labels": True,  # Use all labels or binary toxicity
        "use_model_labels": True,  # Use model predictions as labels
        "train_fca_complement": False,  # Train FCA complement
        "compl_eps": 0.0,  # Weight for complement loss
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
        writer.writerow(["epoch", "avg_loss", "behavior_match", "num_components"])

    # --------- Dataset & Tokenizer ---------
    dataset = load_dataset(config["dataset"], split="train[:5000]")
    print("Model Name:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
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
                comment=item["comment_text"],
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
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        for param in base_model.parameters():
            param.requires_grad = False
        base_model.eval()
        base_model.to(DEVICE2)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(DEVICE)
    print(model)

    # --------- Initial Model Performance ---------
    criterion = nn.CrossEntropyLoss()
    toxic_proportion = 0
    with torch.no_grad():
        total_loss, total_match, total_precision, total_recall = 0.0, 0.0, 0.0, 0.0
        recall = 0
        precision = 0
        for i,batch in enumerate(dataloader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            model_kwargs = {"input_ids": inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"]}
            logits = prep_logits(
                model(**model_kwargs).logits,
                inputs["ans_idx"]
            )
            loss = criterion(logits, inputs["targets"].long())
            total_loss += loss.item()
            match = (logits.argmax(-1) == inputs["targets"])
            acc = match.float().mean().item()
            total_match += acc

            precision += match[logits.argmax(-1)==1].float().sum().item()
            total_precision += (logits.argmax(-1)==1).float().sum().item()
            recall += match[inputs["targets"]==1].float().sum().item()
            total_recall += (inputs["targets"]==1).float().sum().item()

            toxic_proportion += inputs["targets"].long().sum().item()
            print(i,"/", len(dataloader), end="\r")
            if config.get("debugging", False) and i >= 10:
                break
        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision/ total_precision if total_precision > 0 else 0
        avg_recall = recall / total_recall if total_recall > 0 else 0
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
        "max_components": size,
    }

    fca = FunctionalComponentAnalysis(**config["fca_params"])
    if "ortho_fcas" in config and config["ortho_fcas"]:
        fca = load_ortho_fcas(fca, config["ortho_fcas"])
    fca.to(DEVICE)

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
                        k: v.to(DEVICE2) for k,v in model_kwargs.items()
                    }).logits
                    targets = prep_logits(
                        base_output, batch["ans_idx"].to(DEVICE2)
                    ).to(DEVICE)
                    labels = targets.argmax(dim=-1)
                torch.cuda.empty_cache()
            else:
                targets = batch["targets"].long()
                labels = targets

            logits = prep_logits(
                model(**model_kwargs).logits,
                batch["ans_idx"]
            )
            loss = criterion(logits, targets)
            total_loss += loss.item()

            match = (logits.argmax(-1) == labels)
            acc = match.float().mean().item()
            total_match += acc

            precision += match[logits.argmax(-1)==1].float().sum().item()
            total_precision += (logits.argmax(-1)==1).float().sum().item()
            recall += match[inputs["targets"]==1].float().sum().item()
            total_recall += (inputs["targets"]==1).float().sum().item()

            # Track the complement loss
            compl_loss = torch.tensor(0.0, device=logits.device)
            if config.get("train_fca_complement", False):
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
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            fca.reset_cached_weight()

            print(f"Batch {bi+1}/{len(dataloader)}: Loss={loss.item():.4f}, Acc={acc:.4f}, "
                  f"Precision={precision/total_precision:.4f}, Recall={recall/total_recall:.4f}", end=" "*20+"\r")

        avg_loss = total_loss / len(dataloader)
        avg_match = total_match / len(dataloader)
        avg_precision = precision / total_precision if total_precision > 0 else 0
        avg_recall = recall / total_recall if total_recall > 0 else 0
        loss_history.append(avg_loss)
        match_history.append(avg_match)

        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, Acc={avg_match:.4f}, "
              f"Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, Components={fca.num_components}")

        with open(CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                epoch, avg_loss, avg_match, fca.num_components
            ])

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
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(range(len(match_history)), [fca.num_components for _ in match_history], marker='o')
    plt.title("Subspace Evolution")
    plt.xlabel("Epoch")
    plt.ylabel("Number of Components")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "subspace_evolution.png"))
