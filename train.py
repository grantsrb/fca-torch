import torch
from modeling import NeuralNetwork
from tasks import HierarchicalLogicalTask, OPERATION2STRING
import torch.optim as optim
from fca import FunctionalComponentAnalysis, load_fcas, initialize_fcas
import pandas as pd
import copy
import os
import time
import warnings
warnings.filterwarnings('ignore')

from utils import collect_activations
from dl_utils.save_io import (
    record_session, load_checkpoint, get_folder_from_path,
    count_files, count_sub_dirs
)
from dl_utils.utils import pretty_print_config

class PlateauTracker:
    def __init__(self, **kwargs):
        self.patience = kwargs.get("patience", 10)
        self.plateau = kwargs.get("plateau", 0.01)
        self.reset()

    def reset(self):
        self.best_val_loss = float("inf")
        self.best_val_acc = 0.0
        self.counter = 0

    def update(self, val_loss, val_acc):
        if val_loss < (self.best_val_loss-self.plateau):
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

def train(config, device=None):
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    # Load CheckPoint
    checkpoint = None
    if config['model_load_path'] is not None:
        checkpoint = load_checkpoint(config['model_load_path'])
        conf = {
            **copy.deepcopy(config),
            **copy.deepcopy(checkpoint['config'])
        }
        if config.get("save_to_load_path", True):
            conf["save_folder"] = get_folder_from_path(
                config["model_load_path"])
        for k in config.get("persistent_keys", []):
            if k in config:
                conf[k] = config[k]
        config = conf

    print("Using Config:")
    pretty_print_config(config)
    save_path = os.path.join(config['save_folder'], "model.pt")

    # Initialize the model, task, loss function, and optimizer
    kwargs = {**config['task_params']}
    task = HierarchicalLogicalTask(kwargs)
    config["task_params"]["vocab"] = task.vocab
    config["task_params"]["word2idx"] = task.word2idx
    config["task_params"]["idx2word"] = task.idx2word
    config["task_params"]["operations"] = task.operation_names
    kwargs["n_samples"] = int(0.2*kwargs["n_samples"])
    kwargs["operations"] = task.operations
    val_task = HierarchicalLogicalTask(kwargs)
    config["model_params"] = {
        **config["model_params"],
        "vocab_size": task.vocab_size,
        "n_input_tokens": task.n_input_tokens,
        "output_size": task.vocab_size,
    }
    model = NeuralNetwork(**config['model_params'])
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
    record_session(config=config, model=model, globals_dict=globals())
    model.to(device)
    print(model)
    criterion = torch.nn.CrossEntropyLoss()


    optimizer = optim.Adam(
        model.parameters(),
        lr=config['lr']
    )
    if checkpoint is not None:
        optimizer.load_state_dict(
            checkpoint['optimizer_state_dict']
        )
    plateau_tracker = PlateauTracker(**config)

    # Initialize the data dictionary
    data_dict = {
        "epoch": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    # Determine if fca training
    fca_layers = config.get("fca_layers", None)
    do_fca = config.get("do_fca", fca_layers) and fca_layers
    if do_fca:
        for layer in fca_layers:
            data_dict[f"fca_{layer}_rank"] = []

    # Load trained Functional Component Analysis Objects
    loaded_fcas = []
    loaded_handles = []
    if config.get("fca_load_path", None) is not None:
        model.freeze_parameters()
        model_arg = model if config.get("subtract_prev_fcas", True) else None
        print("model arg:", model_arg)
        loaded_fcas, loaded_handles = load_fcas(
            model=model_arg, load_path=config["fca_load_path"])

    # Get the initial loss and accuracy
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        for batch in task.get_batches(config['batch_size']):
            inputs = batch["input_ids"]
            labels = batch["output_ids"].squeeze()

            with torch.no_grad():
                actvs = collect_activations(
                    model,
                    input_data=inputs,
                    layers=config["fca_layers"],
                    ret_pred_ids=True,
                    batch_size=None,
                    to_cpu=True,
                    verbose=True,
                )

            # Collect Means and Stds Potentially for FCA
            if do_fca:
                means = { l: None for l in config["fca_layers"] }
                stds =  { l: None for l in config["fca_layers"] }
                if config.get("zscore_fca", False):
                    layer = config["fca_layers"]
                    means = { l: actvs[l].mean(0).squeeze() for l in config["fca_layers"] }
                    stds =  { l: actvs[l].std(0).squeeze() for l in config["fca_layers"] }
                
            logits = actvs["logits"]
            preds = actvs["pred_ids"]
            loss = criterion(logits, labels)
            acc = (preds == labels).float().mean()
            total_loss += loss.item()
            total_acc += acc.item()

        total_loss_val = 0.0
        total_acc_val = 0.0
        for batch in val_task.get_batches(config['batch_size']):
            inputs = batch["input_ids"]
            labels = batch["output_ids"]
            outputs = model(inputs.to(device))
            loss = criterion(
                outputs,
                labels.squeeze().to(device)
            )
            acc = (outputs.argmax(dim=-1) == labels.squeeze().to(device)).float().mean()
            total_loss_val += loss.item()
            total_acc_val += acc.item()

        print(f'Initial Loss: {total_loss/task.n_batches(config["batch_size"])}')
        print(f'Initial Acc: {total_acc/task.n_batches(config["batch_size"])}')
        print(f'Initial Val Loss: {total_loss_val/val_task.n_batches(config["batch_size"])}')
        print(f'Initial Val Acc: {total_acc_val/val_task.n_batches(config["batch_size"])}')
        print("Ops:", ", ".join([OPERATION2STRING[o] for o in task.operations]))
        print('---')

        ## Ensure high enough accuracy
        thresh = config["fca_acc_threshold"]
        train_acc = total_acc/task.n_batches(config["batch_size"])
        val_acc = total_acc_val/val_task.n_batches(config["batch_size"])
        if do_fca and not loaded_fcas and (thresh>train_acc or thresh>val_acc):
            print("Insufficient Accuracy for FCA Threshold", thresh)
            return None

        data_dict["epoch"].append(0)
        data_dict["train_loss"].append(total_loss/task.n_batches(config["batch_size"]))
        data_dict["train_acc"].append(train_acc)
        data_dict["val_loss"].append(total_loss_val/val_task.n_batches(config["batch_size"]))
        data_dict["val_acc"].append(val_acc)
        if do_fca:
            for layer in fca_layers:
                data_dict[f"fca_{layer}_rank"].append(0)

    # Initialize the Functional Component Analysis Objects
    fcas = {}
    fca_handles = {}
    if do_fca:
        print("Ensuring Ortho", config["ensure_ortho_chain"])
        model.freeze_parameters()
        fcas, fca_handles, fca_parameters = initialize_fcas(
            model=model,
            config=config,
            loaded_fcas=loaded_fcas,
            means=means,
            stds=stds,
        )
        optimizer = optim.Adam(fca_parameters, lr=config['lr'])

    # Training loop
    best_acc = 0
    for epoch in range(1,config['num_epochs']):
        #try:
            model.train()
            startt = time.time()
            total_loss = 0.0
            total_acc = 0.0

            ## Training Batches
            for batch in task.get_batches(config['batch_size']):
                inputs = batch["input_ids"]
                labels = batch["output_ids"].squeeze()

                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                preds = outputs.argmax(dim=-1)
                loss = criterion(outputs, labels.to(device))
                acc = (preds == labels.to(device)).float().mean()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_acc += acc.item()

            ## Validation Batches
            with torch.no_grad():
                model.eval()
                total_loss_val = 0.0
                total_acc_val = 0.0
                for batch in val_task.get_batches(config['batch_size']):
                    inputs = batch["input_ids"]
                    labels = batch["output_ids"]
                    outputs = model(inputs.to(device))
                    loss = criterion(
                        outputs,
                        labels.squeeze().to(device)
                    )
                    acc = (outputs.argmax(dim=-1) == labels.squeeze().to(device)).float().mean()
                    total_loss_val += loss.item()
                    total_acc_val += acc.item()

            ## Record and Print Epoch
            data_dict["epoch"].append(epoch)
            bsize = config['batch_size']
            trn_loss = total_loss/task.n_batches(bsize)
            trn_acc = total_acc/task.n_batches(bsize)
            data_dict["train_loss"].append(trn_loss)
            data_dict["train_acc"].append(trn_acc)
            print(f'Epoch {epoch}/{config["num_epochs"]}')
            print(f"Trn Loss: {trn_loss}, Trn Acc: {trn_acc}")
            val_loss = total_loss_val/val_task.n_batches(bsize)
            val_acc = total_acc_val/val_task.n_batches(bsize)
            data_dict["val_loss"].append(val_loss)  # val_loss
            data_dict["val_acc"].append(val_acc)  # val_acc
            print(f'Val Loss: {val_loss}, Val Acc: {val_acc}')
            print("Ops:", ", ".join([OPERATION2STRING[o] for o in task.operations]))
            if fcas:
                print("FCA Parameters:")
                for layer in fca_layers:
                    rank = 0
                    if layer in fcas:
                        rank = fcas[layer].rank
                        max_rank = fcas[layer].max_rank
                        full_rank = fcas[layer].size
                        print(f"\t{layer}: {rank}/{max_rank}/{full_rank}")
                    data_dict[f"fca_{layer}_rank"].append(rank)
            print("Save Dir:", config["save_folder"])
            print("Exec Time:", time.time()-startt)
            print('---')

            # Save Checkpoint if Best
            if not fcas and val_acc>best_acc:
                best_acc = val_acc
                save_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "epoch": epoch,
                }
                best_path = save_path.replace(".pt", "_best.pt")
                print("Saving model to:", best_path)
                torch.save(save_dict, best_path)

            # Check for loss plateaus
            if plateau_tracker.update(val_loss, val_acc):
                if fcas:
                    print("Updating FCA parameters")
                    new_axes = []
                    for fca in fcas.values():
                        fca.update_parameters()
                        fca.freeze_parameters()
                        p = fca.add_new_axis_parameter()
                        if p is not None:
                            new_axes.append(p)
                    if len(new_axes) == 0:
                        print("FCA Max Rank Early stopping")
                        break
                    optimizer = optim.Adam(new_axes, lr=config['lr'])
                    plateau_tracker.reset()
                else:
                    print("Early stopping")
                    break

            # Check for accuracy thresholds
            thresh = config.get("fca_acc_threshold", 0.99)
            if fcas and trn_acc>thresh and val_acc>thresh:
                print(f"Trn:{trn_acc}, Val:{val_acc} Early Stopping")
                break
        #except:
        #    print("Error occurred, closing out")
        #    break


    ## Save the trained model
    if fcas:
        # Ensure final dimensions are updated
        for fca in fcas.values():
            fca.update_parameters()
            fca.freeze_parameters()

        save_folder = config.get("fca_save_folder", config["save_folder"])
        n_chains = count_files(
            main_folder=save_folder,
            substr="fca_chain",
            exts={".pt"}
        )
        save_path = f"{save_folder}/fca_chain{n_chains}.pt"
        save_dict = {
            "fca_state_dicts": {
                layer: fca.state_dict() for layer,fca in fcas.items()
            },
            "config": config,
        }
        for layer,h in fca_handles.items():
            h.remove()
        for handles in loaded_handles:
            for layer,h in handles.items():
                h.remove()
    else:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
    print("Saving to:", save_path)
    torch.save(save_dict, save_path)
    df = pd.DataFrame(data_dict)
    csv_path = save_path.replace(".pt", ".csv")
    df.to_csv(csv_path, index=False, header=True)
    trn_acc = total_acc/task.n_batches(bsize)
    trn_loss = total_loss/task.n_batches(bsize)
    val_acc = total_acc_val/val_task.n_batches(bsize)
    val_loss = total_loss_val/val_task.n_batches(bsize)
    metrics = {
        "ranks": {},
        "trn_acc":  float(trn_acc),
        "trn_loss": float(trn_loss), 
        "val_acc":  float(val_acc),
        "val_loss": float(val_loss), 
        "save_path": save_path,
    }
    if fcas:
        for layer in fcas:
            metrics["ranks"][layer] = fcas[layer].rank
    return metrics
