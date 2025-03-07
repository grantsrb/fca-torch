"""
This script assumes you want to do a chaining until all dimensions are taken.
The algorithm is as follows:
- freeze model weights at high accuracy
- make new fca
    - train fca to accuracy threshold (no limit on rank)
- make new chained fca
    - limit rank to be full rank minus previous fca ranks
    - train to accuracy threshold or rank completion 
- track acc of last fca and number of fcas compared to model size

"""
from train import train
from utils import join_configs, get_command_line_args
from dl_utils.save_io import (
    get_save_folder, load_checkpoint, get_checkpoints, get_folder_from_path
)
import torch
import numpy as np
import pandas as pd
import os
import time
import copy

defaults = {
    "save_root": "/data2/grantsrb/fca_saves/",
    "exp_name": "fca_prototyping",
    'task_params': {
        'n_pairs': 4,
        'n_samples': 1000,
    },
    'model_params': {
        'embedding_dim': 32,
        'd_model': 512,
        'n_layers': 3,
        'nonlinearity': "ReLU",
        'lnorm': True,
    },
    'lr': 0.001,
    'num_epochs': 100000,
    'batch_size': 128,
    "patience": 100,
    "plateau": 0.005,
    'model_load_path': "/data2/grantsrb/fca_saves/fca_prototyping/fca_prototyping_12/",
    "save_to_load_path": True,

    'fca_load_path': None,
    'fca_params': {
        "max_rank": None
    },
    'fca_layers': ["hidden_layers.1"],
    "fca_acc_threshold": 0.99, # stops early when fca module reaches this trn and val acc
    "ensure_ortho_chain": False, # will load the sd from the previous fca into the newest 

    'persistent_keys': [
        'fca_params', 'fca_layers', 'lr',
        "fca_load_path", "batch_size", "num_epochs",
    ],
}

if __name__ == "__main__":
    arg_config = get_command_line_args()
    config = join_configs(kwargs=arg_config, defaults=defaults)
    if not (config.get("model_load_path", None) and config.get("save_to_load_path", False)):
        save_folder, model_folder = get_save_folder(
            config, mkdirs=True, ret_model_folder=True)
        config["save_folder"] = save_folder
        config["model_folder"] = model_folder
        config["persistent_keys"].append("save_folder")
        config["persistent_keys"].append("model_folder")

    config["seed"] = config.get("seed", 123456)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    
    # track acc of last fca and number of fcas compared to model size
    data_dict = {
        "chain_num": [],
        "trn_acc": [],
        "trn_loss": [],
        "val_acc": [],
        "val_loss": [],
        "rank": [],
        "cumu_rank": [],
    }
    assert not config["fca_load_path"] and len(config["fca_layers"])==1
    if "full_rank" not in config:
        checkpoint = load_checkpoint(config['model_load_path'])
        layer = config["fca_layers"][0]
        sd = checkpoint["model_state_dict"]
        full_rank = None
        for key in sd.keys():
            k = key.replace(".weight", "")
            if k==layer:
                full_rank = sd[key].shape[0]
        if not full_rank:
            full_rank = checkpoint["config"]["model_params"]["d_model"]

    og_config = copy.deepcopy(config)
    metrics = {}
    cumu_rank = 0
    chain_num = -1
    while cumu_rank < full_rank:
        chain_num += 1
        print("New Chain", chain_num,
            "-- Cumu Rank:", cumu_rank,
            "-- Full Rank", full_rank)
        time.sleep(2)
        startt = time.time()
        rank = full_rank - cumu_rank
        config = copy.deepcopy(og_config)
        config["fca_params"]["max_rank"] = rank
        config["fca_load_path"] = metrics.get("save_path", None)

        metrics = train(config)

        fca_rank = list(metrics["ranks"].values())[-1] # list due to layers
        if config["ensure_ortho_chain"]:
            new_rank = fca_rank-cumu_rank
            cumu_rank = fca_rank
        else:
            new_rank = fca_rank
            cumu_rank += fca_rank
        data_dict["chain_num"].append(chain_num)
        data_dict["trn_acc"].append(metrics["trn_acc"])
        data_dict["trn_loss"].append(metrics["trn_loss"])
        data_dict["val_acc"].append(metrics["val_acc"])
        data_dict["val_loss"].append(metrics["val_loss"])
        data_dict["rank"].append(new_rank)
        data_dict["cumu_rank"].append(cumu_rank)
        print("Exec Time:", time.time()-startt)
    save_path = os.path.join(
        get_folder_from_path(config["model_load_path"]),
        "chain_results.csv",
    )
    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, header=True, index=False)
    print("DataFrame:")
    print(df)
    print("Saving to", save_path)
    print("Finished Chain")
    print("----")
