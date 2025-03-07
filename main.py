from train import train
from utils import join_configs, get_command_line_args
from dl_utils.save_io import get_save_folder

defaults = {
    "save_root": "/data2/grantsrb/fca_saves/",
    "exp_name": "fca_prototyping",
    'task_params': {
        'n_pairs': 4,
        'n_samples': 1000,
        "operations": ["or", "xor", "and", "and", "xor", "and", "and"],
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
    'model_load_path': None,
    "save_to_load_path": True,

    'fca_load_path': None,
    'fca_params': {
        "max_rank": None
    },
    'fca_layers': ["hidden_layers.1"],
    "fca_acc_threshold": 0.99,
    "ensure_ortho_chain": False, # will load the sd from the previous fca into the newest 

    'persistent_keys': [
        'fca_params', 'fca_layers', 'lr',
        "fca_load_path", "batch_size", "num_epochs",
        "ensure_ortho_chain",
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
    train(config)
