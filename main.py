from train import train
from utils import join_configs, get_command_line_args
from dl_utils.save_io import get_save_folder
from dl_utils.utils import pretty_print_config
from constants import DEFAULTS
import torch
import numpy as np
import copy

if __name__ == "__main__":
    defaults = copy.deepcopy(DEFAULTS)
    arg_config = get_command_line_args()
    name_keys = list(arg_config.keys())
    config = join_configs(kwargs=arg_config, defaults=defaults)

    config["seed"] = config.get("seed", 123456)
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    if not (config.get("model_load_path", None) and config.get("save_to_load_path", False)):
        save_folder, model_folder = get_save_folder(
            config,
            name_keys=name_keys,
            mkdirs=True,
            ret_model_folder=True)
        config["save_folder"] = save_folder
        config["model_folder"] = model_folder
        config["persistent_keys"].append("save_folder")
        config["persistent_keys"].append("model_folder")

    train(config)
