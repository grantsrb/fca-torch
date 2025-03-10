import torch
import pickle
import os
import json
import yaml
from .utils import get_git_revision_hash, package_versions, get_datetime_str
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def save_checkpt(save_dict,
        save_folder,
        epoch,
        save_name="checkpt",
        ext=".pt",
        del_prev_sd=True,
        best=False):
    """
    Saves a dictionary that contains a statedict

    save_dict: dict
        a dictionary containing all the things you want to save
    save_folder: str
        the full path to save the checkpt file to
    save_name: str
        the name of the file that the save dict will be saved to. This
        function will automatically append the epoch to the end of the
        save name followed by the extention, `ext`.
    epoch: int
        an integer to be associated with this checkpoint
    ext: str
        the extension of the file
    del_prev_sd: bool
        if true, the state_dict of the previous checkpoint will be
        deleted
    best: bool
        if true, additionally saves this checkpoint as the best
        checkpoint under the filename set by BEST_CHECKPT_NAME
    """
    if del_prev_sd and epoch is not None:
        prev_paths = get_checkpoints(save_folder)
        if len(prev_paths) > 0:
            prev_path = prev_paths[-1]
            delete_sds(prev_path)
        elif epoch != 0:
            print("Failed to find previous checkpoint")
    if epoch is None: epoch = 0
    path = "{}_{}{}".format(save_name,epoch,ext)
    path = os.path.join(save_folder, path)
    path = os.path.abspath(os.path.expanduser(path))
    torch.save(save_dict, path)
    if best: save_best_checkpt(save_dict, save_folder)

def delete_sds(checkpt_path):
    """
    Deletes the state_dicts from the argued checkpt path.

    Args:
        checkpt_path: str
            the full path to the checkpoint
    """
    if not os.path.exists(checkpt_path): return
    checkpt = load_checkpoint(checkpt_path)
    keys = list(checkpt.keys())
    for key in keys:
        if "state_dict" in key or "optim_dict" in key:
            del checkpt[key]
    torch.save(checkpt, checkpt_path)

def get_checkpoints(folder, checkpt_exts={'p', 'pt', 'pth'}):
    """
    Returns all .p, .pt, and .pth file names contained within the
    folder. They're sorted by their epoch.

    BEST_CHECKPT_PATH is not included in this list. It is excluded using
    the assumption that it has the extension ".best"

    folder: str
        path to the folder of interest
    checkpt_exts: set of str
        a set of checkpoint extensions to include in the checkpt search.

    Returns:
        checkpts: list of str
            the full paths to the checkpoints contained in the folder
    """
    folder = os.path.expanduser(folder)
    assert os.path.isdir(folder)
    checkpts = []
    for f in os.listdir(folder):
        splt = f.split(".")
        if len(splt) > 1 and splt[-1] in checkpt_exts:
            path = os.path.join(folder,f)
            checkpts.append(path)
    checkpts = sorted(checkpts)
    return checkpts

def foldersort(x):
    """
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    Assumes that the experiment number will always be the rightmost
    occurance of an integer surrounded by underscores (i.e. _1_)

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    for s in reversed(splt[1:]):
        try:
            return int(s)
        except:
            pass
    print("Failed to sort:", x)
    return np.inf

def prep_search_keys(s):
    """
    Removes unwanted characters from the search keys string. This
    allows you to easily append a string representing the search
    keys to the model folder name.
    """
    return s.replace(
            " ", ""
        ).replace(
            "[", ""
        ).replace(
            "]", ""
        ).replace(
            "\'", ""
        ).replace(
            ",", ""
        ).replace(
            "/", ""
        )

def get_exp_num(path):
    """
    Finds and returns the experiment number from the argued path.
    """
    return foldersort(path)

def get_exp_name(x):
    """
    Finds and returns the string before the experiment number from
    the argued path.
    
    A sorting key function to order folder names with the format:
    <path_to_folder>/<exp_name>_<exp_num>_<ending_folder_name>/

    Assumes that the experiment number will always be the rightmost
    occurance of an integer surrounded by underscores (i.e. _1_)

    x: str
    """
    if x[-1] == "/": x = x[:-1]
    splt = x.split("/")
    if len(splt) > 1: splt = splt[-1].split("_")
    else: splt = splt[0].split("_")
    exp_num = None
    for i,s in enumerate(reversed(splt)):
        try:
            exp_num = int(s)
            break
        except:
            pass 
    if exp_num is None: return None
    else: return "_".join(splt[:-(i+1)])
  
def is_model_folder(path, exp_name=None, incl_empty=True):
    """
    checks to see if the argued path is a model folder or otherwise.
    i.e. does the folder contain checkpt files and a hyperparams.json?

    path: str
        path to check
    exp_name: str or None
        optionally include exp_name to determine if a folder is a model
        folder based on the name instead of the contents.
    incl_empty: bool
        if true, will include folders without checkpoints as possible
        model folders.
    """
    check_folder = os.path.expanduser(path)
    if not os.path.isdir(check_folder): return False
    if incl_empty and exp_name is not None:
        # Remove ending slash if there is one
        if check_folder[-1]=="/": check_folder = check_folder[:-1]
        folder_splt = check_folder.split("/")[-1]
        # Need to split on underscores and check for entirety of
        # exp_name because exp_name is only the first part of any
        # model folder
        name_splt = exp_name.split("_")
        folder_splt = folder_splt.split("_")
        match = True
        for i in range(len(name_splt)):
            if i >= len(folder_splt) or name_splt[i] != folder_splt[i]:
                match = False
                break
        if match:
            return True
    contents = os.listdir(check_folder)
    is_empty = True
    has_hyps = False
    for content in contents:
        if ".pt" in content: is_empty = False
        if "hyperparams" in content: has_hyps = True
    if incl_empty: return has_hyps or not is_empty
    return not is_empty

def is_incomplete_folder(path):
    """
    checks to see if the argued path is an empty model folder. 
    i.e. does the folder contain a hyperparams.json without checkpt
    files? Generally it is okay to delete empty model folders.

    WARNING: ONLY RETURNS TRUE IF THE FOLDER CONTAINS A HYPERPARAMETERS
    JSON WITHOUT ANY CHECKPOINTS. WILL RETURN FALSE FOR COMPLETELY
    EMPTY FOLDERS!!!!

    path: str
        path to check
    exp_name: str or None
    """
    check_folder = os.path.expanduser(path)
    if not os.path.isdir(check_folder): return False
    contents = os.listdir(check_folder)
    is_empty = True
    has_hyps = False
    for content in contents:
        if ".pt" in content: is_empty = False
        if "hyperparams" in content: has_hyps = True
    return has_hyps and is_empty

def is_exp_folder(path):
    """
    Checks to see if the argued path is an exp folder. i.e. does it
    contain at least 1 model folder.

    Args:
        path: str
            full path to the folder in question.
    Returns:
        is_folder: bool
            if the argued path is to an experiment folder, will return
            true. Otherwise returns false.
    """
    if not os.path.isdir(path): return False
    mfs = get_model_folders(path)
    return len(mfs)>0

def get_model_folders(exp_folder, incl_full_path=False, incl_empty=True):
    """
    Returns a list of paths to the model folders contained within the
    argued exp_folder

    exp_folder - str
        full path to experiment folder
    incl_full_path: bool
        include extension flag. If true, the expanded paths are
        returned. otherwise only the end folder (i.e.  <folder_name>
        instead of exp_folder/<folder_name>)
    incl_empty: bool
        if true, will include folders without checkpoints as possible
        model folders.

    Returns:
        list of folder names (see incl_full_path for full path vs end
        point)
    """
    folders = []
    exp_folder = os.path.expanduser(exp_folder)
    if exp_folder[-1]=="/":
        exp_name = exp_folder[:-1].split("/")[-1]
    else:
        exp_name = exp_folder.split("/")[-1]
    if ".pt" in exp_folder[-4:]:
        # if model file, return the corresponding folder
        folders = [ "/".join(exp_folder.split("/")[:-1]) ]
    else:
        for d, sub_ds, files in os.walk(exp_folder):
            for sub_d in sub_ds:
                check_folder = os.path.join(d,sub_d)
                is_mf = is_model_folder(
                    check_folder,exp_name=exp_name,incl_empty=incl_empty
                )
                if is_mf:
                    if incl_full_path:
                        folders.append(check_folder)
                    else:
                        folders.append(sub_d)
        if is_model_folder(exp_folder,incl_empty=incl_empty):
            folders.append(exp_folder)
    folders = list(set(folders))
    if incl_full_path: folders = [os.path.expanduser(f) for f in folders]
    return sorted(folders, key=foldersort)

def load_checkpoint(path, checkpt_name="model.pt"):
    """
    Loads the save_dict into python.

    Args:
        path: str
            path to checkpoint file or model_folder
    Returns:
        checkpt: dict
            a dict that contains all the valuable information for the
            training.
    """
    path = os.path.expanduser(path)
    hyps = None
    if os.path.isdir(path):
        checkpts = get_checkpoints(path)
        if len(checkpts)==0: return None
        filt_checkpts = [cp for cp in checkpts if checkpt_name in cp]
        if len(filt_checkpts)==0:
            path = checkpts[-1]
        else:
            path = filt_checkpts[-1]
    data = torch.load(path, map_location=torch.device("cpu"))
    data["loaded_path"] = path
    if "config" not in data: 
        data["config"] = get_config(path)
    return data

def load_model(path, models, load_sd=True, use_best=False,
                                           hyps=None,
                                           verbose=True):
    """
    Loads the model architecture and state dict from a .pt or .pth
    file. Or from a training save folder. Defaults to the last check
    point file saved in the save folder.

    path: str or dict
        either .pt,.p, or .pth checkpoint file; or path to save folder
        that contains multiple checkpoints. if dict, must be a checkpt
        dict.
    models: dict
        A dict of the potential model classes. This function is
        easiest if you import each of the model classes in the calling
        script and simply pass `globals()` as the argument for this
        parameter. If None is argued, `globals()` is used instead.
        (can usually pass `globals()` as the arg assuming you have
        imported all of the possible model classes into the script
        that calls this function)

        keys: str
            the class names of the potential models
        vals: Class
            the potential model classes
    load_sd: bool
        if true, the saved state dict is loaded. Otherwise only the
        model architecture is loaded with a random initialization.
    use_best: bool
        if true, will load the best model based on validation metrics
    hyps: dict (optional)
        if you would like to argue your own hyps, you can do that here
    """
    if type(path) == type(str()):
        path = os.path.expanduser(path)
        hyps = None
        data = load_checkpoint(path,use_best=use_best)
    else: data = path
    if 'hyps' in data:
        kwargs = data['hyps']
    elif 'model_hyps' in data:
        kwargs = data['model_hyps']
    elif "config" in data:
        kwargs = data["config"]
    else:
        kwargs = get_hyps(path)
    if models is None: models = globals()
    model = models[kwargs['model_type']](**kwargs)
    if "state_dict" in data and load_sd:
        print("loading state dict")
        try:
            model.load_state_dict(data["state_dict"])
        except:
            try:
                sd = {k:v.clone() for k,v in data["state_dict"].items()}
                m_sd = model.state_dict()
                keys = list(sd.keys())
                for k in keys:
                    if "model." in key and key not in m_sd:
                        # Simply remove "model." from keys
                        new_key = ".".join(key.split(".")[1:])
                        sd[new_key] = sd[key]
                        del sd[key]
                model.load_state_dict(sd)
            except:
                print("failed to load state dict, attempting fix")
                sd = data["state_dict"]
                m_sd = model.state_dict()
                keys = {*sd.keys(), *m_sd.keys()}
                for k in keys:
                    if k not in sd:
                        print("Error for", k)
                        sd[k] = getattr(model, k)
                    if k not in m_sd:
                        print("Error for", k)
                        setattr(model, k, sd[k])
                model.load_state_dict(sd)
                print("succeeded!")
    else:
        print("state dict not loaded!")
    return model

def get_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json or
    yaml save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    folder = os.path.expanduser(folder)
    if not os.path.isdir(folder):
        folder = "/".join(folder.split("/")[:-1])
    for name in ["hyperparams", "config"]:
        for ext in ["json", "yaml"]:
            f = os.path.join(folder, f"{name}.{ext}")
            if os.path.exists(f):
                hyps = load_json_or_yaml(f)
                return hyps
    return None

def load_hyps(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def load_config(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def get_config(folder):
    """
    Returns a dict of the hyperparameters collected from the json
    save file in the model folder.

    folder: str
        path to the folder that contains checkpts and a hyps json file
    """
    return get_hyps(folder)

def load_init_checkpt(model, config):
    """
    Easily load a checkpoint into the model at initialization.

    model: torch module
    config: dict
        "init_checkpt": str
    """
    init_checkpt = config.get("init_checkpt", None)
    if init_checkpt is not None and init_checkpt.strip()!="":
        if not os.path.exists(init_checkpt):
            init_checkpt = os.path.join(config["save_root"], init_checkpt)
        print("Initializing from checkpoint", init_checkpt)
        checkpt = load_checkpoint(init_checkpt)
        try:
            model.load_state_dict(checkpt["state_dict"])
        except:
            print("Failed to load checkpt, attempting fix...")
            sd = checkpt["state_dict"]
            mskeys = set(model.state_dict().keys())
            sym_diff = mskeys.symmetric_difference(set(sd.keys()))
            if len(sym_diff)>0:
                print("State Dict Symmetric Difference")
                for k in sym_diff: 
                    if k in mskeys:
                        print("MODEL:", k, model.state_dict()[k].shape)
                    else:
                        print("CHECKPT:", k, sd[k].shape)

            for key in sync_keys:
                if key in model.state_dict():
                    sd[key] = model.state_dict()[key]
            model.load_state_dict(sd)
    return model

def exp_num_exists(exp_num, exp_folder):
    """
    Determines if the argued experiment number already exists for the
    argued experiment name.

    exp_num: int
        the number to be determined if preexisting
    exp_folder: str
        path to the folder that contains the model folders
    """
    folders = get_model_folders(exp_folder)
    for folder in folders:
        num = foldersort(folder)
        if exp_num == num:
            return True
    return False

def get_new_exp_num(exp_folder, exp_name, offset=0):
    """
    Finds the next open experiment id number by searching through the
    existing experiment numbers in the folder.

    If an offset is argued, it is impossible to have an exp_num that is
    less than the value of the offset. The returned exp_num will be
    the next available experiment number starting with the value of the
    offset.

    Args:
        exp_folder: str
            path to the main experiment folder that contains the model
            folders. i.e. if the `exp_name` is "myexp" and there is
            a folder that contains a number of model folders, then
            exp_folder would be "/path/to/myexp/"
            If None is argued, assumes "./<exp_name>/
        exp_name: str
            the name of the experiment
        offset: int
            a number to offset the experiment numbers by.

    Returns:
        exp_num: int
    """
    if not exp_folder: exp_folder = os.path.join("./", exp_name)
    name_splt = exp_name.split("_")
    namedex = 1
    if len(name_splt) > 1:
        namedex = len(name_splt)
    exp_folder = os.path.expanduser(exp_folder)
    if not os.path.exists(exp_folder):
        os.mkdir(exp_folder)
    _, dirs, _ = next(os.walk(exp_folder))
    exp_nums = set()
    for d in dirs:
        splt = d.split("_")
        if len(splt) >= 2:
            num = None
            for i in reversed(range(len(splt))):
                try:
                    num = int(splt[i])
                    break
                except:
                    pass
            if namedex > 1 and i > 1:
                name = "_".join(splt[:namedex])
            else: name = splt[0]
            if name == exp_name and num is not None:
                exp_nums.add(num)
    for i in range(len(exp_nums)):
        if i+offset not in exp_nums:
            return i+offset
    return len(exp_nums) + offset

def load_yaml(file_name):
    """
    Loads a yaml file as a python dict

    file_name: str
        the path of the yaml file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        yam = yaml.safe_load(f)
    return yam

def load_json(file_name):
    """
    Loads a json file as a python dict

    file_name: str
        the path of the json file
    """
    file_name = os.path.expanduser(file_name)
    with open(file_name,'r') as f:
        s = f.read()
        j = json.loads(s)
    return j

def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False

def save_json(data, file_name):
    """
    saves a dict to a json file

    data: dict
    file_name: str
        the path that you would like to save to
    """
    failure = True
    n_loops = 0
    while failure and n_loops<10*len(data):
        failure = False
        n_loops += 1
        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
        except (TypeError, OverflowError):
            data = {**data}
            keys = list(data.keys())
            for k in keys:
                if not is_jsonable(data[k]):
                    if type(data[k])==dict:
                        data = {**data, **data[k]}
                        del data[k]
                    elif type(data[k])==set:
                        data[k] = list(data[k])
                    elif hasattr(data[k],"__name__"):
                        data[k] = data[k].__name__
                    else:
                        try:
                            data[k] = str(data[k])
                        except:
                            del data[k]
                            print("Removing", k, "from json")
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
            except:
                print("trying again")
                failure = True

def load_json_or_yaml(file_name):
    """
    Loads a json or a yaml file (determined by its extension) as a python
    dict.

    Args:
        file_name: str
            the path of the json/yaml file
    Returns:
        d: dict
            a dict representation of the loaded file
    """
    if ".json" in file_name:
        return load_json(file_name)
    elif ".yaml" in file_name:
        return load_yaml(file_name)
    raise NotImplemented

def get_save_folder(config, name_keys=[], mkdirs=True, ret_model_folder=False):
    exp_name = config.get("exp_name", "myexperiment")
    exp_folder = os.path.join( config.get("save_root", "./"), exp_name )
    if mkdirs and not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)
    exp_num = get_new_exp_num(exp_folder, exp_name)
    model_folder = f"{exp_name}_{exp_num}"
    if name_keys:
        model_folder += "_"+get_save_identifier(
            config=config,
            save_keys=name_keys,
            save_folder=exp_folder, 
        )
    save_folder = os.path.join(exp_folder, model_folder)
    if mkdirs:
        os.makedirs(save_folder, exist_ok=True)
    if ret_model_folder:
        return save_folder, model_folder
    return save_folder

def record_session(config, model, globals_dict=None, verbose=False):
    """
    Writes important parameters to file. If 'resume_folder' is an entry
    in the config dict, then the txt file is appended to instead of being
    overwritten.

    config: dict
        dict of relevant hyperparameters. needs "save_folder" key.
    model: torch nn.Module
        the model to be trained
    globals_dict: dict
        just argue `globals()`
    """
    try:
        config["git_hash"] = config.get(
            "git_hash", get_git_revision_hash()
        )
    except:
        s="you aren't using git?! you should really version control..."
        config["git_hash"] = s
        print(s)
    git_hash = config["git_hash"]
    sf = config['save_folder']
    if not os.path.exists(sf):
        os.mkdir(sf)
    h = "config"
    mode = "a" if "resume_folder" in config else "w"
    packages = package_versions(globals_dict=globals_dict,verbose=verbose)
    with open(os.path.join(sf,h+".txt"),mode) as f:
        dt_string = get_datetime_str()
        f.write(dt_string)
        f.write("\nGit Hash: {}".format(git_hash))
        f.write("\nPackage Versions:")
        for module_name,v in packages.items():
            f.write("\t{}: {}\n".format(module_name, v))
        f.write("\n"+str(model)+'\n')
        for k in sorted(config.keys()):
            f.write(str(k) + ": " + str(config[k]) + "\n")
    temp_hyps = dict()
    keys = list(config.keys())
    temp_hyps = {k:v for k,v in config.items()}
    if verbose:
        print("\nConfig:")
    for k in keys:
        if verbose and k!="packages":
            print("\t{}:".format(k), temp_hyps[k])
        if type(config[k]) == type(np.array([])):
            del temp_hyps[k]
        elif type(config[k])==np.int64:
            temp_hyps[k] = int(temp_hyps[k])
        elif type(config[k])==type(set()):
            temp_hyps[k] = list(config[k])
    if "packages" not in temp_hyps:
        temp_hyps["packages"] = packages
    save_json(temp_hyps, os.path.join(sf,h+".json"))

def get_folder_from_path(path):
    if os.path.isdir(path): return path
    return "/".join(path.split("/")[:-1])

def get_num_duplicates(folder, fname, sep="_v"):
    """
    Returns the number of files in the folder that
    match fname+"_n" where n is an integer.
    """
    n_dupls = 0
    for f in os.listdir(folder):
        if f==fname or sep.join(f.split(sep)[:-1])==fname:
            n_dupls += 1
    return n_dupls

def get_save_identifier(
        config,
        save_keys,
        save_folder=None,
        abbrevs = {},
        ignores = {
            "print_every",
            "task_params",
            "model_params",
            "model_save_path",
            "model_load_path",
            "fca_params",
            "persistent_keys",
            "exp_name",
        },):
    """
    Args:
        config: dict
            the keys you want to use first for identification
        save_keys: list of str
            the keys to use to create the save id
    """
    abbrevs = {
        **abbrevs,
        "fca_layers": "flyrs",
        "fca_load_path": "fpth",
        "plateau":   "pltu",
        "num_epochs": "neps",
        "batch_size": "bsz",
        "True": "T",
        "False": "F",
        "true": "T",
        "false": "F",
        "relaxed": "rlxd",
    }
    # add key value pairs to folder name
    if save_keys is None:
        save_keys = config["save_keys"]

    save_name = ""
    for k in sorted(list(save_keys)):
        if k in ignores: continue
        has_len = hasattr(config[k],"__len__")
        if type(config[k])!=str and has_len:
            val = "".join([
              str(e)[:3]+str(e)[-2:] if len(str(e))>3 else str(e)[:3] for e in config[k]
            ][:3])
        else:
            if hasattr(config[k], "__name__"):
                val = config[k].__name__[:5]
            else:
                c = str(config[k])
                if c[-1]=="/": c = c[:-1]
                val = c.split("/")[-1][:5]
        save_name += abbrevs.get(k, k).replace("_","")[:7]+val+"_"
    save_name = save_name[:-1]
    for k,v in abbrevs.items():
        save_name = save_name.replace(k,v)
    if save_folder:
        n_dupls = get_num_duplicates(save_folder, save_name)
        save_name = save_name + f"_v{n_dupls}"
    return save_name
