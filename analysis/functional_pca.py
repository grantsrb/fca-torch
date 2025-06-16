import torch
import numpy as np
import pandas as pd
import os
import time
import sys
sys.path.append("../")
sys.path.append("./")
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from modeling import NeuralNetwork
import dl_utils.save_io as savio
from tasks import HierarchicalLogicalTask
from fca import FunctionalComponentAnalysis, load_fcas
from utils import collect_activations, perform_pca

def sort_key(f):
    if "chained" not in f: return -1
    else:
        return int(
            f.split("chained")[-1].split(".pt")[0].split("_")[0]
        )

def find_dims_to_sum(arr, sum_val):
    s = 0
    for i,a in enumerate(arr):
        s += a
        if s>sum_val: return i+1
    return -1
        
def make_fca_from_pca(pc_vecs):
    """
    Args:
        pc_vecs: list-like of pytorch tensor vectors
    """
    pc_vecs = [p.squeeze() for p in pc_vecs]
    size = pc_vecs[0].shape[0]
    fca = FunctionalComponentAnalysis(size=size)
    fca.add_params_from_vector_list(pc_vecs, overwrite=True)
    fca.freeze_parameters()
    fca.set_fixed(True)
    return fca


if __name__=="__main__":
    n_samples = 1000
    overwrite = True
    
    device = "cpu" if not torch.cuda.is_available() else 0
    model_folders = []
    for main_folder in sys.argv[1:]:
        model_folders += savio.get_model_folders(main_folder, incl_full_path=True)
    for model_folder in model_folders:
        startt = time.time()
        csv_path = os.path.join(model_folder, "functional_pca.csv")
        if os.path.exists(csv_path) and not overwrite:
            print("Skipping", model_folder, "due to preexisting csv")
            continue
        print("Running Functional PCA on", model_folder)

        # Load CheckPoint
        checkpoint = savio.load_checkpoint(model_folder)
        config = checkpoint["config"]
        
        # Load Model
        model = NeuralNetwork(**config['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.freeze_parameters()
        model.eval()
        model.to(device)
        
        kwargs = {**config['task_params']}
        kwargs["n_samples"] = n_samples
        print("Task Config:")
        for k in sorted(list(kwargs.keys())):
            print(f"\t{k}", kwargs[k])
        task = HierarchicalLogicalTask(kwargs)
        data = next(task.get_batches(n_samples))
        
        fca_files = []
        for f in os.listdir(model_folder):
            if "fca" in f and ".pt" in f:
                fca_files.append(
                    os.path.join(model_folder, f))
        fca_files = sorted(fca_files, key=sort_key)
        if len(fca_files)>0:
            fca_path = fca_files[-1] # Pick the most recent. The others will be automatically loaded as well.
            checkpt = torch.load(fca_path)
            fca_layers = checkpt["config"]["fca_layers"][:1]
            ortho_ensured = checkpt["config"]["ensure_ortho_chain"]
            print("Orthogonality Ensured:", ortho_ensured)
        else: fca_layers = ["hidden_layers.1"]
        
        print("Collecting Activations")
        with torch.no_grad():
            actvs = collect_activations(
                model,
                input_data=data["input_ids"],
                layers=fca_layers,
                ret_pred_ids=True,
                batch_size=None,
                to_cpu=True,
                verbose=True,
            )
        
        actvs[fca_layers[0]] = actvs[fca_layers[0]].squeeze()
        actvs["pred_ids"] = actvs["pred_ids"].squeeze()
        
        print("N Samples:", n_samples)
        acc = (actvs["pred_ids"]==torch.zeros_like(actvs["pred_ids"])).float().mean().item()
        print("All Zeros Acc:", acc)
        acc = (actvs["pred_ids"]==torch.ones_like(actvs["pred_ids"])).float().mean().item()
        print("All Ones Acc:", acc)
        acc = (actvs["pred_ids"]==data["output_ids"].squeeze()).float().mean().item()
        print("Model Accuracy:", acc)

        ## PCA
        print("Normalizing")
        X = actvs[fca_layers[0]]
        print("X:", X.shape, type(X))
        print("Performing PCA")
        pca = perform_pca(X, center=True, scale=True)

        exp_var = pca["prop_explained_variance"].cpu().data.numpy()
        print(
            "Explained Variance",
            [round(x, 3) for x in exp_var])
        print("Total Dims:", X.shape[1])
        print(
            "Dims to 95%:",
            find_dims_to_sum(
                [round(x, 3) for x in exp_var]
                ,0.95
            )
        )
        print(
            "Dims to 99%:",
            find_dims_to_sum(
                [round(x, 3) for x in exp_var]
                ,0.99
            )
        )
        print(
            "Dims to 99.5%:",
            find_dims_to_sum(
                [round(x, 3) for x in exp_var],
                0.995
            )
        )
        
        pcs = pca["components"]
        means = pca["means"]
        stds = pca["stds"]

        fca = make_fca_from_pca(pcs)
        fca.set_means(means)
        fca.set_stds(stds)
        fca.to(device)
        for layer,modu in model.named_modules():
            if layer == fca_layers[0]:
                handle = modu.register_forward_hook(
                    fca.get_forward_hook()
                )

        dim_data = {
            "n_dims": [],
            "acc": [],
            "exp_var_p": [],
            "cumu_exp_var_p": [],
        }
        data["input_ids"] = data["input_ids"].to(device)
        targets = data["output_ids"].to(device).squeeze()
        print("Performing functional PCA")
        for n_dims in tqdm(reversed(range(1,len(pcs)+1))):
            fca.component_mask = torch.arange(n_dims).long().to(device)

            with torch.no_grad():
                actvs = collect_activations(
                    model,
                    input_data=data["input_ids"],
                    layers=[],
                    ret_pred_ids=True,
                    batch_size=None,
                    to_cpu=False,
                    verbose=False,
                )
            acc = (actvs["pred_ids"]==targets).float().mean().item()
            dim_data["n_dims"].append(n_dims)
            dim_data["acc"].append(acc)
            dim_data["exp_var_p"].append(exp_var[n_dims-1])
            cevp = exp_var[:n_dims].sum()
            dim_data["cumu_exp_var_p"].append(cevp)

        df = pd.DataFrame(dim_data)
        df["layer"] = fca_layers[-1]
        df.to_csv(csv_path, index=False, header=True)
        print("Dims to 95% Acc  :", np.min(df.loc[df["acc"]>=0.95, "n_dims"]))
        print("Dims to 99% Acc  :", np.min(df.loc[df["acc"]>=0.99, "n_dims"]))
        print("Dims to 99.5% Acc:", np.min(df.loc[df["acc"]>=0.995, "n_dims"]))
        print("Finishing", model_folder.split("/")[-1])
        print("Exec Time", time.time()-startt)
        print()
        
        # Clean Up
        handle.remove()
        model.cpu()
        del data

