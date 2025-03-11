import torch
import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA, IncrementalPCA
import warnings
warnings.filterwarnings('ignore')

import dl_utils.save_io as savio
from modeling import NeuralNetwork
from tasks import HierarchicalLogicalTask
from fca import FunctionalComponentAnalysis, load_fcas
from utils import collect_activations

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
    fca.add_params_from_vector_list(pc_vecs)
    fca.freeze_parameters()
    fca.set_fixed(True)
    return fca
            

if __name__=="__main__":
    n_samples = 1000
    overwrite = False
    df_save_name = "pca_df.csv"
    model_folders = []
    for main_folder in sys.argv[1:]:
        model_folders += savio.get_model_folders(main_folder)
    for model_folder in model_folders:
        print("Running Functional PCA on", model_folder)

        # Load CheckPoint
        checkpoint = savio.load_checkpoint(model_folder)
        config = checkpoint["config"]
        
        # Load Model
        model = NeuralNetwork(**config['model_params'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.freeze_parameters()
        model.eval()
        
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
        fca_path = fca_files[-1] # Pick the most recent. The others will be automatically loaded as well.
        checkpt = torch.load(fca_path)
        fca_layers = checkpt["fca_layers"][:1]
        ortho_ensured = checkpt["config"]["ensure_ortho_chain"]
        print("Orthogonality Ensured:", ortho_ensured)
        
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
        X = X.numpy()
        X = (X-X.mean(0))/(X.std(0)+1e-5)
        print(X.shape)
        print("Performing PCA")
        pca = PCA()
        pca.fit(X)
        pc_vecs = pca.transform(X)
        print("Total Dims:", X.shape[1])
        print(
            "Explained Variance",
            [round(x, 3) for x in pca.explained_variance_ratio_])
        print(
            "Dims to 95%:",
            find_dims_to_sum(
                [round(x, 3) for x in pca.explained_variance_ratio_]
                ,0.95
            )
        )
        print(
            "Dims to 99%:",
            find_dims_to_sum(
                [round(x, 3) for x in pca.explained_variance_ratio_]
                ,0.99
            )
        )
        print(
            "Dims to 99.5%:",
            find_dims_to_sum(
                [round(x, 3) for x in pca.explained_variance_ratio_],
                0.995
            )
        )
        
        pcs = pca.components_
        
        dim_data = {
            "n_dims": [],
            "acc": [],
        }
        for n_dims in reversed(range(1,len(pcs)+1)):
            fca = make_fca_from_pca(torch.Tensor(pcs).float()[:n_dims])
            for layer,modu in model.named_modules():
                if layer == fca_layers[0]:
                    handle = modu.register_forward_hook(fca.get_forward_hook())
                    
            with torch.no_grad():
                actvs = collect_activations(
                    model,
                    input_data=data["input_ids"],
                    layers=[],
                    ret_pred_ids=True,
                    batch_size=None,
                    to_cpu=True,
                    verbose=False,
                )
            handle.remove()
            acc = (actvs["pred_ids"]==data["output_ids"].squeeze()).float().mean().item()
            dim_data["n_dims"].append(n_dims)
            dim_data["acc"].append(acc)
            handle.remove()
        df = pd.DataFrame(dim_data)
        csv_path = os.path.join(model_folder, "functional_pca.csv")
        df.to_csv(csv_path, index=False, header=True)
        
        
        
        
        