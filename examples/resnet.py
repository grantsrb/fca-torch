import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import pandas as pd

from fca.fca import (
    PCAFunctionalComponentAnalysis
)
from fca.utils import (
    collect_activations_using_loader, get_command_line_args
)



if __name__ == "__main__":
    config = {
        "imagenet_path": '/data2/grantsrb/imagenet_mini-1000', # Set this to your ImageNet path
        "fca_layers": [ "layer4.2.conv2", ], # Specify the layers to collect activations from
        "batch_size": 64, # Batch size for training
        "val_batch_size": 64, # Batch size for validation
        "num_workers": 4, # Number of workers for DataLoader
    }

    config = {**config, **{get_command_line_args()}}  # Merge with command line args if any

    # Path to ImageNet (you must download it manually and set this path)
    imagenet_path = config["imagenet_path"]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet model (e.g., ResNet-50)
    model = models.resnet50(pretrained=True)
    model.eval().to(device)
    print(model)
    print("Training Proceeding in 3 seconds...")
    time.sleep(3)

    # ImageNet preprocessing (standard ImageNet normalization)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(
        root=f'{imagenet_path}/train',
        transform=transform
    )
    val_dataset = datasets.ImageFolder(
        root=f'{imagenet_path}/val',
        transform=transform
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("val_batch_size",config["batch_size"]),
        shuffle=False,
        num_workers=4
    )

    # Collect representations from the specified layers
    with torch.no_grad():
        outputs = collect_activations_using_loader(
            model=model,
            data_loader=train_loader,
            layers=config["fca_layers"],
            device=device,
            to_cpu=True,
        )

    # Build FCA Objects
    pc_fcas = []
    for layer in config["fca_layers"]:
        pc_fca = PCAFunctionalComponentAnalysis(
            X=outputs[layer],  # Use the outputs from the current layer
            center=True,      # Center the data
            scale=True,       # Scale the data
        )
        pc_fcas.append(pc_fca)

    comms_dict = {}
    handles = []
    for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
        handle = pc_fca.hook_model_layer(model, layer, comms_dict=comms_dict)
        handles.append(handle)

    # Track information for each rank
    metrics = {
        "rank": [],
        "layer": [],
        "trn_expl_var": [],
        "val_expl_var": [],
        "accuracy": [],
    }
    print("Tracking explained variance for each rank:")
    # Iterate over ranks from 1 to the maximum rank available in the outputs
    max_rank = min(outputs[layer].shape[-1] for layer in config["fca_layers"])
    for rank in range(1, max_rank + 1):
        print(f"Rank: {rank}")
        for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
            pc_fca.set_max_rank(rank)
            print(f"Layer: {layer}")
            expl_var = pc_fca.proportion_explained_variance(
                rank=rank,
                actvs=outputs[layer],
            )
            print(f"Explained Variance: {expl_var}")
        actvs = {pc_fca: [] for pc_fca in pc_fcas}
        avg_acc = 0.0
        n_loops = len(val_dataset) // config["val_batch_size"]
        for bi,batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            with torch.no_grad():
                logits = model(images)

            # Collect pre activations for each PCA layer
            for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
                actvs[pc_fca].append(comms_dict[pc_fca].cpu())

            preds = logits.argmax(dim=-1)
            acc = (preds==labels).float().mean().item()
            avg_acc += acc
            print(f"Acc: {acc:.4f}", f"{bi}/{n_loops}", end='\r')
        avg_acc /= n_loops
        print(f"Avg Acc: {avg_acc:.4f}")
        for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
            trn_expl_var = pc_fca.proportion_explained_variance()
            val_expl_var = pc_fca.proportion_explained_variance(
                rank=rank,
                actvs=torch.vstack(actvs[pc_fca]),
            )
            metrics["trn_expl_var"].append(trn_expl_var)
            metrics["val_expl_var"].append(val_expl_var)
            metrics["accuracy"].append(avg_acc)
            metrics["rank"].append(rank)
            metrics["layer"].append(layer)
            print(f"Layer: {layer}, Rank: {rank}, "
                  f"Train Expl Var: {trn_expl_var:.4f}, "
                  f"Val Expl Var: {val_expl_var:.4f}, "
                  f"Accuracy: {avg_acc:.4f}")
    df = pd.DataFrame(metrics)
    save_name = config.get("save_name", "resnet_fca_metrics.csv")
    df.to_csv(save_name, index=False, header=True)
    print("Metrics saved to resnet_fca_metrics.csv")
