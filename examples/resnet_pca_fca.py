import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import time
import pandas as pd
import os

from fca.fca import (
    PCAFunctionalComponentAnalysis
)
from fca.utils import (
    collect_activations_using_loader, get_command_line_args, fca_image_prep
)

def handle_config(config):
    """
    Handle configuration parameters, ensuring required ones are set.
    """
    if type(config["fca_layers"]) == str:
        config["fca_layers"] = [config["fca_layers"]]
    config["fca_layers"] = sorted([
        l.strip() for l in config["fca_layers"] if l.strip()
    ])
    return config

if __name__ == "__main__":
    overwrite = False  # Set to True to overwrite existing files
    config = {
        "imagenet_path": '/data2/grantsrb/imagenet-mini', # Set this to your ImageNet path
        "fca_layers": [ "layer1.0.conv2", ], # Specify the layers to collect activations from
        "batch_size": 224, # Batch size for training
        "val_batch_size": 224, # Batch size for validation
        "num_workers": 4, # Number of workers for DataLoader
    }

    config = {**config, **get_command_line_args()}  # Merge with command line args if any
    if "overwrite" in config:
        overwrite = config["overwrite"]

    # Error handling for required config parameters
    config = handle_config(config)

    # Path to ImageNet (you must download it manually and set this path)
    imagenet_path = config["imagenet_path"]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained ResNet model (e.g., ResNet-50)
    model = models.resnet18(pretrained=True)
    model.eval().to(device)
    print(model)

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
    if config.get("debugging", False):
        train_dataset = val_dataset

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

    print("Images:")
    images, labels = next(iter(train_loader))
    print("\tTrain:", images.shape)
    images, labels = next(iter(val_loader))
    print("\tValid:", images.shape)
    with torch.no_grad():
        output = model(images.to(device))
    print("Model Output:", output.shape)

    # Collect Full Rank Accuracy
    print("Collecting Validation Predictions")
    with torch.no_grad():
        acc_ceiling = 0.0
        n_loops = len(val_dataset) // config["val_batch_size"]
        for bi,batch in enumerate(val_loader):
            images, labels = batch
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=-1)
            acc = (preds==labels.to(device)).float().mean().item()
            acc_ceiling += acc
            print(f"Acc: {acc:.4f}", f"{bi}/{n_loops}", end='\r')
        acc_ceiling /= n_loops
        print(f"Avg Acc: {acc_ceiling:.4f}")

    # Collect train representations from the specified layers
    print("Collecting Train Activations")
    with torch.no_grad():
        outputs = collect_activations_using_loader(
            model=model,
            data_loader=train_loader,
            layers=config["fca_layers"],
            to_cpu=True,
            verbose=True,
        )

    # Build FCA Objects
    print("Building FCA Objects")
    pc_fcas = []
    for layer in config["fca_layers"]:
        X = fca_image_prep(outputs[layer])
        pc_fca = PCAFunctionalComponentAnalysis(
            X=X,
            center=True, # Center the data before PCA and FCA
            scale=True,  # Scale the data before PCA and FCA
        )
        pc_fca.to(device)
        pc_fcas.append(pc_fca)

    comms_dict = {}
    handles = []
    for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
        handle = pc_fca.hook_model_layer(
            model,
            layer,
            comms_dict=comms_dict,
            rep_type="images",
        )
        handles.append(handle)

    # Track information for each rank
    metrics = {
        "rank": [],
        "layer": [],
        "all_layers": [],
        "trn_expl_var": [],
        "val_expl_var": [],
        "accuracy": [],
        "recovered_accuracy": [],
    }
    print("Tracking explained variance for each rank:")
    # Iterate over ranks from 1 to the maximum rank available in the outputs
    max_rank = min(
        fca_image_prep(outputs[layer]).shape[-1] for layer in config["fca_layers"]
    )
    print("Max Rank:", max_rank)
    for rank in range(1, max_rank + 1):
        print(f"Rank: {rank}/{max_rank}")
        for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
            pc_fca.set_max_rank(rank)
            print(f"Layer: {layer}")
            samp = torch.randperm(outputs[layer].shape[0])[:1000].long()
            expl_var = pc_fca.proportion_expl_var(
                rank=rank,
                actvs=fca_image_prep(outputs[layer][samp].to(device)),
            ).mean().item()
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
            acc = (preds==labels.to(device)).float().mean().item()
            avg_acc += acc
            print(f"Acc: {acc:.4f}", f"{bi}/{n_loops}", end='\r')
        avg_acc /= n_loops
        recovered_acc = avg_acc / acc_ceiling
        print(f"Avg Acc: {avg_acc:.4f}/{acc_ceiling:.4f} = {recovered_acc:.4f}")
        for pc_fca, layer in zip(pc_fcas, config["fca_layers"]):
            with torch.no_grad():
                trn_expl_var = pc_fca.proportion_expl_var().mean().item()
                pc_fca.cpu()
                val_expl_var = pc_fca.proportion_expl_var(
                    rank=rank,
                    actvs=fca_image_prep(torch.vstack(actvs[pc_fca])),
                ).mean().item()
                pc_fca.to(device)
            metrics["trn_expl_var"].append(trn_expl_var)
            metrics["val_expl_var"].append(val_expl_var)
            metrics["recovered_accuracy"].append(recovered_acc)
            metrics["accuracy"].append(avg_acc)
            metrics["rank"].append(rank)
            metrics["layer"].append(layer)
            metrics["all_layers"].append(",".join(config["fca_layers"]))
            print(f"Layer: {layer}, Rank: {rank}, "
                  f"Train Expl Var: {trn_expl_var:.4f}, "
                  f"Val Expl Var: {val_expl_var:.4f}, "
                  f"Accuracy: {avg_acc:.4f}, "
                  f"Recovered Acc: {recovered_acc:.4f}")

    df = pd.DataFrame(metrics)
    if config.get("debugging", False):
        print(df.head())
    else:
        save_folder = config.get("save_folder", f"results")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        dtime = datetime.now().strftime("d%Y-%m-%d_t%H-%M-%S")
        save_name = os.path.join(
            save_folder,
            config.get("save_name", "resnet_fca_metrics_{dtime}.csv"),
        )
        print(f"Saving metrics to {save_name}")
        df.to_csv(save_name, index=False, header=True)
        print("Metrics saved to {save_name}")
