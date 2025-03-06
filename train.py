import torch
from modeling import NeuralNetwork
from tasks import HierarchicalLogicalTask, OPERATION2STRING
import torch.optim as optim
from fca import FunctionalComponentAnalysis
import pandas as pd

config = {
    'task_params': {
        'n_pairs': 4,
        'n_samples': 1000,
    },
    'model_params': {
        'embedding_dim': 32,
        'd_model': 512,
        'n_layers': 3,
        'nonlinearity': torch.nn.ReLU,
        'lnorm': True,
    },
    'lr': 0.001,
    'num_epochs': 10000,
    'batch_size': 128,
    "patience": 20,
    "plateau": 0.01,
    'model_save_path': 'model.pth',
    'model_load_path': "model.pth",
    'fca_load_path': None,

    'fca_params': {"max_rank": 512},
    'fca_layers': ["hidden_layers.0"],
    'persistent_keys': [
        'fca_params', 'fca_layers', 'lr',
        "batch_size", "model_save_path",
        "num_epochs", "fca_load_path",
    ],
}

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
        checkpoint = torch.load(config['model_load_path'])
        conf = checkpoint['config']
        for k in config.get("persistent_keys", []):
            if k in config:
                conf[k] = config[k]
        config = conf

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

    # Load trained Functional Component Analysis Objects
    if config.get("fca_load_path", None) is not None:
        model.freeze_parameters()
        fca_checkpoint = torch.load(config["fca_load_path"])
        state_dicts = fca_checkpoint["fca_state_dicts"]
        fca_config = fca_checkpoint["config"]
        kwargs = fca_config.get("fca_params", {})
        loaded_fcas = {}
        loaded_handles = []
        for name,modu in model.named_modules():
            if name in fca_config["fca_layers"]:
                kwargs["size"] = modu.weight.shape[-1]
                kwargs["remove_components"] = True
                loaded_fcas[name] = FunctionalComponentAnalysis(
                    **kwargs
                )
                loaded_fcas[name].load_sd(state_dicts[name])
                loaded_fcas[name].update_parameters()
                loaded_fcas[name].freeze_parameters()
                h = modu.register_forward_hook(
                    loaded_fcas[name].get_forward_hook()
                )
                loaded_handles.append(h)

    # Get the initial loss and accuracy
    with torch.no_grad():
        model.eval()
        total_loss = 0.0
        total_acc = 0.0
        for batch in task.get_batches(config['batch_size']):
            inputs = batch["input_ids"]
            labels = batch["output_ids"].squeeze()

            outputs = model(inputs.to(device))
            preds = outputs.argmax(dim=-1)
            #for j in range(3):
            #    print("Input:", " ".join([task.idx2word[int(i)] for i in inputs[j]]))
            #    print("Label:", int(labels[j]))
            #    print("Preds:", int(preds[j]))
            #    print()
            loss = criterion(outputs, labels.to(device))
            acc = (preds == labels.to(device)).float().mean()
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

        data_dict["epoch"].append(0)
        data_dict["train_loss"].append(total_loss/task.n_batches(config["batch_size"]))
        data_dict["train_acc"].append(total_acc/task.n_batches(config["batch_size"]))
        data_dict["val_loss"].append(total_loss_val/val_task.n_batches(config["batch_size"]))
        data_dict["val_acc"].append(total_acc_val/val_task.n_batches(config["batch_size"]))

    # Initialize the Functional Component Analysis Objects
    fcas = {}
    fca_handles = []
    fca_layers = config.get("fca_layers", None)
    if fca_layers is not None and fca_layers:
        model.freeze_parameters()
        fca_parameters = []
        kwargs = config.get("fca_params", {})
        for name,modu in model.named_modules():
            if name in config["fca_layers"]:
                kwargs["size"] = modu.weight.shape[-1]
                fcas[name] = FunctionalComponentAnalysis(
                    **kwargs
                )
                h = modu.register_forward_hook(
                    fcas[name].get_forward_hook()
                )
                fca_handles.append(h)
                fca_parameters += list(fcas[name].parameters())
        optimizer = optim.Adam(fca_parameters, lr=config['lr'])

    # Training loop
    for epoch in range(1,config['num_epochs']):
        model.train()
        total_loss = 0.0
        total_acc = 0.0

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

        data_dict["epoch"].append(epoch)
        bsize = config['batch_size']
        loss = total_loss/task.n_batches(bsize)
        acc = total_acc/task.n_batches(bsize)
        data_dict["train_loss"].append(loss)
        data_dict["train_acc"].append(acc)
        print(f'Epoch {epoch}/{config["num_epochs"]}')
        print(f"Trn Loss: {loss}, Trn Acc: {acc}")
        loss = total_loss_val/val_task.n_batches(bsize)
        acc = total_acc_val/val_task.n_batches(bsize)
        data_dict["val_loss"].append(loss)  # val_loss
        data_dict["val_acc"].append(acc)  # val_acc
        print(f'Val Loss: {loss}, Val Acc: {acc}')
        print("Ops:", ", ".join([OPERATION2STRING[o] for o in task.operations]))
        if fcas:
            print("FCA Parameters:")
            for layer,fca in fcas.items():
                print(f"{layer}: {fca.rank}")
        print('---')

        if plateau_tracker.update(loss, acc):
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
                    print("Early stopping")
                    break
                optimizer = optim.Adam(new_axes, lr=config['lr'])
                plateau_tracker.reset()
            else:
                print("Early stopping")
                break


    # Save the trained model
    save_path = config['model_save_path']
    if fcas:
        if config.get("fca_load_path", None) is not None:
            save_path = save_path.replace(".pth", "_chained_fca.pth")
        else:
            save_path = save_path.replace(".pth", "_fca.pth")
        save_dict = {
            "fca_state_dicts": {
                layer: fca.state_dict() for layer,fca in fcas.items()
            },
            "config": config,
        }
        for h in fca_handles:
            h.remove()
    else:
        save_dict = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }
    print("Saving model to:", save_path)
    torch.save(save_dict, save_path)
    # Save the training data
    df = pd.DataFrame(data_dict)
    save_path = save_path.replace(".pth", ".csv")
    df.to_csv(save_path, index=False, header=True)

if __name__ == "__main__":
    train(config)