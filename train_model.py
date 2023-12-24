#%%
import json
import tqdm
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import wandb  # Add this import
from models import Asereje, AserejeOnlyE2E
import argparse
from data import DataLoader
from utils import flatten_dict, sum_nested_dicts
from torch.optim.lr_scheduler import ExponentialLR

# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file_path", default="train_cfg/default_cfg.json", type=str, help="Path to the training cfg file. It contains the model config file path.")
args, unknown = parser.parse_known_args()

with open(args.cfg_file_path, "r") as f:
    cfg = json.load(f)

# Data configs
data_path = cfg["data"]["path"]

# Model configs
model_cfg_file_path = cfg["model"]["cfg_file_path"]
model_checkpoint_path = cfg["model"]["checkpoint_path"]

# Initialize WandB
wandb.init(project="SMoE with funky encoder", name="With Block-wise Loss and BatchNorm", mode="online" if cfg["wandb"] else "disabled")


# Device configuration
if torch.backends.mps.is_available() and cfg["gpu"]:
    device = torch.device('mps')
elif torch.cuda.is_available() and cfg["gpu"]:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define your dataloader
train_loader = DataLoader(data_path)
train_loader.initialize()

# Define your model
model = AserejeOnlyE2E(model_cfg_file_path, device=device)
model: nn.Module
#%%
if model_checkpoint_path is not None:
    with open(model_checkpoint_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=False)

# Define your loss function
criterion = model.loss

# Define your optimizer
optimizer = optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])
num_epochs = 100
scheduler = ExponentialLR(optimizer, gamma=cfg["scheduler"]["gamma"])

historic_loss = []

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    nr_batches = len(train_loader.training_data)//cfg["batch_size"] if cfg["batch_size"] else 1
    # Iterate over the training dataset
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader.get(data="train", limit_to=None, batch_size=cfg["batch_size"])), total=nr_batches, desc=f"Epoch {epoch}"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        total_loss_mean = 0
        for _input, _label in tqdm.tqdm(zip(inputs, labels), total=len(inputs), desc=f"Batch {i}"):
            # Forward pass
            output = model(_input)

            # Compute the loss
            loss = criterion(output, _label, return_separate_losses=True)
            total_loss = sum_nested_dicts(loss)
            total_loss.backward()
            total_loss_mean += total_loss.item()/len(inputs)


        # Backward pass
        # make_dot(total_loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("full_backwards", format="pdf")
        # total_loss.backward()

        # Update the weights
        optimizer.step()
        print("Mean Loss: ", total_loss_mean)
        # Log the loss for this epoch to WandB
        wandb.log(flatten_dict({"Losses": loss, "Total Loss": total_loss, "Learning Rate": scheduler.get_last_lr()[0]}))

    scheduler.step()
    # Save the model if the loss is lower than the historic loss
    if not historic_loss or total_loss < historic_loss[-1]:
        # Save the trained model
        torch.save(model.state_dict(), f'models/saves/last_run/_epoch_{epoch}.pth')

    historic_loss.append(total_loss.item())
    if epoch > 25 and all([abs(new_loss) >= abs(old_loss) for new_loss, old_loss in zip(historic_loss[-5:], historic_loss[-6:-1])]):
        break

# Save the model at the end of training
torch.save(model.state_dict(), "models/saves/last_run/trained_model.pth")

# Finish the WandB run
wandb.finish()
