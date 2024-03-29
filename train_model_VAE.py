#%%
import json
import re
import tqdm
from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
import wandb  # Add this import
from models import Asereje, BserejePipeline as Bsereje
import argparse
from data import DataLoader
from utils import flatten_dict, sum_nested_dicts, CosineAnnealingWarmupRestarts
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR, ChainedScheduler
import os

# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file_path", default="train_cfg/train_basic_vae.json", type=str, help="Path to the training cfg file. It contains the model config file path.")
args, unknown = parser.parse_known_args()

with open(args.cfg_file_path, "r") as f:
    cfg = json.load(f)

# Data configs
data_path = cfg["data"]["path"]

# Model configs
model_cfg_file_path = cfg["model"]["cfg_file_path"]
model_checkpoint_path = cfg["model"]["checkpoint_path"]

# Device configuration
if torch.backends.mps.is_available() and cfg["gpu"]:
    device = torch.device('mps')
elif torch.cuda.is_available() and cfg["gpu"]:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define your dataloader
train_loader = DataLoader(data_path)
train_loader.initialize(n_repeats=1, force_reinitialize=True)

# Define your model
model = Bsereje(model_cfg_file_path, device=device)
model: nn.Module

if model_checkpoint_path is not None:
    with open(model_checkpoint_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=False)

# Define your loss function
criterion = model.loss

# Define your optimizer
optimizer = optim.AdamW(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

scheduler = ExponentialLR(optimizer, gamma=cfg["scheduler"]["gamma"])
# scheduler = CosineAnnealingWarmupRestarts(
#         optimizer=optimizer,
#         first_cycle_steps=cfg["scheduler"]["first_cycle_steps"],
#         cycle_mult=cfg["scheduler"]["cycle_mult"],
#         max_lr=cfg["optimizer"]["lr"],
#         min_lr=cfg["scheduler"]["min_lr"],
#         warmup_steps=cfg["scheduler"]["warmup_steps"],
#         gamma=cfg["scheduler"]["gamma"]
#     )
log_cfg = {
    **model.cfg,
    "optimizer": cfg["optimizer"],
    "scheduler": cfg["scheduler"],
    }

# Initialize WandB
wandb.init(project="SMoE with VAE", name="Using KL-Div - No Batchnorm - No Dropout", config=log_cfg, mode="online" if cfg["wandb"] else "disabled")
#%%
historic_loss = []

# Function to get the next count for the folder name
def get_next_count(parent_dir, base_name):
    # Regular expression to match folders with the base name followed by a count
    pattern = re.compile(f'^{base_name}_(\\d+)$')
    max_count = 0
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)):
            match = pattern.match(item)
            if match:
                # Extract the count number and update max_count if higher
                count = int(match.group(1))
                if count > max_count:
                    max_count = count
    return max_count + 1

# Check if the folder exists
if os.path.exists('models/saves/last_run'):
    parent_dir = os.path.dirname('models/saves/last_run')
    # Get the next count number for the new folder name
    next_count = get_next_count(parent_dir, 'archived_run')
    # Create a new name for the folder by appending the next count
    new_folder_name = f"archived_run_{next_count}"
    # Rename the folder
    os.rename('models/saves/last_run', os.path.join(parent_dir, new_folder_name))
os.mkdir("models/saves/last_run/")

# Training loop
for epoch in range(cfg["num_epochs"]):
    # Set model to training mode
    model.train()

    nr_batches = len(train_loader.training_data)//cfg["batch_size"] if cfg["batch_size"] else 1
    # Iterate over the training dataset
    for i, (inputs, labels) in tqdm.tqdm(enumerate(train_loader.get(data="train", limit_to=None, batch_size=cfg["batch_size"])), total=nr_batches, desc=f"Epoch {epoch}"):
        inputs = inputs.to(device)
        labels = labels.to(device)

        batch_size = inputs.shape[0]

        # Zero the gradients
        optimizer.zero_grad()

        mean_total_loss = torch.tensor([0], dtype= torch.float32, device=device)
        mean_rec_loss = torch.tensor([0], dtype=torch.float32, device=device)
        mean_kl_div = torch.tensor([0], dtype=torch.float32, device=device)
        for _input, _label in tqdm.tqdm(zip(inputs, labels), total=len(inputs), desc=f"Batch {i}"):
            # Forward pass
            output, z_mean, z_logvar = model(_input)

            # Compute the loss
            loss, rec_loss, kl_div = criterion(output, _label, z_mean, z_logvar)
            
            loss = torch.div(loss, batch_size)
            rec_loss = torch.div(rec_loss, batch_size)
            kl_div = torch.div(kl_div, batch_size)

            loss.backward()

            mean_total_loss += loss.detach()
            mean_rec_loss += rec_loss
            mean_kl_div += kl_div
        if not mean_total_loss.abs() + mean_rec_loss.abs() + mean_kl_div.abs() > 0:
            """
            There's something slightly off with the batching which creates an empty batch, so it has loss 0 every time
            just skip it since loss 0 wont be reached anyway.
            """
            continue
        # Log the loss for this batch to WandB
        wandb.log(flatten_dict({"Reconstruction Loss": mean_rec_loss, "KL Loss": mean_kl_div, "Total Loss": mean_total_loss, "Learning Rate": scheduler.get_last_lr()[0]}))
        # Update the weights
        optimizer.step()
        scheduler.step()
        model.step()
        print("Mean Loss: ", mean_total_loss, "Mean Rec Loss: ", mean_rec_loss, "Mean KL Loss: ", mean_kl_div)


    # Save the model if the loss is lower than the historic loss
    if not historic_loss or mean_total_loss < historic_loss[-1]:
        # Save the trained model
        torch.save(model.state_dict(), f'models/saves/last_run/_epoch_{epoch}.pth')

    historic_loss.append(mean_total_loss)
    if epoch > 5000 and all([abs(new_loss) >= abs(old_loss) for new_loss, old_loss in zip(historic_loss[-5:], historic_loss[-6:-1])]):
        break

# Save the model at the end of training
torch.save(model.state_dict(), "models/saves/last_run/trained_model.pth")

# Finish the WandB run
wandb.finish()
