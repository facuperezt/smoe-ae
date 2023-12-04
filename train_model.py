#%%
import torch
import torch.nn as nn
import torch.optim as optim
import wandb  # Add this import
from models import Asereje
import argparse
from data import DataLoader
from utils import flatten_dict, sum_nested_dicts
from torch.optim.lr_scheduler import ExponentialLR
from torchviz import make_dot

# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str, help="Path to the model file")
parser.add_argument("--lr", default=0.001, type=float, help="Learning rate for optimizer")
args, unknown = parser.parse_known_args()

# Initialize WandB
wandb.init(project="SMoE with funky encoder", name="With Block-wise Loss and BatchNorm", mode="online")
# wandb.define_metric("Losses", summary="mean")
# define a metric we are interested in the minimum of
wandb.define_metric("Total Loss", summary="min")


# Device configuration
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Define your dataloader
train_loader = DataLoader()
train_loader.initialize()

# Define your model
model = Asereje("models/config_files/base_config.json", device=device)
model: nn.Module

if args.model_path is not None:
    with open(args.model_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=False)

# Define your loss function
criterion = model.loss

# Define your optimizer
optimizer = optim.AdamW(model.parameters(), lr=args.lr)
num_epochs = 100
scheduler = ExponentialLR(optimizer, gamma=0.9)

historic_loss = []

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()

    # Iterate over the training dataset
    for i, (inputs, labels) in enumerate(train_loader.get(data="train", limit_to=None, batch_size=5)):
        print(i)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        total_loss = 0
        for _input, _label in zip(inputs, labels):
            # Forward pass
            output = model(_input)

            # Compute the loss
            loss = criterion(output, _label, return_separate_losses=True)
            total_loss = sum_nested_dicts(loss)
            total_loss.backward()


        # Backward pass
        # make_dot(total_loss, params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render("full_backwards", format="pdf")
        # total_loss.backward()

        # Update the weights
        optimizer.step()

        # Log the loss for this epoch to WandB
        wandb.log(flatten_dict({"Losses": loss, "Total Loss": total_loss}))

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
