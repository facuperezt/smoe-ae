from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import Asereje
from data import DataLoader
import argparse
from utils import flatten_dict, sum_nested_dicts


# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=None, type=str, help="Path to the model file")
args = parser.parse_args()

def set_title(ax: plt.Axes, title: str) -> None:
    plt.sca(ax)
    ax.set_title(title, fontsize=12)
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

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

with open(args.model_path, "rb") as f:  # If this file doesn't exist, checkout the "with_uploaded_model" branch
    model.load_state_dict(torch.load(f, map_location=device), strict=False)

model.eval()
# Iterate over the training dataset
for i, (inputs, labels) in enumerate(train_loader.get(data="valid")):
    print(i)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss = model.loss(outputs, labels, return_separate_losses=True)
    print(sum_nested_dicts(loss["e2e_loss"]).item(), sum_nested_dicts(loss["blockwise_loss"]).item())
    fig, axs = plt.subplots(3,2, figsize=(7,8))
    set_title(axs[0][0], "Original")
    model.visualize_output(inputs[0])
    set_title(axs[0][1], "Reconstructed")
    model.visualize_output(outputs[0])
    set_title(axs[1][0], "End-to-End L1 Loss")
    _l1 = loss["e2e_loss"]["l1_loss"].squeeze()
    model.visualize_output(_l1, vmin=_l1.min(), vmax=_l1.max())
    set_title(axs[1][1], "End-to-End L2 Loss")
    _l2 = loss["e2e_loss"]["l2_loss"].squeeze()
    model.visualize_output(_l2, vmin=_l2.min(), vmax=_l2.max())
    set_title(axs[2][0], "Blockwise L1 Loss")
    _l1 = loss["blockwise_loss"]["l1_loss"].squeeze()
    model.visualize_output(_l1, vmin=_l1.min(), vmax=_l1.max())
    set_title(axs[2][1], "Blockwise L2 Loss")
    _l2 = loss["blockwise_loss"]["l2_loss"].squeeze()
    model.visualize_output(_l2, vmin=_l2.min(), vmax=_l2.max())

    with open(f"data/visualizations/reconstruction_losses/A/with_blockwise_optimization/{i}.png", "wb") as f:
        fig.savefig(f, bbox_inches='tight')