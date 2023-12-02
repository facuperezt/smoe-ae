from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import Asereje
from data import DataLoader

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

with open("models/saved_models/first_results.pth", "rb") as f:  # If this file doesn't exist, checkout the "with_uploaded_model" branch
    model.load_state_dict(torch.load(f, map_location=device), strict=False)

model.eval()
# Iterate over the training dataset
for i, (inputs, labels) in enumerate(train_loader.get(data="valid")):
    print(i)
    inputs = inputs.to(device)
    labels = labels.to(device)
    outputs = model(inputs)
    loss_end, loss_blocks = model.loss(outputs, labels, return_separate_losses=True)
    print(loss_end.sum().item(), loss_blocks.sum().item())
    fig, axs = plt.subplots(2,2, figsize=(7,8))
    set_title(axs[0][0], "Original")
    model.visualize_output(inputs[0])
    set_title(axs[0][1], "Reconstructed")
    model.visualize_output(outputs[0])
    set_title(axs[1][0], "End-to-End Loss")
    model.visualize_output(loss_end[0], vmin=loss_end.min(), vmax=loss_end.max())
    set_title(axs[1][1], "Blockwise Loss")
    model.visualize_output(loss_blocks, vmin=loss_blocks.min(), vmax=loss_blocks.max())
    with open(f"data/visualizations/reconstruction_losses/A/no_blockwise_optimization/{i}.png", "wb") as f:
        fig.savefig(f, bbox_inches='tight')