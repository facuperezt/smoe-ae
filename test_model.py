#%%
import json
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Asereje, Elvira2
from data import DataLoader
import argparse
from utils import flatten_dict, sum_nested_dicts, plot_kernels, plot_kernel_centers, get_gpu_memory_usage

from PIL import Image

from line_profiler import LineProfiler

from SinGAN.code.get_trained_discriminators import get_trained_discriminators

# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file_path", default="train_cfg/elvira_model.json", type=str, help="Path to the training cfg file. It contains the model config file path.")
parser.add_argument("--save", action="store_true", help="Saves the results")
args, unknown = parser.parse_known_args()

lp = LineProfiler()
lp_wrapper = lp.add_function(Elvira2.forward)
from models.elvira_2_ import TorchSMoE_SMoE
lp_wrapper = lp.add_function(TorchSMoE_SMoE.torch_smoe)
lp_wrapper = lp(Elvira2.embed_artifacts)
lp_wrapper2 = lp(Elvira2.embed_artifacts_without_resize)


with open(args.cfg_file_path, "r") as f:
    cfg = json.load(f)

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
    torch.mps.set_per_process_memory_fraction(0.)
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# device = torch.device('cpu')
# Define your dataloader
train_loader = DataLoader(cfg["data"]["path"])
train_loader.initialize()

# Define your model
model = Elvira2(config_file_path=cfg["model"]["cfg_file_path"], device=device)
model: nn.Module

# discriminator_loss = get_trained_discriminators("SinGAN/logs/fully_trained_rgb", device=device)

model_checkpoint_path = cfg.get("model", {}).get("checkpoint_path")
if model_checkpoint_path is not None:
    print("Loading Checkpoint...")
    with open(model_checkpoint_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=False)

model.eval()
img = Image.open("data/professional_photos/train/ali-inay-2273.png")
img_arr = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
img_arr = img_arr[None, :512, :512, :].permute(0, 3, 1, 2)
# with torch.no_grad():
#     # start = time.time()
#     # out = model.embed_artifacts(img_arr)
#     # print(f"Time for normal out: {time.time()-start}")
#     out = lp_wrapper(model, img_arr)
#     out2 = lp_wrapper2(model, img_arr)
# model.visualize_output(out.cpu())
# lp.print_stats()
# print(out.shape)
# img_arr = torch.nn.functional.interpolate(img_arr, size=(512, 512), mode="bilinear", align_corners=True)
compressed = model.embed_artifacts_without_resize(img_arr)
# l = discriminator_loss.forward(compressed, img_arr)
blocks = model.img_to_blocks(img_arr)
emb = model.clipper(model.ae(blocks))
print(emb.mean(dim=0))
out = model.smoe(emb).cpu()
emb = emb.cpu()
blocks = blocks.cpu()
for emb_i, block_i, out_i in zip(emb, blocks.squeeze(), out):
    plt.close()
    fig, ax = plt.subplots(1, 2)
    # add padding around out_i
    padding = [5, 5, 5, 5]
    out_i = torch.nn.functional.pad(out_i, padding, value=0)
    model.visualize_output(out_i, ax=ax[0])
    plot_kernel_centers(emb_i, ax=ax[0], block_size=model.block_size, padding=padding)
    plot_kernels(emb_i, ax=ax[0], block_size=model.block_size, padding=padding)
    block_i = torch.nn.functional.pad(block_i, padding, value=0)
    model.visualize_output(block_i, ax=ax[1])
    plot_kernel_centers(emb_i, ax=ax[1], block_size=model.block_size, padding=padding)
    plot_kernels(emb_i, ax=ax[1], block_size=model.block_size, padding=padding)
    plt.show()

exit()
# Iterate over the training dataset
for i, (inputs, labels) in enumerate(train_loader.get(data="valid", limit_to=5)):
    print(i)
    inputs = inputs.to(device)
    labels = labels.to(device)
    blocks = model.img_to_blocks(inputs).cpu()
    emb = model.clipper(model.ae(blocks)).cpu()
    for emb_i, block_i in zip(emb, blocks.squeeze()):
        plt.close()
        fig, ax = plt.subplots(1, 2)
        model.visualize_output(block_i, ax=ax[0])
        plot_kernel_centers(emb_i, ax=ax[0])
        plot_kernels(emb_i, ax=ax[0])
        model.visualize_output(emb_i, ax=ax[1])
    outputs = model(inputs)
    loss = model.loss(outputs, labels, return_separate_losses=True)
    # print(sum_nested_dicts(loss["e2e_loss"]).item(), sum_nested_dicts(loss["blockwise_loss"]).item())
    fig, axs = plt.subplots(3,2, figsize=(7,8))
    set_title(axs[0][0], "Original")
    model.visualize_output(inputs[0])
    set_title(axs[0][1], "Reconstructed")
    model.visualize_output(outputs)
    set_title(axs[1][0], "End-to-End L1 Loss")
    _l1 = loss["e2e_loss"]["l1_loss"].squeeze()
    model.visualize_output(_l1, vmin=_l1.min(), vmax=_l1.max())
    set_title(axs[1][1], "End-to-End L2 Loss")
    _l2 = loss["e2e_loss"]["l2_loss"].squeeze()
    model.visualize_output(_l2, vmin=_l2.min(), vmax=_l2.max())
    # set_title(axs[2][0], "Blockwise L1 Loss")
    # _l1 = loss["blockwise_loss"]["l1_loss"].squeeze()
    # model.visualize_output(_l1, vmin=_l1.min(), vmax=_l1.max())
    # set_title(axs[2][1], "Blockwise L2 Loss")
    # _l2 = loss["blockwise_loss"]["l2_loss"].squeeze()
    # model.visualize_output(_l2, vmin=_l2.min(), vmax=_l2.max())

    if args.save:
        with open(f"data/visualizations/reconstruction_losses/A/with_blockwise_optimization/{i}.png", "wb") as f:
            fig.savefig(f, bbox_inches='tight')
    else:
        plt.show()
# %%
