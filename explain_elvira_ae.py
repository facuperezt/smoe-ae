#%%
import json
import pickle
import time
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
#%%
# lp = LineProfiler()
# lp_wrapper = lp.add_function(Elvira2.forward)
# from models.elvira_2_ import TorchSMoE_SMoE
# lp_wrapper = lp.add_function(TorchSMoE_SMoE.torch_smoe)
# lp_wrapper = lp(Elvira2.embed_artifacts)
# lp_wrapper2 = lp(Elvira2.embed_artifacts_without_resize)

# %%
def load_model():
    cfg_file_path = "train_cfg/elvira_model.json"
    with open(cfg_file_path, "r") as f:
        cfg = json.load(f)

    # Device configuration
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        torch.mps.set_per_process_memory_fraction(0.)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Define your dataloader
    train_loader = DataLoader(cfg["data"]["path"])
    train_loader.initialize()

    # Define your model
    model = Elvira2(config_file_path=cfg["model"]["cfg_file_path"], device=device)
    model.eval()

    # Load your model checkpoint
    model_checkpoint_path = cfg.get("model", {}).get("checkpoint_path")
    if model_checkpoint_path is not None:
        print("Loading Checkpoint...")
        with open(model_checkpoint_path, "rb") as f:
            model.load_state_dict(torch.load(f, map_location=device), strict=False)
    
    return model

#%%

def plot(model, kernel_nr: int = 0):
    # # Load example image into right format
    # img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    # img_arr = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    # img_arr = img_arr[None, :512, :512]

    from zennit.composites import EpsilonGammaBox
    from zennit.canonizers import SequentialMergeBatchNorm
    from zennit.attribution import Gradient

    device = model.device
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)

    data = torch.load("utils/tmp_files/tmp_out.th").view(1, 1, 8, 8).to(device)/255
    _model = model.ae 
    _out = model.ae(data)

    from utils.visualize_kernels import plot_kernels, plot_kernel_centers
    clipped = model.clipper(_out)
    rec = model.smoe(clipped)
    # rec_img = model.blocks_to_img(rec)

    block_size = model.block_size
    padding = [2, 2, 2, 2]

    fig, axs = plt.subplots(3, 2)
    data: torch.Tensor

    # Input Image
    ax = axs[0][0]
    _data = torch.nn.functional.pad(data, padding, value=torch.nan)
    ax: plt.Axes
    ax.imshow(_data.squeeze().cpu(), vmin=0, vmax=1, cmap="gray")
    plot_kernel_centers(clipped.squeeze(), ax, block_size, padding, model.n_kernels, kernel_nr)
    plot_kernels(clipped.squeeze(), ax, block_size, padding, model.n_kernels, kernel_nr)
    ax.set_title("Original + Kernels")

    # Reconstructed Image
    ax = axs[0][1]
    rec = torch.nn.functional.pad(rec, padding, value=torch.nan)
    ax.imshow(rec.squeeze().detach().cpu(), vmin=0, vmax=1, cmap="gray")
    ax.set_title("Reconstruction")

    # X-Coordinate relevance
    ax = axs[1][0]
    out = torch.zeros_like(_out)
    out[:, kernel_nr] = _out[:, kernel_nr]
    with Gradient(model=_model, composite=composite) as attributor:
        _, x_rel = attributor(data, out)
    cbar_range = max(x_rel.min().abs(), x_rel.max().abs())
    cbar_range = [-cbar_range, cbar_range]
    x_rel = torch.nn.functional.pad(x_rel, padding, value=torch.nan)
    im = ax.imshow(x_rel.squeeze().detach().cpu(), cmap="bwr", vmin=cbar_range[0], vmax=cbar_range[1])
    ax.set_title("X-Coordinate relevance")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im , cax=cax, orientation='vertical')

    # Y-Coordinate relevance
    ax = axs[1][1]
    out = torch.zeros_like(_out)
    out[:, kernel_nr+model.n_kernels] = _out[:, kernel_nr+model.n_kernels]
    with Gradient(model=_model, composite=composite) as attributor:
        _, y_rel = attributor(data, out)
    cbar_range = max(y_rel.min().abs(), y_rel.max().abs())
    cbar_range = [-cbar_range, cbar_range]
    y_rel = torch.nn.functional.pad(y_rel, padding, value=torch.nan)
    im = ax.imshow(y_rel.squeeze().detach().cpu(), cmap="bwr", vmin=cbar_range[0], vmax=cbar_range[1])
    ax.set_title("Y-Coordinate relevance")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Combined Relevance
    ax = axs[2][0]
    out = torch.zeros_like(_out)
    out[:, kernel_nr] = _out[:, kernel_nr]
    out[:, kernel_nr+model.n_kernels] = _out[:, kernel_nr+model.n_kernels]
    out[:, kernel_nr+(2*model.n_kernels)] = _out[:, kernel_nr+(2*model.n_kernels)]
    with Gradient(model=_model, composite=composite) as attributor:
        _, combined_rel = attributor(data, out)
    cbar_range = max(combined_rel.min().abs(), combined_rel.max().abs())
    cbar_range = [-cbar_range, cbar_range]
    combined_rel = torch.nn.functional.pad(combined_rel, padding, value=torch.nan)
    im = ax.imshow(combined_rel.squeeze().detach().cpu(), cmap="bwr", vmin=cbar_range[0], vmax=cbar_range[1])
    ax.set_title("Combined relevance")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')

    # Expert relevance
    ax = axs[2][1]
    out = torch.zeros_like(_out)
    out[:, kernel_nr+(2*model.n_kernels)] = _out[:, kernel_nr+(2*model.n_kernels)]
    with Gradient(model=_model, composite=composite) as attributor:
        _, expert_rel = attributor(data, out)
    cbar_range = max(expert_rel.min().abs(), expert_rel.max().abs())
    cbar_range = [-cbar_range, cbar_range]
    expert_rel = torch.nn.functional.pad(expert_rel, padding, value=torch.nan)
    im = ax.imshow(expert_rel.squeeze().detach().cpu(), cmap="bwr", vmin=cbar_range[0], vmax=cbar_range[1])
    ax.set_title("Expert relevance")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')


    ylim, xlim = axs[0][0].get_ylim(), axs[0][0].get_xlim()
    for ax in axs.ravel():
        ax.vlines([-0.5+padding[0], -0.5+padding[0]+block_size], -0.5+padding[2], -0.5+padding[2]+block_size, color="tab:blue")
        ax.hlines([-0.5+padding[2], -0.5+padding[2]+block_size], -0.5+padding[0], -0.5+padding[0]+block_size, color="tab:blue")
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        ax.axis('off')
        ax.set_ylim(-1, padding[3]+block_size+1)
        ax.set_xlim(-1, padding[1]+block_size+1)
    plt.show()
    return out[:, 2*model.n_kernels:3*model.n_kernels].squeeze()
# %%
if __name__ == "__main__":
    model = load_model()
    plot(model, 0)
    plt.show()

# %%
