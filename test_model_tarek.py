#%%
# %load_ext autoreload
# %autoreload 2
import json
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models import Asereje
from models.tarek_model import TarekModel
from data import DataLoader
import argparse
from utils import flatten_dict, sum_nested_dicts, plot_kernels, plot_kernel_centers, get_gpu_memory_usage


from PIL import Image

from line_profiler import LineProfiler

from SinGAN.code.get_trained_discriminators import get_trained_discriminators

# Add argparse to load model from a path
parser = argparse.ArgumentParser()
parser.add_argument("--cfg_file_path", default="train_cfg/tarek_model.json", type=str, help="Path to the training cfg file. It contains the model config file path.")
parser.add_argument("--save", action="store_true", help="Saves the results")
args, unknown = parser.parse_known_args()

# lp = LineProfiler()
# lp_wrapper = lp.add_function(Elvira2.forward)
# from models.elvira_2_ import TorchSMoE_SMoE
# lp_wrapper = lp.add_function(TorchSMoE_SMoE.torch_smoe)
# lp_wrapper = lp(Elvira2.embed_artifacts)
# lp_wrapper2 = lp(Elvira2.embed_artifacts_without_resize)


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
model = TarekModel(config_file_path=cfg["model"]["cfg_file_path"], device=device)
model: nn.Module

# discriminator_loss = get_trained_discriminators("SinGAN/logs/fully_trained_rgb", device=device)

model_checkpoint_path = cfg.get("model", {}).get("checkpoint_path")
if model_checkpoint_path is not None:
    print("Loading Checkpoint...")
    with open(model_checkpoint_path, "rb") as f:
        model.load_state_dict(torch.load(f, map_location=device), strict=False)

model.train()
img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
img_arr = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)

# get random 512x512 rectangle within the image
x_min = np.random.randint(0, img_arr.shape[0] - 512)
y_min = np.random.randint(0, img_arr.shape[1] - 512)
img_arr = img_arr[x_min:x_min+512, y_min:y_min+512]

inp = torch.tensor(img_arr)
# inp = torch.tensor(inp, requires_grad=True)
inp_b = model.img_to_blocks(inp).reshape(-1 , 8, 8)
ae_o = model.ae(inp_b)
smoe_o = model.smoe(ae_o)
out = model.blocks_to_img(smoe_o)


out.sum().backward()
# print(inp.grad.sum())
plt.imshow(out.detach().cpu().squeeze(), cmap="gray")
#%%
from tarek_code.Code.load_model import load_model


t_img = pickle.load(open("tarek_code/pickles/baboon_ref.pckl", "rb"))["img_blocked"]
t_img = np.reshape(t_img, (-1, t_img.shape[2], t_img.shape[3]))
#%%
def get_tarek_model_description(i):
    return [a.name for a in load_model(i)[0].weights]

def get_res(img, i):
    def tarek(img, i):
        t_model, g = load_model(i)
        return t_model(img).numpy().transpose(0, 3, 1, 2)
    def mine(img, i):
        return model.ae(torch.tensor(img, dtype=torch.float32, device=model.device), i).detach().cpu().numpy()
    t, m = tarek(img, i), mine(img, 2*i)
    if t.shape == m.shape:
        return t, m
    else:
        print(t.shape)
        print(m.shape)
        return None, None

# %%
t, m = get_res(t_img, 2)
def plot_tm(t, m):
    ch, w, h = t.shape
    print(t.shape)
    # plot each channel side by side
    fig, axs = plt.subplots(ch, 2, figsize=(5, 30))
    for i in range(ch):
        axs[i, 0].imshow(t[i], cmap="gray")
        axs[i, 1].imshow(m[i], cmap="gray")
        axs[i, 0].set_ylabel(f"tarek {i}")
        axs[i, 1].set_ylabel(f"mine {i}")
        axs[i, 0].set_yticks([])
        axs[i, 1].set_yticks([])
        axs[i, 0].set_xticks([])
        axs[i, 1].set_xticks([])
    plt.show()

plot_tm(t[0], m[0])

# %%
