#%%
import os
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from data import DataLoader
from utils.visualize_kernels import plot_kernels, plot_kernel_centers

from PIL import Image

from models.facu import VAE, SimpleMLP, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts
from models.elvira import Vanilla as AE

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

n_kernels = 4
block_size = 8
img_size = 128

def load_model(who: str) -> torch.nn.Module:
    if who == "elvira":
        model = AE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device, force_conv_layers=[16, 32, 64], force_dense_layers=[64])
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\elvira_small\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "mlp":
        model = SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device, force_hidden_sizes=[4*block_size**2, 8*block_size**2, 4*block_size**2, block_size**2])
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\mlp_big\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "vae":
        model = VAE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device)
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\vae\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "vae_kernels_inside":
        model = VAE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device)
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\vae_kernels_inside\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "vae_kernels_outside":
        model = VAE_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device)
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\vae_kernels_outside\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "vae_negative_experts":
        model = VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device)
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\vae_negative_experts\\final.pth"
        model.load_state_dict(torch.load(path))
    elif who == "vae_kernels_outside_negative_experts":
        model = VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device)
        path = "C:\\Users\\fq\\Facu\\clean-smoe\\models\\facu\\checkpoints\\vae_kernels_outside_negative_experts\\final.pth"
        model.load_state_dict(torch.load(path))
    else:
        raise ValueError(f"Who is {who}?")
    return model

def visualize_output(model, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1., ax: plt.Axes = None) -> None:
    if blocked_output.squeeze().ndim > 2:
        img = model.block2img(blocked_output).cpu().detach().numpy()
    else:
        img = blocked_output.squeeze().cpu().detach().numpy()
    if ax is None:
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.show()
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

if __name__ == "__main__":
    for model_name in ["vae_kernels_inside", "vae_kernels_outside", "vae_negative_experts", "vae_kernels_outside_negative_experts"]:
        model = load_model(model_name)
        os.makedirs(f"models/facu/kernel_pics/{model_name}/", exist_ok=True)
        model.eval()
        img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
        img_arr = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
        img_arr = img_arr[None, :img_size, :img_size]
        blocks = model.img2block(img_arr)
        emb = model.encoder(blocks)
        if model_name == "elvira_small":
            emb = model.clipper(emb)
        elif "vae" in model_name:
            emb = emb[0]
        out = model.decoder(emb).cpu()
        emb = emb.cpu()
        blocks = blocks.cpu()
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
        for i, (emb_i, block_i, out_i) in enumerate(zip(emb, blocks.squeeze(), out)):
            plt.close()
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"{model_name}")
            ax = axs[1]
            # add padding around out_i
            padding = [5, 5, 5, 5]
            out_i = torch.nn.functional.pad(out_i, padding, value=1)
            visualize_output(model, out_i, ax=ax[0])
            plot_kernel_centers(emb_i, ax=ax[0], block_size=block_size, padding=padding, colors=colors)
            plot_kernels(emb_i, ax=ax[0], block_size=block_size, padding=padding, colors=colors)
            ax[0].axis("off")
            
            block_i = torch.nn.functional.pad(block_i, padding, value=1)
            visualize_output(model, block_i, ax=ax[1])
            plot_kernel_centers(emb_i, ax=ax[1], block_size=block_size, padding=padding, colors=colors)
            plot_kernels(emb_i, ax=ax[1], block_size=block_size, padding=padding, colors=colors)
            ax[1].axis("off")

            ax = axs[0]
            # out_i = torch.nn.functional.pad(out_i, padding, value=1)
            visualize_output(model, out_i, ax=ax[0])
            ax[0].axis("off")

            # block_i = torch.nn.functional.pad(block_i, padding, value=1)
            visualize_output(model, block_i, ax=ax[1])
            ax[1].axis("off")
            plt.show()
            # plt.savefig(f"models/facu/kernel_pics/{model_name}/{i}_block.png")
