#%%
%load_ext autoreload
%autoreload 2
import os
import time
from typing import Dict
from matplotlib import pyplot as plt
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import *
from models.elvira import Vanilla
from models.facu.variational_abstract import VAE_Abstract
import wandb

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from pycolormap_2d import ColorMap2DZiegler

class EarlyStoppingScoreHistory:
    def __init__(self, memory_size=10):
        self.last_scores = []   
        self.memory_size = memory_size

    def __getitem__(self, idx):
        return self.last_scores[idx]
    
    def __setitem__(self, idx, value):
        self.last_scores[idx] = value

    def append(self, value):
        self.last_scores.append(value)
        self.last_scores = self.last_scores[-self.memory_size:]

    def __len__(self):
        return len(self.last_scores)
    
    def __iter__(self):
        return iter(self.last_scores)
    
    def __str__(self):
        return str(self.last_scores)
    
    def __repr__(self):
        return repr(self.last_scores)
    
    def __contains__(self, item):
        return item in self.last_scores
    
    def __add__(self, other):
        return self.last_scores + other
    
    def __radd__(self, other):
        return other + self.last_scores
    
    def __iadd__(self, other):
        self.last_scores += other

    def get_trend(self):
        diff = [self.last_scores[i] - self.last_scores[i-1] for i in range(1, len(self.last_scores))]
        return sum(diff)/len(diff)

class KLD_Weight_CosineAnnealing:
    def __init__(self, min_val, max_val, cycle_length, warmup: int = 50, wait_to_start: int = 50):
        self.start_weight = max_val
        self.end_weight = min_val
        self.nr_epochs = cycle_length
        self.current_epoch = 0
        self.warmup = warmup
        self.wait = wait_to_start

    def __call__(self):
        self.current_epoch += 1
        if self.current_epoch < self.wait:
            return 0.0
        elif self.current_epoch < self.warmup + self.wait:
            return ((self.current_epoch - self.wait)/self.warmup) * self.start_weight
        return self.start_weight + (self.end_weight - self.start_weight) * (1 + torch.cos(torch.tensor(self.current_epoch/self.nr_epochs) * 3.141592653589793))/2

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, memory_size=20, trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.trace_func = trace_func
        self.last_scores = EarlyStoppingScoreHistory(memory_size=memory_size)

    def __call__(self, val_loss, model, path):
        score = -val_loss
        self.last_scores.append(score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and self.last_scores.get_trend() >= -self.delta:
                self.early_stop = True
            elif self.counter >= self.patience and self.last_scores.get_trend() < -self.delta:
                self.trace_func("Early Stopping prevented because of downward trend on error.")
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
            
    def save_checkpoint(self, val_loss, model, path = None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if path is None:
            path = self.path
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.show()

#%%
def plot_latent_space_conv(model: CAE_Codec):
    data = train_loader.get_m_blocks_with_n_kernels(batch_size, n_kernels, kernels_outside="kernelsoutside" in str(model.__class__).lower(), 
                                            negative_experts="negativeexperts" in str(model.__class__).lower(), device=device)
    
    orig_x = data[:, 0*n_kernels:1*n_kernels].flatten().cpu().numpy()
    orig_y = data[:, 1*n_kernels:2*n_kernels].flatten().cpu().numpy()
    orig_nu = data[:, 2*n_kernels:3*n_kernels].flatten().cpu().numpy()
    
    blocks = model.decoder(data)[:, None]
    with torch.no_grad():
        recon = model.encoder.encode(blocks).detach().cpu().numpy()

    x  = recon[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon[:, 2*n_kernels:3*n_kernels].flatten()

    _cmap = ColorMap2DZiegler((-0.5, 1.5), (-0.5, 1.5))
    cmap = [_cmap(a, b)/255 for a, b in zip(orig_x, orig_y)]

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(f"Latent space")
    a = axs[0].scatter(orig_x, orig_y, c=cmap, cmap="jet", alpha=0.5)
    axs[0].set_title("Original")
    a = axs[1].scatter(x, y, c=cmap, cmap="jet", alpha=0.5)
    axs[1].set_title("Reconstructed")
    plt.colorbar(a)


def plot_latent_space_variational(model: VAE_Abstract):
    data = train_loader.get_m_blocks_with_n_kernels(batch_size, n_kernels, kernels_outside="kernelsoutside" in str(model.__class__).lower(), 
                                            negative_experts="negativeexperts" in str(model.__class__).lower(), device=device)
    
    orig_x = data[:, 0*n_kernels:1*n_kernels].flatten().cpu().numpy()
    orig_y = data[:, 1*n_kernels:2*n_kernels].flatten().cpu().numpy()
    orig_nu = data[:, 2*n_kernels:3*n_kernels].flatten().cpu().numpy()
    
    blocks = model.decoder(data)[:, None]
    with torch.no_grad():
        recon = model.encoder.encode(blocks)

    if len(recon) == 2:
        recon_mu, recon_s = [a.detach().cpu().numpy() for a in recon]

    _cmap = ColorMap2DZiegler((-0.5, 1.5), (-0.5, 1.5))
    cmap = np.array([_cmap(a, b)/255 for a, b in zip(orig_x, orig_y)])

    x  = recon_mu[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon_mu[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon_mu[:, 2*n_kernels:3*n_kernels].flatten()

    xs = recon_s[:, 0*n_kernels:1*n_kernels].flatten()
    ys = recon_s[:, 1*n_kernels:2*n_kernels].flatten()
    nus = recon_s[:, 2*n_kernels:3*n_kernels].flatten()


    variance_x = np.exp(xs)
    variance_y = np.exp(ys)
    variance_xy = variance_x + variance_y

    # sort x, y and variance_xy by variance
    idx = np.argsort(variance_xy)
    sorted_x, sorted_y, sorted_variance_xy = x[idx], y[idx], variance_xy[idx]
    sorted_x_orig, sorted_y_orig = orig_x[idx], orig_y[idx]
    sorted_cmap = cmap[idx]

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(f"Latent space - color by combined variance")
    a = axs[0].scatter(sorted_x_orig, sorted_y_orig, c=sorted_variance_xy, cmap="jet", alpha=0.5)
    axs[0].set_title("Original")
    a = axs[1].scatter(sorted_x, sorted_y, c=sorted_variance_xy, cmap="jet", alpha=0.5)
    axs[1].set_title("Reconstructed")
    plt.colorbar(a)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(f"Latent space - color by original position")
    a = axs[0].scatter(sorted_x_orig, sorted_y_orig, c=sorted_cmap, cmap="jet", alpha=0.5)
    axs[0].set_title("Original")
    a = axs[1].scatter(sorted_x, sorted_y, c=sorted_cmap, cmap="jet", alpha=0.5)
    axs[1].set_title("Reconstructed")
    plt.colorbar(a)

    idx = np.argsort(variance_x)
    sorted_x, sorted_y, sorted_variance_x = x[idx], y[idx], variance_x[idx]

    fig, axs = plt.subplots(1, 1)
    fig.suptitle(f"Reconstructed - Latent space - color by x variance")
    a = axs.scatter(sorted_x, sorted_y, c=sorted_variance_x, cmap="jet", alpha=0.5)
    axs.set_title("Mean")
    plt.colorbar(a)

    idx = np.argsort(variance_y)
    sorted_x, sorted_y, sorted_variance_y = x[idx], y[idx], variance_y[idx]

    fig, axs = plt.subplots(1, 1)
    fig.suptitle(f"Reconstructed - Latent space - color by y variance")
    a = axs.scatter(sorted_x, sorted_y, c=sorted_variance_y, cmap="jet", alpha=0.5)
    axs.set_title("Mean")
    plt.colorbar(a)

    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle(f"Latent space - {len(x)=}")
    # axs[0].scatter(x, y, c=nu, cmap="jet")
    # axs[0].set_title("Mean")
    # a = axs[1].scatter(xs, ys, c=nu, cmap="jet")
    # axs[1].set_title("log - Variance")
    # plt.colorbar(a)

    # fig, axs = plt.subplots(1, 2)
    # fig.suptitle("Reconstructed latent space - shared color")
    # axs[0].scatter(x, y, c=nu, cmap="jet")
    # axs[0].set_title("Mean")
    # a = axs[1].scatter(np.exp(xs), np.exp(ys), c=nu, cmap="jet")
    # axs[1].set_title("Variance")
    # plt.colorbar(a)
    

def plot_all(model, valid_pic):
    with torch.no_grad():
        recon_mu, recon_s = [a.detach().cpu().numpy() for a in model.encoder.encode(model.img2block(valid_pic)[:, None])]

    x  = recon_mu[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon_mu[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon_mu[:, 2*n_kernels:3*n_kernels].flatten()

    xs = recon_s[:, 0*n_kernels:1*n_kernels].flatten()
    ys = recon_s[:, 1*n_kernels:2*n_kernels].flatten()
    nus = recon_s[:, 2*n_kernels:3*n_kernels].flatten()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")
    a = axs[1].scatter(xs, ys, c=nu, cmap="jet")
    axs[1].set_title("log - Variance")
    plt.colorbar(a)

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")
    a = axs[1].scatter(np.exp(xs), np.exp(ys), c=nu, cmap="jet")
    axs[1].set_title("Variance")
    plt.colorbar(a)

    # plt.scatter(x, y, c=np.exp(xs)+np.exp(ys), cmap="Reds")
    # plt.colorbar()
    plt.figure()
    plt.title("Reconstructed latent space")
    plt.hexbin(x, y, np.exp(xs)+np.exp(ys), marginals=True)

    with torch.no_grad():
        recon = model.encoder(model.img2block(valid_pic)[:, None])[0].detach().cpu()

    x  = recon[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon[:, 2*n_kernels:3*n_kernels].flatten()

    plt.figure()
    plt.title("Reconstructed latent space - after repameterization")
    plt.scatter(x, y, c=nu, cmap="jet")
    plt.colorbar()
    plt.show()

    plt.figure()
    # Plot seaborn histogram overlaid with KDE
    ax = sns.histplot(data=x, bins=100, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
    ax.get_lines()[0].set_color('black') # edit line color due to bug in sns v 0.11.0
    # Edit legemd and add title
    ax.legend(frameon=False)
    ax.set_title('Histogramm reconstructed X', fontsize=14, pad=15)
    plt.show()

    plt.figure()
    # Plot seaborn histogram overlaid with KDE
    ax = sns.histplot(data=y, bins=100, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
    ax.get_lines()[0].set_color('black') # edit line color due to bug in sns v 0.11.0
    # Edit legemd and add title
    ax.legend(frameon=False)
    ax.set_title('Histogramm reconstructed Y', fontsize=14, pad=15)
    plt.show()

    plt.figure()
    # Plot seaborn histogram overlaid with KDE
    ax = sns.histplot(data=nu, bins=100, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
    ax.get_lines()[0].set_color('black') # edit line color due to bug in sns v 0.11.0
    # Edit legemd and add title
    ax.legend(frameon=False)
    ax.set_title('Histogramm reconstructed nu', fontsize=14, pad=15)
    plt.show()

    plt.figure()
    plt.title("Reconstructed latent space - after repameterization")
    plt.hexbin(x, y, nu, marginals=True)
    plt.show()

n_kernels, block_size, img_size = 4, 16, 512
train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)

device = "cuda" if torch.cuda.is_available() else "cpu" 

hidden_dims = [2, 4, 8, 16, 32, 64, 128]

# nr_epochs = 1500
nr_epochs = 1000
nr_batches = 15

batch_size = 2_500
load_final = True
input_is_img = False
batch_norm = False
negative_experts = False
kernels_outside = False
downsample = False
bias = True
residual = False
order = "lbadr"
dropout = 0.1
activation = "swish"
kld_params_list = [
    {"kld_loss": 3e-5, "ignore_steering_space": False},
    {"kld_loss": 3e-5, "ignore_steering_space": True},
]

train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)

for model, model_name, lr in [
    (CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, kernels_outside=kernels_outside, negative_experts=negative_experts, batch_norm=batch_norm, device=device,
            bias=bias, downsample=downsample, residual=residual, dropout=dropout, order=order, activation=activation), "cae_kernels_inside", 1e-3),
    (VAE_KernelsInside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, batch_norm=True, device=device, conv_args = {"bias": False}), "vae_kernels_inside", 1e-3),
]:
    if load_final:
        model.load_state_dict(torch.load(f"models/facu/checkpoints/{model_name}_random_blocks/final.pth"))
    
    nr_model_params = sum(p.numel() for p in model.parameters())
    print(f"Number of params for model '{model_name}': {nr_model_params}")
    valid_pic = train_loader.get_valid_pic()
    valid_pic = valid_pic.to(device)
    plot_latent_space_conv(model)
    plt.show()

# %%
