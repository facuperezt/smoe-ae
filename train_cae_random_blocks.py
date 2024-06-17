#%%
import json
import os
import time
from typing import Dict, List, Union
from matplotlib import pyplot as plt
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import *
from models.elvira import Vanilla
from models.facu.variational_abstract import VAE_Abstract
import wandb

from utils import plot_block_with_kernels

from datetime import datetime

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

def plot_all(model, valid_pic, n_kernels):
    import torch
    import seaborn as sns
    import numpy as np
    import matplotlib.pyplot as plt

    with torch.no_grad():
        recon_mu = model.encoder.encode(model.img2block(valid_pic)[:, None]).detach().cpu()

    x  = recon_mu[:, 0*n_kernels:1*n_kernels].flatten()
    y  = recon_mu[:, 1*n_kernels:2*n_kernels].flatten()
    nu = recon_mu[:, 2*n_kernels:3*n_kernels].flatten()

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")

    fig, axs = plt.subplots(1, 2)
    fig.suptitle("Reconstructed latent space - shared color")
    axs[0].scatter(x, y, c=nu, cmap="jet")
    axs[0].set_title("Mean")

    # plt.scatter(x, y, c=np.exp(xs)+np.exp(ys), cmap="Reds")
    # plt.colorbar()
    plt.figure()
    plt.title("Reconstructed latent space")
    plt.hexbin(x, y, marginals=True)

    with torch.no_grad():
        recon = model.encoder(model.img2block(valid_pic)[:, None]).detach().cpu()

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
    ax = sns.histplot(data=x, bins=20, stat='density', alpha= 1, kde=True,
                    edgecolor='white', linewidth=0.5,
                    line_kws=dict(color='black', alpha=0.5, linewidth=1.5, label='KDE'))
    ax.get_lines()[0].set_color('black') # edit line color due to bug in sns v 0.11.0
    # Edit legemd and add title
    ax.legend(frameon=False)
    ax.set_title('Seaborn histogram overlaid with KDE', fontsize=14, pad=15)
    plt.show()

    plt.figure()
    plt.title("Reconstructed latent space - after repameterization")
    plt.hexbin(x, y, nu, marginals=True)
    plt.show()

def plot_grad_flow(named_parameters):
    import matplotlib.pyplot as plt
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            if p.grad is None:
                print(f"None gradient for {n}")
                continue
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
    fig = plt.figure()
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.tight_layout()
    return fig

def train_model(n_kernels: int,
                block_size: int,
                img_size: int,
                hidden_dims: List[int],
                kernels_outside: bool,
                negative_experts: bool,
                downsample: bool,
                batch_norm: bool,
                bias: bool,
                residual: bool,
                dropout: float,
                order: str,
                activation: Union[str, torch.nn.Module], 
                device: torch.device = "cuda",
                model_name: str = "temp",
                lr: float = 1e-4,
                nr_epochs: int = 100,
                nr_batches: int = 20,
                batch_size: int = 1000,
                load_model: bool = "",
                input_is_img: bool = False,
                wandb_mode: str = "disabled"):
    model: CAE_Codec = CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims,
                                 kernels_outside=kernels_outside, negative_experts=negative_experts, downsample=downsample,
                                 batch_norm=batch_norm, bias=bias, residual=residual, dropout=dropout, order=order,
                                 activation=activation, device=device)
    disable_tqdm = False
    nr_model_params = sum(p.numel() for p in model.parameters())
    start_time = str(datetime.now()).split(".")[0].replace(":", ";")
    model_name = "_".join([model_name, start_time])
    model_path = f"models/facu/checkpoints/random_blocks/{model_name}"
    if load_model:
        model.load_state_dict(torch.load(load_model))
    run = wandb.init(mode=wandb_mode,
                    name=f"{model_name}", group="vae", project=f"{hidden_dims}",
                    config={
                        "n_kernels": n_kernels,
                        "block_size": block_size,
                        "img_size": img_size,
                        "hidden_dims": hidden_dims,
                        "kernels_outside": kernels_outside,
                        "negative_experts": negative_experts,
                        "downsample": downsample,
                        "batch_norm": batch_norm,
                        "bias": bias,
                        "residual": residual,
                        "dropout": dropout,
                        "order": order,
                        "activation": activation,
                        "device": device,
                        "batch_size": batch_size,
                        "initial_lr": lr,
                        "start_time": start_time,
                    }
                )
    wandb.watch(model, log="gradients", log_freq=nr_epochs//20)
    print(f"Number of params for model '{model_name}': {nr_model_params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=20, verbose=False)
    early_stopping = EarlyStopping(patience=250, verbose=True, delta=1e-4, memory_size=20, trace_func=print)

    os.makedirs(f"{model_path}", exist_ok=True)

    train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)
    valid_pic = train_loader.get_valid_pic()
    wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)], "nr_params": nr_model_params}, step=0, commit=True)
    valid_pic = valid_pic.to(device)

    _debug_var = False
    try:
        outter_pbar = tqdm.tqdm(range(nr_epochs), desc=f"Epoch Loss: 0.0 - LR: {optimizer.param_groups[0]['lr']:.2e}", disable=disable_tqdm)
        x = None
        for epoch in outter_pbar:
            mean_epoch_loss = 0
            with tqdm.tqdm(total=nr_batches, desc=f"Batch 0 - Loss: 0.0", disable=disable_tqdm) as pbar:
                batch = 0
                while batch < nr_batches:
                    x_batch = train_loader.get_m_blocks_with_n_kernels(batch_size, n_kernels, kernels_outside=kernels_outside, 
                                                                       negative_experts=negative_experts, device=device)
                    # if x is not None:
                    #     _prev = out[2][:, :2*n_kernels]
                    #     tmp = torch.cdist(x_batch[:, :2*n_kernels], _prev)
                    #     tmp = tmp.mean(dim=-1).sort().indices
                    #     sample_inds = torch.multinomial(tmp[:len(tmp)//2].float(), x_batch.size(0), replacement=True)
                    #     _x_batch = x_batch[sample_inds, :]
                    #     if _debug_var:
                    #         plt.scatter(*out[2][:, :2].T.detach().cpu())
                    #         plt.scatter(*x_batch[:, :2].T.cpu())
                    #         plt.scatter(*_x_batch[:, :2].T.cpu())
                    #         plt.show()
                    #     x_batch = _x_batch
                    optimizer.zero_grad()
                    total_loss = torch.tensor(0.0, device=device)
                    loss_present = False
                    x = model.decoder(x_batch.to(device)).float()
                    out = model(x, return_all=True, input_is_img=input_is_img)
                    losses = model.loss_function(*out, x_batch, n_kernels=n_kernels)
                    loss = losses["loss"]
                    total_loss += loss
                    loss_present = True
                    if torch.cuda.memory_allocated() > torch.cuda.mem_get_info()[1]*0.85:
                        total_loss.backward()  # Has to be computed on every few batches to prevent memory overruns
                        mean_epoch_loss += total_loss.item()
                        total_loss = torch.tensor(0.0, device=device)
                        loss_present = False
                    if loss_present:
                        total_loss.backward()
                        mean_epoch_loss += total_loss.item()
                    # Update inner progress bar
                    pbar.set_description(f"Completed Batches {batch+1} - Total Loss: {mean_epoch_loss:.1e} - Loss Per Batch: {mean_epoch_loss/(batch+1):.2f}")
                    pbar.update(1)
                    batch += 1
            optimizer.step()
            # Update learning rate
            scheduler.step(mean_epoch_loss)

            # Update outter progress bar
            outter_pbar.set_description(f"Epoch Loss: {mean_epoch_loss:.2e} - LR: {optimizer.param_groups[0]['lr']:.2e}")
            if epoch % 10 == 0:
                # Save image
                with torch.no_grad():
                    orig_blocks = model.img2block(valid_pic)[:, None].to(device)
                    encoded = model.encoder(orig_blocks)
                    reconstructed_blocks = model.decoder(encoded)
                    _recon = model.block2img(reconstructed_blocks)
                    recon = _recon.detach().cpu().squeeze()
                    inds = ((reconstructed_blocks[:, None] - orig_blocks)**2).flatten(start_dim=1).mean(dim=1).sort().indices
                    best_recon = reconstructed_blocks[inds[0], :].squeeze().detach().cpu()
                    worst_recon = reconstructed_blocks[inds[-1], :].squeeze().detach().cpu()
                    orig_best_recon = orig_blocks[inds[0], :].squeeze().detach().cpu()
                    orig_worst_recon = orig_blocks[inds[-1], :].squeeze().detach().cpu()

                    best_fig = plt.figure()
                    plot_block_with_kernels(encoded[inds[-1]], best_recon, block_size=block_size)
                    plt.axis('off')
                    worst_fig = plt.figure()
                    plot_block_with_kernels(encoded[inds[0]], worst_recon, block_size=block_size)
                    plt.axis('off')
                    gf = plot_grad_flow(model.named_parameters())
            
                # save grad flow plot in wandb
                wandb.log({"gradient_flow": wandb.Image(gf)}, commit=False)
                # save image in wandb
                wandb.log({"reconstructed": [wandb.Image(recon.numpy()*255)], "best_recon": {"recon":[wandb.Image(best_fig)], "orig": [wandb.Image(orig_best_recon.numpy()*255)]}, "worst_recon": {"recon": [wandb.Image(worst_fig)], "orig": [wandb.Image(orig_worst_recon.numpy()*255)]}}, commit=False)
                if _debug_var:
                    best_fig.canvas.manager.window.wm_geometry("+0+0")
                    worst_fig.canvas.manager.window.wm_geometry("+600+0")
                    gf.canvas.manager.window.wm_geometry("+0+600")
                # close plots
                plt.close("all")
            # save learning rate
            wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, commit=False)
            # save mean epoch loss
            wandb.log(losses, commit=True)

            # Early stopping
            early_stopping(mean_epoch_loss, model, f"{model_path}/{epoch}.pth")
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        torch.save(model.state_dict(), f"{model_path}/final.pth")
        wandb.save(f"{model_path}/final.pth")
        wandb.finish(0)
    except KeyboardInterrupt:
        print("Training interrupted")
        wandb.finish(1)

#%%
if __name__ == "__main__":
    with open("models/facu/model_cfg/great_success.json", "r") as f:
        model_configs = json.load(f)
    train_model(**model_configs)
            