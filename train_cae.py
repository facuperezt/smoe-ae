#%%
import json
import os
import time
from typing import Dict, List, Literal, Union
from matplotlib import pyplot as plt
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import *
from models.elvira import Vanilla
from models.facu.variational_abstract import VAE_Abstract
import wandb

from utils import EarlyStopping, plot_reconstructions, plot_grad_flow

from datetime import datetime

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
nvmlInit()
_gpu_handle = nvmlDeviceGetHandleByIndex(0)

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
                data_mode: Literal["dataset", "synthetic"] = "synthetic",
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

    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters()},
        ], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=20, verbose=False)
    early_stopping = EarlyStopping(patience=250, verbose=True, delta=1e-4, memory_size=20, trace_func=print)

    os.makedirs(f"{model_path}", exist_ok=True)

    train_loader = DataLoader(mode=data_mode, data_path="professional_photos", img_size=img_size, block_size=block_size,
                              batch_size=batch_size, n_kernels=n_kernels, kernels_outside=kernels_outside, negative_experts=negative_experts,
                              device=device)
    
    valid_pic = train_loader.get_valid_pic()
    wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)], "nr_params": nr_model_params}, step=0, commit=True)
    valid_pic = valid_pic.to(device)

    _debug_var = False
    try:
        outter_pbar = tqdm.tqdm(range(nr_epochs), desc=f"Epoch Loss: 0.0 - LR: {optimizer.param_groups[0]['lr']:.2e}", disable=disable_tqdm)
        x = None
        for epoch in outter_pbar:
            epoch_loss = 0
            current_loss = torch.tensor(0.0, device=device)
            loss_present = False
            with tqdm.tqdm(total=nr_batches, desc=f"Batch 0 - Loss: 0.0", disable=disable_tqdm) as pbar:
                batch = 0
                while batch < nr_batches:
                    x_batch = train_loader.get(include_zero=True)
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
                    if data_mode == "synthetic":
                        x = model.decoder(x_batch.to(device)).float()
                    elif data_mode == "dataset":
                        x = x_batch.to(device)
                    out = model(x, return_all=True, input_is_img=input_is_img)
                    losses = model.loss_function(*out, x_batch, n_kernels=n_kernels if not input_is_img else False)
                    loss = losses["loss"]
                    current_loss += loss
                    epoch_loss += loss.item()
                    loss_present = True
                    _gpu_info = nvmlDeviceGetMemoryInfo(_gpu_handle)
                    if torch.cuda.memory_reserved()/_gpu_info.total > 0.75 and torch.cuda.memory_allocated()/_gpu_info.total > 0.5:
                        current_loss.backward()  # Has to be computed on every few batches to prevent memory overruns
                        current_loss = torch.tensor(0.0, device=device)
                        loss_present = False
                    # Update inner progress bar
                    pbar.set_description(f"Completed Batches {batch+1} - Total Loss: {epoch_loss:.1e} - Loss Per Batch: {epoch_loss/(batch+1):.2f}")
                    pbar.update(1)
                    batch += 1
            if loss_present:
                current_loss.backward()
            optimizer.step()
            # Update learning rate
            scheduler.step(epoch_loss)

            # Update outter progress bar
            outter_pbar.set_description(f"Epoch Loss: {epoch_loss:.2e} - LR: {optimizer.param_groups[0]['lr']:.2e}")
            if epoch % 10 == 0:
                # Save image
                with torch.no_grad():
                    recon, worst_fig, median_fig, best_fig = plot_reconstructions(model, valid_pic, model.device)
                    gf = plot_grad_flow(model.named_parameters())
            
                # save grad flow plot in wandb
                wandb.log({"gradient_flow": wandb.Image(gf)}, commit=False)
                # save image in wandb
                wandb.log({"reconstructed": [wandb.Image(recon.numpy()*255)], "blockwise_recon": {"best":[wandb.Image(best_fig)], "median": [wandb.Image(median_fig)], "worst": [wandb.Image(worst_fig)]}}, commit=False)
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
            early_stopping(epoch_loss, model, f"{model_path}/{epoch}.pth")
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
            