import os
import time
from typing import Dict, List
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import *
from models.elvira import Vanilla
from models.facu.variational_abstract import VAE_Abstract
import wandb

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

def train_models(n_kernels: int,
                 block_size: int,
                 img_size: int,
                 hidden_dims: List[int],
                 batch_norm: bool,
                 device: torch.device,
                 nr_epochs: int,
                 nr_batches: int,
                 batch_size: int,
                 load_final: bool,
                 input_is_img: bool,
                 mode: str = "disabled"):
    for model, model_name, lr in [
            [CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, kernels_outside=False, negative_experts=False, downsample=False, batch_norm=batch_norm, device=device), "cae_kernels_inside_jahsgdgjhafjhgkdsf", 1e-4],
            [CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, kernels_outside=True, negative_experts=False, downsample=False, batch_norm=batch_norm, device=device), "cae_kernels_outside", 1e-4],
            [CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, kernels_outside=False, negative_experts=True, downsample=False, batch_norm=batch_norm, device=device), "cae_negative_experts", 1e-4],
            [CAE_Codec(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, kernels_outside=True, negative_experts=True, downsample=False, batch_norm=batch_norm, device=device), "cae_kernels_outside_negative_experts", 1e-4],
        ]:
        model: CAE_Codec
        if load_final:
            model.load_state_dict(torch.load(f"models/facu/checkpoints/{model_name}_random_blocks/final.pth"))
        disable_tqdm = False
        nr_model_params = sum(p.numel() for p in model.parameters())
        run = wandb.init(mode=mode,
                        name=f"{model_name}", group="vae", project=f"{hidden_dims}",
                        config={
                            "n_kernels": n_kernels,
                            "block_size": block_size,
                            "img_size": img_size,
                            "hidden_dims": hidden_dims,
                            "batch_norm": batch_norm,
                            "device": device,
                            "epochs": nr_epochs,
                            "initial_lr": lr,
                        }
                    )
        wandb.watch(model, log="gradients", log_freq=nr_epochs//20)
        print(f"Number of params for model '{model_name}': {nr_model_params}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, cooldown=20, verbose=False)
        early_stopping = EarlyStopping(patience=100, verbose=True, delta=1e-4, memory_size=20, trace_func=print)
        os.makedirs(f"models/facu/checkpoints/{model_name}_random_blocks/", exist_ok=True)
        os.makedirs(f"models/facu/images2/{model_name}_random_blocks/", exist_ok=True)
        valid_pic = train_loader.get_valid_pic()
        wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)], "nr_params": nr_model_params}, step=0, commit=True)
        valid_pic = valid_pic.to(device)
        try:
            outter_pbar = tqdm.tqdm(range(nr_epochs), desc=f"Epoch Loss: 0.0 - LR: {optimizer.param_groups[0]['lr']:.2e}", disable=disable_tqdm)
            x = None
            for epoch in outter_pbar:
                mean_epoch_loss = 0
                with tqdm.tqdm(total=nr_batches, desc=f"Batch 0 - Loss: 0.0", disable=disable_tqdm) as pbar:
                    batch = 0
                    while batch < nr_batches:
                        x_batch = train_loader.get_m_blocks_with_n_kernels(batch_size, n_kernels, kernels_outside="kernelsoutside" in str(model.__class__).lower(), 
                                            negative_experts="negativeexperts" in str(model.__class__).lower(), device=device)
                        if x is not None:
                            _prev = out[2][:, :2*n_kernels]
                            tmp = torch.cdist(x_batch[:, :2*n_kernels], _prev)
                            tmp = tmp.mean(dim=-1).sort().indices
                            sample_inds = torch.multinomial(tmp[:len(tmp)//2].float(), x_batch.size(0), replacement=True)
                            _x_batch = x_batch[sample_inds, :]
                            if False:
                                import matplotlib.pyplot as plt
                                plt.scatter(*out[2][:, :2].T.detach().cpu())
                                plt.scatter(*x_batch[:, :2].T.cpu())
                                plt.scatter(*_x_batch[:, :2].T.cpu())
                                plt.show()
                            x_batch = _x_batch
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0, device=device)
                        loss_present = False
                        x = model.decoder(x_batch.to(device)).float()
                        # x_hat, x, mu, log_var, z = model(x, return_all=True, input_is_img=input_is_img)
                        # losses = model.loss_function(x_hat.squeeze(), x.squeeze(), mu, log_var, kld_weight=kld_loss, ignore_steering_space=kld_ignore_steering_space)
                        out = model(x, return_all=True, input_is_img=input_is_img)
                        losses = model.loss_function(*out, x_batch, n_kernels=n_kernels)
                        loss = losses["loss"]
                        total_loss += loss
                        loss_present = True
                        if torch.cuda.memory_allocated() > torch.cuda.mem_get_info()[1]*0.85:
                            total_loss.backward()  # Has to be computed on every few batches to prevent memory leaks
                            mean_epoch_loss += total_loss.item()
                            total_loss = torch.tensor(0.0, device=device)
                            loss_present = False
                        if loss_present:
                            total_loss.backward()
                            mean_epoch_loss += total_loss.item()
                        # plot_grad_flow(model.named_parameters())
                        optimizer.step()
                        # Update inner progress bar
                        pbar.set_description(f"Completed Batches {batch+1} - Total Loss: {mean_epoch_loss:.1e} - Loss Per Batch: {mean_epoch_loss/(batch+1):.2f}")
                        pbar.update(1)
                        batch += 1
                # Update learning rate
                scheduler.step(mean_epoch_loss)

                # Update outter progress bar
                outter_pbar.set_description(f"Epoch Loss: {mean_epoch_loss:.2e} - LR: {optimizer.param_groups[0]['lr']:.2e}")
                # Save image
                with torch.no_grad():
                    orig_blocks = model.img2block(valid_pic)[:, None].to(device)
                    encoded = model.encoder.encode(orig_blocks)
                    reconstructed_blocks = model.decoder(encoded)
                    recon = model.block2img(reconstructed_blocks).detach().cpu().numpy().squeeze()
                    inds = ((reconstructed_blocks[:, None] - orig_blocks)**2).flatten(start_dim=1).mean(dim=1).sort().indices
                    best_recon = reconstructed_blocks[inds[-1], :].squeeze().detach().cpu().numpy()
                    worst_recon = reconstructed_blocks[inds[0], :].squeeze().detach().cpu().numpy()
                    # recon = model(valid_pic)[0].detach().cpu().numpy().squeeze()
                    if False:
                        plot_all(model, valid_pic)
                
                # save image in wandb
                wandb.log({"reconstructed": [wandb.Image(recon*255)], "best_recon": [wandb.Image(best_recon*255)], "worst_recon": [wandb.Image(worst_recon*255)]}, commit=False)
                # save learning rate
                wandb.log({"learning_rate": optimizer.param_groups[0]['lr']}, commit=False)
                # save mean epoch loss
                wandb.log(losses, commit=True)

                # Early stopping
                early_stopping(mean_epoch_loss, model, f"models/facu/checkpoints/{model_name}_random_blocks/{epoch}.pth")
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            
            wandb.save(f"models/facu/checkpoints/{model_name}_random_blocks/final.pth")
            wandb.finish(0)
        except KeyboardInterrupt:
            print("Training interrupted")
            wandb.finish(1)

def plot_all(model, valid_pic):
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

if __name__ == "__main__":
    n_kernels, block_size, img_size = 4, 16, 512
    train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    hidden_dims = [2, 4, 8, 16, 32, 64, 128]

    # nr_epochs = 1500
    nr_epochs = 1000
    nr_batches = 15
    
    batch_size = 2_500
    load_final = False
    input_is_img = False
    batch_norm = False

    train_models(n_kernels=n_kernels,
                 block_size=block_size,
                 img_size=img_size,
                 hidden_dims=hidden_dims,
                 batch_norm=batch_norm,
                 device=device,
                 nr_epochs=nr_epochs,
                 nr_batches=nr_batches,
                 batch_size=batch_size,
                 load_final=load_final,
                 input_is_img=input_is_img,
                 mode="online")
            