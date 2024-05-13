import os
import time
from typing import Dict
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import VAE_KernelsInside, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP,\
        VAE_Residual, VAE_Residual_NegativeExperts, VAE_Residual_KernelsOutside, VAE_Residual_KernelsOutsideNegativeExperts,\
        VAE_Residual_Downsampling, VAE_Residual_Downsampling_NegativeExperts, VAE_Residual_Downsampling_KernelsOutside, VAE_Residual_Downsampling_KernelsOutsideNegativeExperts, \
        VAE_Residual_DeepConv, VAE_Residual_DeepConv_NegativeExperts, VAE_Residual_DeepConv_KernelsOutside, VAE_Residual_DeepConv_KernelsOutsideNegativeExperts, \
        VAE_Residual_DeepConv_Downsampling, VAE_Residual_DeepConv_Downsampling_NegativeExperts, VAE_Residual_DeepConv_Downsampling_KernelsOutside, VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts
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
        val = self.start_weight + (self.end_weight - self.start_weight) * (1 + torch.cos(torch.tensor(self.current_epoch/self.nr_epochs) * 3.141592653589793))/2
        if self.current_epoch < self.wait:
            return 0.0
        elif self.current_epoch < self.warmup + self.wait:
            return ((self.current_epoch - self.wait)/self.warmup) * self.start_weight
        return val

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

def train_models(n_kernels, block_size, img_size, hidden_dims, batch_norm, device, nr_epochs, nr_batches, batch_size, load_final, input_is_img, kld_loss_params: Dict = None, mode="disabled"):
    for model, model_name, lr in [
        # [Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device), "elvira", 1e-3],
        # [Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device, force_conv_layers=[16, 32, 64], force_dense_layers=[64]), "elvira_small", 1e-4],

        [VAE_KernelsInside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, batch_norm=batch_norm, device=device), "vae_kernels_inside", 4e-2],
        [VAE_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, batch_norm=batch_norm, device=device), "vae_kernels_outside", 4e-2],
        # [VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, batch_norm=batch_norm, device=device), "vae_negative_experts", 4e-2],
        # [VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, batch_norm=batch_norm, device=device), "vae_kernels_outside_negative_experts", 4e-2],
        
        # [VAE_Residual(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_inside", 4e-4],
        # [VAE_Residual_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_negative_experts", 4e-4],
        # [VAE_Residual_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_outside", 4e-4],
        # [VAE_Residual_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_outside_negative_experts", 4e-4],

        # [VAE_Residual_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_inside_downsampling", 4e-4],
        # [VAE_Residual_Downsampling_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_negative_experts_downsampling", 4e-4],
        # [VAE_Residual_Downsampling_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_outside_downsampling", 4e-4],
        # [VAE_Residual_Downsampling_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_vae_kernels_outside_negative_experts_downsampling", 4e-4],

        # [VAE_Residual_DeepConv(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_inside", 4e-4],
        # [VAE_Residual_DeepConv_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_negative_experts", 4e-4],
        # [VAE_Residual_DeepConv_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_outside", 4e-4],
        # [VAE_Residual_DeepConv_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_outside_negative_experts", 4e-4],

        # [VAE_Residual_DeepConv_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "NO_LOG_VAR_residual_deep_conv_vae_kernels_inside_downsampling", 1e-4],
        # [VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "NO_LOG_VAR_residual_deep_conv_vae_kernels_outside_negative_experts_downsampling", 1e-4],
        # [VAE_Residual_DeepConv_Downsampling_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "NO_LOG_VAR_residual_deep_conv_vae_negative_experts_downsampling", 1e-4],
        # [VAE_Residual_DeepConv_Downsampling_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "NO_LOG_VAR_residual_deep_conv_vae_kernels_outside_downsampling", 1e-4],

        # [VAE_Residual_DeepConv_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_inside_downsampling", 1e-3],
        # [VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_outside_negative_experts_downsampling", 1e-3],
        # [VAE_Residual_DeepConv_Downsampling_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_negative_experts_downsampling", 1e-3],
        # [VAE_Residual_DeepConv_Downsampling_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device, batch_norm=batch_norm), "residual_deep_conv_vae_kernels_outside_downsampling", 1e-3],
        
        # [SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device), "mlp", 1e-4],
        # [SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device, force_hidden_sizes=[4*block_size**2, 8*block_size**2, 4*block_size**2, block_size**2]), "mlp_big", 1e-4],
    ]:
        if kld_loss_params is None:
            kld_loss = 1e-4
            kld_ignore_steering_space = False
        else:
            kld_loss = kld_loss_params.get("kld_loss", 1e-4)
            kld_ignore_steering_space = kld_loss_params.get("ignore_steering_space", False)
        model: VAE_Abstract
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
                            "ignore_steering_space": kld_ignore_steering_space,
                        }
                    )
        wandb.watch(model, log="gradients", log_freq=nr_epochs//20)
        print(f"Number of params for model '{model_name}': {nr_model_params}")
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=nr_epochs//20, cooldown=nr_epochs//40, verbose=False)
        early_stopping = EarlyStopping(patience=nr_epochs//10, verbose=True, delta=1e-4, memory_size=20, trace_func=print)
        kld_annealing = KLD_Weight_CosineAnnealing(0, kld_loss, 100, nr_batches//5, 200)

        os.makedirs(f"models/facu/checkpoints/{model_name}_random_blocks/", exist_ok=True)
        os.makedirs(f"models/facu/images2/{model_name}_random_blocks/", exist_ok=True)
        valid_pic = train_loader.get_valid_pic()
        # save original image
        wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)], "nr_params": nr_model_params}, step=0, commit=True)
        # Image.fromarray(valid_pic.numpy()*255).convert("L").save(f"models/facu/images2/{model_name}/original.jpeg")
        valid_pic = valid_pic.to(device)
        try:
            outter_pbar = tqdm.tqdm(range(nr_epochs), desc=f"Epoch Loss: 0.0 - LR: {optimizer.param_groups[0]['lr']:.2e}", disable=disable_tqdm)
            for epoch in outter_pbar:
                mean_epoch_loss = 0
                mean_epoch_reconstr_loss = 0
                mean_epoch_kl_loss = 0
                # data_queue = train_loader.generate_random_blocks_queue(nr_batches, batch_size, n_kernels, block_size, include_zero=True, 
                #                                         negative_experts="negativeexperts" in str(model.__class__).lower())
                with tqdm.tqdm(total=nr_batches, desc=f"Batch 0 - Loss: 0.0", disable=disable_tqdm) as pbar:
                    batch = 0
                    kld_loss = kld_annealing()
                    while batch < nr_batches:
                        x_batch = train_loader.get_m_blocks_with_n_kernels(batch_size, n_kernels, kernels_outside="kernelsoutside" in str(model.__class__).lower(), 
                                            negative_experts="negativeexperts" in str(model.__class__).lower(), device=device)
                        optimizer.zero_grad()
                        total_loss = torch.tensor(0.0, device=device)
                        loss_present = False
                        x = model.decoder(x_batch.to(device)).float()
                        x_hat, x, mu, log_var, z = model(x, return_all=True, input_is_img=input_is_img)
                        losses = model.loss_function(x_hat.squeeze(), x.squeeze(), mu, log_var, kld_weight=kld_loss, ignore_steering_space=kld_ignore_steering_space)
                        loss = losses["loss"]
                        losses["kld_weight"] = kld_loss
                        mean_epoch_reconstr_loss += losses["Reconstruction_Loss"].item()
                        mean_epoch_kl_loss += losses["KLD"].item()
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
                    recon = model(valid_pic)[0].detach().cpu().numpy().squeeze()

                # save image in wandb
                wandb.log({"reconstructed": [wandb.Image(recon*255)]}, commit=False)
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

if __name__ == "__main__":
    n_kernels, block_size, img_size = 4, 16, 512
    train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)

    device = "cuda" if torch.cuda.is_available() else "cpu" 

    hidden_dims = [2, 4, 8, 16, 32, 64, 128, 256]

    nr_epochs = 2000
    nr_batches = 50
    batch_size = 1_000_000
    load_final = False
    input_is_img = False
    batch_norm = [True, False]
    kld_params_list = [{"kld_loss": 1e-4, "ignore_steering_space": False},
                       {"kld_loss": 1e-4, "ignore_steering_space": True},
                       {"kld_loss": 1e-3, "ignore_steering_space": False},
                       {"kld_loss": 1e-3, "ignore_steering_space": True},]

    for bn in batch_norm:
        for kld in kld_params_list:
            train_models(n_kernels, block_size, img_size, hidden_dims, bn, device, nr_epochs, nr_batches, batch_size, load_final, input_is_img, kld,
                         mode="disabled")