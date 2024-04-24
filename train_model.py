import os
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import VAE, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP,\
        VAE_Residual, VAE_Residual_NegativeExperts, VAE_Residual_KernelsOutside, VAE_Residual_KernelsOutsideNegativeExperts,\
        VAE_Residual_Downsampling, VAE_Residual_Downsampling_NegativeExperts, VAE_Residual_Downsampling_KernelsOutside, VAE_Residual_Downsampling_KernelsOutsideNegativeExperts, \
        VAE_Residual_DeepConv, VAE_Residual_DeepConv_NegativeExperts, VAE_Residual_DeepConv_KernelsOutside, VAE_Residual_DeepConv_KernelsOutsideNegativeExperts, \
        VAE_Residual_DeepConv_Downsampling, VAE_Residual_DeepConv_Downsampling_NegativeExperts, VAE_Residual_DeepConv_Downsampling_KernelsOutside, VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts
from models.elvira import Vanilla
from models.facu.variational_abstract import VAE_Abstract
import wandb

n_kernels, block_size, img_size = 3, 8, 256
train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)
train_loader.initialize(n_repeats=5, force_reinitialize=False)

device = "cuda" if torch.cuda.is_available() else "cpu" 

hidden_dims = [16, 16, 16, 64, 64, 128]

nr_epochs = 50

for model, model_name, lr in [
    # [Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device), "elvira", 1e-3],
    # [Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device, force_conv_layers=[16, 32, 64], force_dense_layers=[64]), "elvira_small", 1e-4],

    # [VAE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "vae_kernels_inside", 4e-4],
    # [VAE_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "vae_kernels_outside", 4e-4],
    # [VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "vae_negative_experts", 4e-4],
    # [VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "vae_kernels_outside_negative_experts", 4e-4],

    # [VAE_Residual(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_inside", 4e-4],
    # [VAE_Residual_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_negative_experts", 4e-4],
    # [VAE_Residual_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_outside", 4e-4],
    # [VAE_Residual_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_outside_negative_experts", 4e-4],

    # [VAE_Residual_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_inside_downsampling", 4e-4],
    # [VAE_Residual_Downsampling_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_negative_experts_downsampling", 4e-4],
    # [VAE_Residual_Downsampling_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_outside_downsampling", 4e-4],
    # [VAE_Residual_Downsampling_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_vae_kernels_outside_negative_experts_downsampling", 4e-4],

    [VAE_Residual_DeepConv(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_inside", 4e-4],
    [VAE_Residual_DeepConv_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_negative_experts", 4e-4],
    [VAE_Residual_DeepConv_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_outside", 4e-4],
    [VAE_Residual_DeepConv_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_outside_negative_experts", 4e-4],

    [VAE_Residual_DeepConv_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_inside_downsampling", 6e-4],
    [VAE_Residual_DeepConv_Downsampling_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_negative_experts_downsampling", 6e-4],
    [VAE_Residual_DeepConv_Downsampling_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_outside_downsampling", 6e-4],
    [VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device), "residual_deep_conv_vae_kernels_outside_negative_experts_downsampling", 6e-4],

    # [SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device), "mlp", 1e-4],
    # [SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device, force_hidden_sizes=[4*block_size**2, 8*block_size**2, 4*block_size**2, block_size**2]), "mlp_big", 1e-4],
]:
    model: VAE_Abstract
    disable_tqdm = False
    # start disabled run
    run = wandb.init(mode="online",
                     name=f"{model_name}", group="vae", project=f"{hidden_dims}",
                     config={
                        "n_kernels": n_kernels,
                        "block_size": block_size,
                        "img_size": img_size,
                        "hidden_dims": hidden_dims,
                        "device": device,
                        "epochs": nr_epochs,
                        "initial_lr": lr,
                    }
                )
    print(f"Number of params for model '{model_name}': {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False)

    os.makedirs(f"models/facu/checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/facu/images2/{model_name}", exist_ok=True)
    valid_pic = train_loader.get_valid_pic()
    # save original image
    wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)]}, step=0, commit=True)
    # Image.fromarray(valid_pic.numpy()*255).convert("L").save(f"models/facu/images2/{model_name}/original.jpeg")
    valid_pic = valid_pic.to(device)
    try:
        outter_pbar = tqdm.tqdm(range(nr_epochs), desc=f"Epoch Loss: 0.0 - LR: {optimizer.param_groups[0]['lr']:.2e}", disable=disable_tqdm)
        for epoch in outter_pbar:
            mean_epoch_loss = 0
            nr_batches = 10
            pbar = tqdm.tqdm(enumerate(train_loader.get("train", None, -nr_batches)), total=nr_batches, desc=f"Batch 0 - Loss: 0.0", disable=disable_tqdm)
            for batch, (x_batch, _) in pbar:
                optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=device)
                loss_present = False
                for i, x in enumerate(x_batch):
                    x = x.to(device)
                    x_hat, x, mu, log_var = model(x)
                    loss = model.loss_function(x_hat.squeeze(), x.squeeze(), mu, log_var)['loss']
                    total_loss += loss
                    loss_present = True
                    if torch.cuda.memory_allocated() > 5e9:
                        total_loss.backward()  # Has to be computed on every few batches to prevent memory leaks
                        mean_epoch_loss += total_loss.item()
                        total_loss = torch.tensor(0.0, device=device)
                        loss_present = False
                if loss_present:
                    total_loss.backward()
                    mean_epoch_loss += total_loss.item()
                optimizer.step()
                # Update inner progress bar
                pbar.set_description(f"Completed Batches {batch+1} - Total Loss: {mean_epoch_loss:.1e} - Loss Per Batch: {mean_epoch_loss/(batch+1):.2f}")
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
            wandb.log({"mean_epoch_loss": mean_epoch_loss}, commit=True)

            # Image.fromarray(recon*255).convert("L").save(f"models/facu/images2/{model_name}/reconstructed_{epoch}.jpeg")

            # Save model
            torch.save(model.state_dict(), f"models/facu/checkpoints/{model_name}/{epoch}.pth")

        wandb.finish(0)
    except KeyboardInterrupt:
        print("Training interrupted")
        wandb.finish(1)
    finally:
        # Save final model
        torch.save(model.state_dict(), f"models/facu/checkpoints/{model_name}/final.pth")
        wandb.save(f"models/facu/checkpoints/{model_name}/final.pth")
