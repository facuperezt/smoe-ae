import os
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import VAE, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP
from models.elvira import Vanilla
import wandb

n_kernels, block_size, img_size = 3, 8, 256
train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)
train_loader.initialize(n_repeats=5, force_reinitialize=False)

device = "cuda" if torch.cuda.is_available() else "cpu" 

hidden_dims = [32, 32, 64, 64, 128, 256]

nr_epochs = 250


for model, model_name, lr in zip(
        [
            # Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
            # Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device, force_conv_layers=[16, 32, 64], force_dense_layers=[64]),
            VAE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device),
            VAE_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device),
            VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device),
            VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device),
            # SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device),
            # SimpleMLP(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device, force_hidden_sizes=[4*block_size**2, 8*block_size**2, 4*block_size**2, block_size**2]),
        ],
        [
            # "elvira",
            # "elvira_small",
            "vae_kernels_inside",
            "vae_kernels_outside",
            "vae_negative_experts",
            "vae_kernels_outside_negative_experts",
            # "mlp",
            # "mlp_big",
        ],
        [
            # 1e-3,
            # 1e-4,
            5e-4,
            5e-4,
            5e-4,
            5e-4,
            # 1e-4,
            # 1e-4,
        ]
    ):
    # start disabled run
    run = wandb.init(mode="online",
                     name=model_name, group="vae",
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=20, verbose=False)

    os.makedirs(f"models/facu/checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/facu/images2/{model_name}", exist_ok=True)
    valid_pic = train_loader.get_valid_pic()
    # save original image
    wandb.log({"original": [wandb.Image(valid_pic.numpy()*255)]}, step=0, commit=True)
    # Image.fromarray(valid_pic.numpy()*255).convert("L").save(f"models/facu/images2/{model_name}/original.jpeg")
    valid_pic = valid_pic.to(device)
    try:
        for epoch in tqdm.tqdm(range(nr_epochs), "Epoch: "):
            mean_epoch_loss = 0
            for batch, (x_batch, _) in enumerate(train_loader.get("train", None, -5)):
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
                print(f"Batch {batch}, Loss {loss.item()}")
                optimizer.step()

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

            # Update learning rate
            scheduler.step(loss)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Save final model
        torch.save(model.state_dict(), f"models/facu/checkpoints/{model_name}/final.pth")
        wandb.finish(0)