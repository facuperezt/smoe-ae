import os
import torch
from PIL import Image
import tqdm
from data import DataLoader
from models.facu import VAE, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP
from models.elvira import Vanilla

n_kernels, block_size, img_size = 4, 8, 128
train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)
train_loader.initialize(n_repeats=5, force_reinitialize=False)

device = "cuda" if torch.cuda.is_available() else "cpu" 

for model, model_name, lr in zip(
        [
            # Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
            # Vanilla(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device, force_conv_layers=[16, 32, 64], force_dense_layers=[64]),
            VAE(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
            VAE_KernelsOutside(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
            VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
            VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, load_tf_model=False, device=device),
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
            1e-4,
            1e-4,
            1e-4,
            1e-4,
            # 1e-4,
            # 1e-4,
        ]
    ):
    print(f"Number of params for model '{model_name}': {sum(p.numel() for p in model.parameters())}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.33, patience=5, verbose=True)

    os.makedirs(f"models/facu/checkpoints/{model_name}", exist_ok=True)
    os.makedirs(f"models/facu/images2/{model_name}", exist_ok=True)
    valid_pic = next(train_loader.get("valid", 1, 1))[0].squeeze()
    Image.fromarray(valid_pic.numpy()*255).convert("L").save(f"models/facu/images2/{model_name}/original.jpeg")
    valid_pic = valid_pic.to(device)
    try:
        for epoch in tqdm.tqdm(range(30), "Epoch: "):
            for batch, (x_batch, _) in enumerate(train_loader.get("train", None, -5)):
                optimizer.zero_grad()
                for i, x in enumerate(x_batch):
                    x = x.to(device)

                    x_hat, x, mu, log_var = model(x)
                    loss = model.loss_function(x_hat.squeeze(), x.squeeze(), mu, log_var)['loss']
                    loss.backward()
                print(f"Batch {batch}, Loss {loss.item()}")
                optimizer.step()

                # Save image
                with torch.no_grad():
                    recon = model(valid_pic)[0].detach().cpu().numpy().squeeze()

                Image.fromarray(recon*255).convert("L").save(f"models/facu/images2/{model_name}/reconstructed_{epoch}_{batch}.jpeg")

            # Save model
            torch.save(model.state_dict(), f"models/facu/checkpoints/{model_name}/{epoch}.pth")

            # Update learning rate
            scheduler.step(loss)
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        # Save final model
        torch.save(model.state_dict(), f"models/facu/checkpoints/{model_name}/final.pth")