import os
import torch
import json
from data import DataLoader
from models.facu import VAE

n_kernels, block_size, img_size = 4, 16, 512
train_loader = DataLoader("professional_photos", img_size=img_size, block_size=block_size)
train_loader.initialize(n_repeats=5)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(n_kernels=n_kernels, block_size=block_size, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

for epoch in range(10):
    for i, (x, _) in enumerate(train_loader.training):
        x = x.to(device)
        optimizer.zero_grad()

        x_hat, x, mu, log_var = model(x)
        loss = model.loss_function(x_hat, x, mu, log_var)['loss']
        loss.backward()
        print(f"Epoch {epoch}, Batch {i}, Loss {loss.item()}")
        optimizer.step()
    scheduler.step(loss)

os.makedirs("models/facu/checkpoints", exist_ok=True)
torch.save(model.state_dict(), "models/facu/checkpoints/vae.pth")