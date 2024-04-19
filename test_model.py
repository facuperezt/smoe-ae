import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.elvira import Vanilla
from models.facu import VAE, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP

from analyze_models import load_model

def test_elvira_vanilla():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Vanilla(n_kernels=4, block_size=16, load_tf_model=True, device=device)
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :512, :512]
    y = model(x)
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

def test_vanilla_VAE():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("vae")
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :512, :512]
    y = model(x)
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

def test_VAE_KernelsOutside():
    n_kernels, block_size, img_size = 4, 16, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model("vae_kernels_outside")
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :img_size, :img_size]
    y = model(x)[0]
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

def test_VAE_NegativeExperts():
    n_kernels, block_size, img_size = 4, 16, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE_NegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device)
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :img_size, :img_size]
    y = model(x)
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

def test_VAE_KernelsOutsideNegativeExperts():
    n_kernels, block_size, img_size = 4, 16, 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE_KernelsOutsideNegativeExperts(n_kernels=n_kernels, block_size=block_size, img_size=img_size, device=device)
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :img_size, :img_size]
    y = model(x)
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

if __name__ == "__main__":
    # test_elvira_vanilla()
    test_VAE_NegativeExperts()
    plt.show()
