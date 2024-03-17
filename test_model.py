import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.elvira import Vanilla
from models.facu import VAE

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
    model = VAE(n_kernels=4, block_size=16, device=device)
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :512, :512]
    y = model(x)
    plt.imshow(y.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plt.show()
    print("Done")

if __name__ == "__main__":
    # test_elvira_vanilla()
    test_vanilla_VAE()
    plt.show()
