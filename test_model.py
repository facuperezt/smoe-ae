#%%
%load_ext autoreload
%autoreload 2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from models.elvira import Vanilla
from models.facu import VAE, VAE_KernelsOutside, VAE_NegativeExperts, VAE_KernelsOutsideNegativeExperts, SimpleMLP, VAE_Residual_DeepConv_Downsampling

from analyze_models_old import load_model
from utils import plot_kernels_chol, plot_kernel_centers, shade_kernel_areas

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

def test_block_reconstruction():
    n_kernels, block_size, img_size = 3, 8, 256
    hidden_dims = [16, 16, 64, 64, 256, 256, 256, 256]
    model_name = "residual_deep_conv_vae_kernels_inside_downsampling"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE_Residual_DeepConv_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
    model.load_state_dict(torch.load(f"models/facu/checkpoints/{model_name}/final.pth"))
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :img_size, :img_size]
    y = model(x)
    no_vars = model.block2img(model.decoder(y[2]))
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    axs[0][0].imshow(y[0].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    axs[0][1].imshow(y[1].detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    axs[1][0].imshow(no_vars.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    plot_kernel_centers(y[2][0], axs[1][1], block_size, n_kernels=n_kernels)
    plot_kernels_chol(y[2][0], axs[1][1], block_size, n_kernels=n_kernels)
    fig.show()
    print("Done")

def plot_distribution(_mu):
    xs = _mu[:, :3].flatten()
    ys = _mu[:, 3:6].flatten()
    nus = _mu[:, 6].flatten()
    _xs = np.concatenate([xs, np.array([0, 0, 1, 1])])
    _ys = np.concatenate([ys, np.array([0, 1, 0, 1])])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].hexbin(_xs, _ys, gridsize=(200, 200))
    axs[0].axis('off')
    axs[1].hist(nus, 100, range=(0, 1), density=True)

# %%
if __name__ == "__main__":
    n_kernels, block_size, img_size = 3, 8, 256
    hidden_dims = [16, 16, 64, 64, 256, 256, 256, 256]
    model_name = "residual_deep_conv_vae_kernels_inside_downsampling"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VAE_Residual_DeepConv_Downsampling(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
    model.load_state_dict(torch.load(f"models/facu/checkpoints/{model_name}/final.pth"))
    model.eval()
    img = Image.open("data/professional_photos/train/ali-inay-2273.png").convert("L")
    img = (torch.tensor(np.array(img), dtype=torch.float32, device=device, requires_grad=False)/255)
    x = img[None, :img_size, :img_size]
    y = model(x, return_all=True)
    _mu = model.encoder.output_nonlinearities(y[2])
    # _mu = y[2]
    _z = y[4]
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))

    i = -1
    recon, orig, mu, z = model.img2block(y[0])[i], model.img2block(y[1])[i], _mu[i], _z[i]
    z_img = model.decoder(z.view(1, -1))
    axs[0][0].imshow(orig.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    axs[0][1].imshow(recon.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
    axs[1][1].imshow(z_img.detach().cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)


    pd01 = [1, 0]
    pd02 = [0, 1]
    l01 = .1
    l02 = .1
    P0 = np.array([pd01, pd02])
    D0 = np.array([[l01, 0], [0, l02]])
    chol0 = np.linalg.cholesky(P0@D0@np.linalg.inv(P0)).flatten()
    pos0 = [0, 0.5]
    nu0 = -1

    pd11 = [1, 0]
    pd12 = [0, 1]
    l11 = .1
    l12 = .1
    P1 = np.array([pd11, pd12])
    D1 = np.array([[l11, 0], [0, l12]])
    chol1 = np.linalg.cholesky(P1@D1@np.linalg.inv(P1)).flatten()
    pos1 = [0.5, 0.5]
    nu1 = 0

    pd21 = [1, 1]
    pd22 = [0, 1]
    l21 = 5
    l22 = 5
    P2 = np.array([pd21, pd22])
    D2 = np.array([[l21, 0], [0, l22]])
    chol2 = np.linalg.cholesky(P2@D2@np.linalg.inv(P2)).flatten()
    pos2 = [1, 0.5]
    nu2 = 2

    # manually override mu
    mu = torch.tensor([[pos0[0], pos1[0], pos2[0], pos0[1], pos1[1], pos2[1], nu0, nu1, nu2, *chol0, *chol1, *chol2]]).squeeze()
    mu = _mu.mean(dim=0)
    mu_img = model.decoder(mu.cuda().view(1, -1))
    axs[1][0].imshow(mu_img.detach().cpu().numpy().squeeze().T, cmap='gray', vmin=0, vmax=1)

    plot_kernel_centers(mu, axs[1][0], block_size, n_kernels=n_kernels)
    plot_kernels_chol(mu, axs[1][0], block_size, n_kernels=n_kernels)
    plot_kernel_centers(z, axs[1][1], block_size, n_kernels=n_kernels)
    plot_kernels_chol(z, axs[1][1], block_size, n_kernels=n_kernels)
    #%%
    %load_ext autoreload
    %autoreload 2
    shade_kernel_areas(mu, None, block_size, mu_img.squeeze(), n_kernels=n_kernels)

    print("Done")
    # %%
    from zennit.composites import EpsilonGammaBox
    from zennit.canonizers import SequentialMergeBatchNorm
    from zennit.attribution import Gradient, IntegratedGradients

    device = model.device
    canonizers = [SequentialMergeBatchNorm()]
    composite = EpsilonGammaBox(low=-3., high=3., canonizers=canonizers)
    _model = model.cpu()
    with Gradient(model=_model, composite=composite) as attributor:
        _, y_rel = attributor(x.cpu(), np.zeros_like(y.cpu().detach()))
    # %%

