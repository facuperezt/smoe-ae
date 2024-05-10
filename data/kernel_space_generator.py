#%%
from matplotlib import pyplot as plt
import numpy as np
import torch
from utils.visualize_kernels import plot_block_with_kernels, plot_kernel_centers, plot_kernels_chol
from models.components.decoders.smoe_decoder import VanillaSMoE

def _sample_x(n: int = 10, pad: float = 0.1, device: torch.device = torch.device("cpu")):
    return torch.tensor(np.random.uniform(0 - pad, 1 + pad, n), device=device)

def _sample_y(n: int = 10, pad: float = 0.1, device: torch.device = torch.device("cpu")):
    return torch.tensor(np.random.uniform(0 - pad, 1 + pad, n), device=device)

def _sample_nu(n: int = 10, negative_experts: bool = False, device: torch.device = torch.device("cpu")):
    return torch.tensor(np.random.uniform(-1 if negative_experts else 0, 1, n), device=device)

def _sample_chol_decomp(vmin: float = -5, vmax: float = 5, n: int = 10, include_zero: bool = False, device: torch.device = torch.device("cpu")):
    a = torch.rand(n, device=device) * (vmax - vmin) + vmin
    b = torch.rand(n, device=device) * (vmax - vmin) + vmin
    c = torch.rand(n, device=device) * (vmax - vmin) + vmin
    z = torch.zeros(n, device=device)
    if include_zero:
        return torch.stack((a, z, b, c)).T.flatten()
    return torch.stack((a, b, c)).T.flatten()

def sample_n_kernels(n: int = 10, 
                     pad: float = 0.0,  # padding for the kernel centers
                     negative_experts: bool = False,  # whether to include negative experts
                     vmin: float = -5, vmax: float = 5, include_zero: bool = False,  # range for the Cholesky decomposition
                     device: torch.device = torch.device("cpu")
                     ):
    x = _sample_x(n, pad, device)
    y = _sample_y(n, pad, device)
    nu = _sample_nu(n, negative_experts, device)
    abc = _sample_chol_decomp(vmin, vmax, n, include_zero, device)
    return torch.hstack([x, y, nu, abc])

def get_m_samples_with_n_kernels(m: int, n: int, pad: float = 0.0, negative_experts: bool = False, vmin: float = -5, vmax: float = 5, include_zero: bool = False):
    out = []
    for i in range(m):
        print(i, end="\r")
        out.append(sample_n_kernels(n, pad, negative_experts, vmin, vmax, include_zero))
    return torch.stack(out)
    return torch.stack([sample_n_kernels(n, pad, negative_experts, vmin, vmax, include_zero) for _ in range(m)])

def get_m_samples_with_n_kernels_mp(m: int, n: int, pad: float = 0.0, negative_experts: bool = False, vmin: float = -5, vmax: float = 5, include_zero: bool = False):
    # open a multiprocessing pool
    from multiprocessing import Pool, cpu_count
    with Pool(cpu_count()) as pool:
        # sample n kernels
        samples = pool.starmap(sample_n_kernels, [(n, pad, negative_experts, vmin, vmax, include_zero)]*m)

    return torch.stack(samples)
#%%

if __name__ == "__main__":
    n_kernels = 4
    block_size = 128
    t = sample_n_kernels(vmin=-block_size/5, vmax=block_size/5, n=n_kernels, negative_experts=True, include_zero=True)
    
    decoder = VanillaSMoE(n_kernels, block_size)
    plot_block_with_kernels(t, decoder(t.view(1, -1)).squeeze(), n_kernels, block_size)
# %%
