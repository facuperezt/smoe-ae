from typing import Optional
import torch

from .variational_abstract import VAE_Abstract
from models.components.encoders.variational_encoder import VAE as VanillaVAE, KernelsInsideVAE, NegativeExpertsVAE, KernelsOutsideVAE, KernelsOutsideNegativeExpertsVAE
from models.components.decoders.smoe_decoder import VanillaSMoE

__all__ = [
    'VAE_KernelsInside',
    'VAE_NegativeExperts',
    'VAE_KernelsOutside',
    'VAE_KernelsOutsideNegativeExperts',
]

class VAE(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)

class VAE_KernelsInside(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsInsideVAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class VAE_NegativeExperts(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = NegativeExpertsVAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class VAE_KernelsOutside(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsOutsideVAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class VAE_KernelsOutsideNegativeExperts(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsOutsideNegativeExpertsVAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)
