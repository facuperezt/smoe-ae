from typing import Optional
import torch

from .variational_abstract import VAE_Abstract
from models.components.encoders.variational_encoder import VanillaVAE, NegativeExpertsVAE, KernelsOutsideVAE, KernelsOutsideNegativeExpertsVAE
from models.components.decoders.smoe_decoder import VanillaSMoE

__all__ = [
    'VAE',
    'VAE_NegativeExperts',
    'VAE_KernelsOutside',
    'VAE_KernelsOutsideNegativeExperts',
]

class VAE(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = VanillaVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_NegativeExperts(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = NegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_KernelsOutside(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = KernelsOutsideVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_KernelsOutsideNegativeExperts(VAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = KernelsOutsideNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)
