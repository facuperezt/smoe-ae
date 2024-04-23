from typing import Optional
import torch

from .variational_abstract import VAE_Abstract
from models.components.encoders.residual_variational_encoder import ResidualVAE, ResidualDownsamplingVAE
from models.components.decoders.smoe_decoder import VanillaSMoE

__all__ = [
    "VAE_Residual",
    "VAE_Residual_Downsampling",
]

class VAE_Residual(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = ResidualVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)
    
class VAE_Residual_Downsampling(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(block_size, img_size, device)
        self._encoder = ResidualDownsamplingVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)