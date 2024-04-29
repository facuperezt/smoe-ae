from typing import Optional
import torch

from .variational_abstract import VAE_Abstract
from models.components.encoders.residual_variational_encoder import ResidualVAE, ResidualNegativeExpertsVAE, ResidualKernelsOutsideVAE, ResidualKernelsOutsideNegativeExpertsVAE, \
                    ResidualDownsamplingVAE, ResidualDownsamplingNegativeExpertsVAE, ResidualDownsamplingKernelsOutsideVAE, ResidualDownsamplingKernelsOutsideNegativeExpertsVAE, \
                    DeepResidualVAE, DeepResidualNegativeExpertsVAE, DeepResidualKernelsOutsideVAE, DeepResidualKernelsOutsideNegativeExpertsVAE, \
                    DeepResidualDownsamplingVAE, DeepResidualDownsamplingNegativeExpertsVAE, DeepResidualDownsamplingKernelsOutsideVAE, DeepResidualDownsamplingKernelsOutsideNegativeExpertsVAE
from models.components.decoders.smoe_decoder import VanillaSMoE

__all__ = [
    "VAE_Residual",
    "VAE_Residual_Downsampling",
    "VAE_Residual_KernelsOutside",
    "VAE_Residual_NegativeExperts",
    "VAE_Residual_KernelsOutsideNegativeExperts",
    "VAE_Residual_Downsampling_KernelsOutside",
    "VAE_Residual_Downsampling_NegativeExperts",
    "VAE_Residual_Downsampling_KernelsOutsideNegativeExperts",
    "VAE_Residual_DeepConv",
    "VAE_Residual_DeepConv_NegativeExperts",
    "VAE_Residual_DeepConv_KernelsOutside",
    "VAE_Residual_DeepConv_KernelsOutsideNegativeExperts",
    "VAE_Residual_DeepConv_Downsampling",
    "VAE_Residual_DeepConv_Downsampling_NegativeExperts",
    "VAE_Residual_DeepConv_Downsampling_KernelsOutside",
    "VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts",
]

class VAE_Residual(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_NegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_KernelsOutside(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualKernelsOutsideVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_KernelsOutsideNegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualKernelsOutsideNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)
    
class VAE_Residual_Downsampling(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualDownsamplingVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_Downsampling_NegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualDownsamplingNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_Downsampling_KernelsOutside(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualDownsamplingKernelsOutsideVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_Downsampling_KernelsOutsideNegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = ResidualDownsamplingKernelsOutsideNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_NegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_KernelsOutside(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualKernelsOutsideVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_KernelsOutsideNegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualKernelsOutsideNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_Downsampling(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualDownsamplingVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_Downsampling_NegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualDownsamplingNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_Downsampling_KernelsOutside(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualDownsamplingKernelsOutsideVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)

class VAE_Residual_DeepConv_Downsampling_KernelsOutsideNegativeExperts(VAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)
        self._encoder = DeepResidualDownsamplingKernelsOutsideNegativeExpertsVAE(1, n_kernels, block_size, hidden_dims=hidden_dims).to(device)
        self._decoder = VanillaSMoE(n_kernels, block_size, device=device)