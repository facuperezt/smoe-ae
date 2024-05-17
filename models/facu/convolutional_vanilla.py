from typing import List, Literal, Optional, Union
import torch

from .convolutional_abstract import CAE_Abstract
from models.components.encoders.convolutional_encoder import *
from models.components.decoders.smoe_decoder import VanillaSMoE

__all__ = [
    'CAE_KernelsInside',
    'CAE_NegativeExperts',
    'CAE_KernelsOutside',
    'CAE_KernelsOutsideNegativeExperts',
    'CAE_SwishKernelsInside',
    'CAE_SwishNegativeExperts',
    'CAE_SwishKernelsOutside',
    'CAE_SwishKernelsOutsideNegativeExperts',
    'CAE_Codec',
]

class CAE(CAE_Abstract):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, device: str = "cuda"):
        super().__init__(n_kernels, block_size, img_size, device)

class CAE_KernelsInside(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsInsideCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_NegativeExperts(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = NegativeExpertsCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_KernelsOutside(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsOutsideCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_KernelsOutsideNegativeExperts(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = KernelsOutsideNegativeExpertsCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_SwishKernelsInside(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = SwishKernelsInsideCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_SwishNegativeExperts(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = SwishNegativeExpertsCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_SwishKernelsOutside(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = SwishKernelsOutsideCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_SwishKernelsOutsideNegativeExperts(CAE):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, hidden_dims: Optional[list] = None, batch_norm: bool = True, device: str = "cuda"):
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = SwishKernelsOutsideNegativeExpertsCAE(in_channels=1, n_kernels=n_kernels, block_size=block_size, hidden_dims=hidden_dims, batch_norm=batch_norm).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)

class CAE_Codec(CAE):
    def __init__(
        self,
        n_kernels: int = 4,
        block_size: int = 16,
        img_size: int = 512,
        hidden_dims: List = None,
        kernels_outside: bool = False,
        negative_experts: bool = False,
        downsample: bool = False,
        batch_norm: bool = False,
        bias: bool = True,
        residual: bool = False,
        dropout: float = 0.0,
        order: str = "lbadr",
        activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
        device: str = "cuda",
        **kwargs
    ) -> None:
        super().__init__(n_kernels=n_kernels, block_size=block_size, img_size=img_size, hidden_dims=hidden_dims, device=device)
        self._encoder = GeneralCAE(
            in_channels=1,
            n_kernels=n_kernels,
            block_size=block_size,
            hidden_dims=hidden_dims,
            kernels_outside=kernels_outside,
            negative_experts=negative_experts,
            downsample=downsample,
            batch_norm=batch_norm,
            bias=bias,
            residual=residual,
            dropout=dropout,
            order=order,
            activation=activation,
            **kwargs
        ).to(device)
        self._decoder = VanillaSMoE(n_kernels=n_kernels, block_size=block_size, device=device)
