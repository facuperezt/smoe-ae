from functools import partial
from typing import List, Tuple
import torch

from models.components.conv_blocks.conv_blocks import CustomLastLayerActivations, DeepResidualConvBlock, DeepResidualDownsamplingConvBlock, LinearBlock, ResidualConvBlock, ResidualDownsamplingConvBlock, ShiftedSigmoid
from models.components.encoders.variational_encoder import VAE

__all__ = [
    'ResidualVAE',
    'ResidualKernelsOutsideVAE',
    'ResidualNegativeExpertsVAE',
    'ResidualKernelsOutsideNegativeExpertsVAE',
    'ResidualDownsamplingVAE',
    'ResidualDownsamplingKernelsOutsideVAE',
    'ResidualDownsamplingNegativeExpertsVAE',
    'ResidualDownsamplingKernelsOutsideNegativeExpertsVAE',
    'DeepResidualVAE',
    'DeepResidualKernelsOutsideVAE',
    'DeepResidualNegativeExpertsVAE',
    'DeepResidualKernelsOutsideNegativeExpertsVAE',
    'DeepResidualDownsamplingVAE',
    'DeepResidualDownsamplingKernelsOutsideVAE',
    'DeepResidualDownsamplingNegativeExpertsVAE',
    'DeepResidualDownsamplingKernelsOutsideNegativeExpertsVAE',
]

############################################################################################
#################################### ENCODERS ##############################################
############################################################################################

############################
### WITHOUT DOWNSAMPLING ###
############################

class ResidualVAE(VAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **{**kwargs, "conv_args": {"min_block_size": 2, "curr_block_size": block_size}})
        
    def _build_conv_layers(self, in_channels: int, hidden_dims: List[int], conv_block: ResidualConvBlock, curr_block_size: int, min_block_size: int = 2) -> torch.nn.Sequential:
        conv_modules = []
        for h_dim in hidden_dims:
            layer = conv_block(in_channels=in_channels, out_channels=h_dim, curr_block_size=curr_block_size, min_block_size=min_block_size)
            if in_channels < h_dim and curr_block_size//2 >= min_block_size and in_channels != 1 and type(layer) in [ResidualDownsamplingConvBlock, DeepResidualDownsamplingConvBlock]:
                curr_block_size = curr_block_size // 2
            conv_modules.append(layer)
            in_channels = h_dim
        return torch.nn.Sequential(*conv_modules), in_channels, curr_block_size
    
    def _build_lin_layers(self, in_channels: int, hidden_dims: List[int], lin_block: LinearBlock) -> torch.nn.Sequential:
        hidden_dims_lin = [h_d for h_d in hidden_dims if h_d >= self.latent_dim]
        if len(hidden_dims_lin) == 0:
            hidden_dims_lin = hidden_dims
        lin_modules = []
        for h_dim in hidden_dims_lin[::-1]:
            layer = lin_block(in_channels, h_dim)
            lin_modules.append(layer)
            in_channels = h_dim
        return torch.nn.Sequential(*lin_modules), in_channels
    
class ResidualKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
class ResidualNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

class ResidualKernelsOutsideNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

### DEEP CONV BLOCKS ###

class DeepResidualVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)

class DeepResidualKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )

class DeepResidualNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

class DeepResidualKernelsOutsideNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )        

#########################      
### WITH DOWNSAMPLING ###
#########################

class ResidualDownsamplingVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)

class ResidualDownsamplingKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
class ResidualDownsamplingNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

class ResidualDownsamplingKernelsOutsideNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(ResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

### DEEP CONV BLOCKS ###

class DeepResidualDownsamplingVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)

class DeepResidualDownsamplingKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )

class DeepResidualDownsamplingNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        
class DeepResidualDownsamplingKernelsOutsideNegativeExpertsVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        conv_block = partial(DeepResidualDownsamplingConvBlock, batch_norm=batch_norm)
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        