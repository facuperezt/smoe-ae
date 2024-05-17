from functools import partial
from typing import List, Literal, Tuple, Union
import torch

from models.components.conv_blocks.conv_blocks import *

__all__ = [
    'KernelsInsideCAE',
    'KernelsOutsideCAE',
    'NegativeExpertsCAE',
    'KernelsOutsideNegativeExpertsCAE',
    'SwishKernelsInsideCAE',
    'SwishKernelsOutsideCAE',
    'SwishNegativeExpertsCAE',
    'SwishKernelsOutsideNegativeExpertsCAE',
    'GeneralCAE',
]

class CAE(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 conv_block: torch.nn.Module = ConvBlock,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = 7 * n_kernels
        if hidden_dims is None:
            # Same as Elvira's model?
            # hidden_dims = [16, 32, 64, 128, 256, 512, 1024]
            hidden_dims = [16, 32, 64]

        self._block_size = block_size
        ret = self._build_conv_layers(in_channels, hidden_dims, conv_block, **kwargs.get("conv_args", {}))
        
        if len(ret) == 2:
            self.conv, in_channels = ret
        else:
            self.conv, in_channels, block_size = ret
            self._block_size = block_size
        self.lin, in_channels = self._build_lin_layers(in_channels*block_size**2, hidden_dims, LinearBlock)

        self.encoder = torch.nn.Sequential(
            self.conv,
            torch.nn.Flatten(),
            self.lin
        )

        self.fc = torch.nn.Linear(in_channels, self.latent_dim)

        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
    
    def _build_conv_layers(
                self,
                in_channels: int,
                hidden_dims: List[int],
                conv_block: ResidualConvBlock,
                curr_block_size: int = None,
                downsample: bool = False,
                min_block_size: int = 2,
                block_size_reduction: int = 2,
            ) -> torch.nn.Sequential:
        if curr_block_size is None:
            curr_block_size = self._block_size
        conv_modules = []
        for h_dim in hidden_dims:
            if (
                    downsample and
                    in_channels < h_dim and
                    curr_block_size//block_size_reduction >= min_block_size and 
                    in_channels != 1
                ):
                curr_block_size = curr_block_size // block_size_reduction
                _stride = 2
            else:
                _stride = 1
            layer = conv_block(in_channels=in_channels, out_channels=h_dim, stride=_stride)
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

    def encode(self, input: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        # result = self.encoder(input)
        result = input
        result = self.encoder(result)
        return self.fc(result)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.encode(x)
        return x

class KernelsInsideCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = BatchNormConvBlock
        else:
            conv_block = ConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )

class KernelsOutsideCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = BatchNormConvBlock
        else:
            conv_block = ConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(-1.25, 2.5), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
class NegativeExpertsCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = BatchNormConvBlock
        else:
            conv_block = ConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), ShiftedSigmoid(-1, 3), torch.nn.Identity())
            )

class KernelsOutsideNegativeExpertsCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = BatchNormConvBlock
        else:
            conv_block = ConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(-1.25, 2.5), ShiftedSigmoid(-1, 3), torch.nn.Identity())
            )
        
class SwishKernelsInsideCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = SwishBatchNormConvBlock
        else:
            conv_block = SwishConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
class SwishKernelsOutsideCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = SwishBatchNormConvBlock
        else:
            conv_block = SwishConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(-1.25, 2.5), torch.nn.Sigmoid(), torch.nn.Identity())
            )

class SwishNegativeExpertsCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = SwishBatchNormConvBlock
        else:
            conv_block = SwishConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), ShiftedSigmoid(-1, 3), torch.nn.Identity())
            )
        
class SwishKernelsOutsideNegativeExpertsCAE(CAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 batch_norm: bool = False,
                 **kwargs) -> None:
        if batch_norm:
            conv_block = SwishBatchNormConvBlock
        else:
            conv_block = SwishConvBlock
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(-1.25, 2.5), ShiftedSigmoid(-1, 3), torch.nn.Identity())
            )
        
class GeneralCAE(CAE):
    def __init__(
            self,
            in_channels: int = 1,
            n_kernels: int = 4,
            block_size: int = 16,
            hidden_dims: List = None,
            kernels_outside: bool = False,
            negative_experts: bool = False,
            downsample: bool = False,
            batch_norm: bool = False,
            bias: bool = True,
            residual: bool = True,
            dropout: float = 0.0,
            order: str = "lbad",
            activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
            **kwargs
        ) -> None:

        ### Convolutional layers ###
        conv_block = partial(GeneralConvBlock, batch_norm=batch_norm, bias=bias, residual=residual, dropout=dropout, order=order, activation=activation)
        if downsample:
            self._build_conv_layers = partial(
                self._build_conv_layers,
                downsample=downsample,
                min_block_size=kwargs.get("min_block_size", 2),
                block_size_reduction=kwargs.get("block_size_reduction", 2)
            )
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, conv_block, **kwargs)
        
        ### Output non-linearities ###
        if kernels_outside:
            xy_nonlinearity = ShiftedSigmoid(**kwargs.get("xy_nonlinearity_params", {"shift": -1.5, "scale": 2.5}))
        else:
            xy_nonlinearity = torch.nn.Sigmoid()
        if negative_experts:
            nu_nonlinearity = ShiftedSigmoid(**kwargs.get("nu_nonlinearity_params", {"shift": -1.0, "scale": 3.0}))
        else:
            nu_nonlinearity = torch.nn.Sigmoid()
        steer_nonlinearity = torch.nn.Identity()
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (xy_nonlinearity, nu_nonlinearity, steer_nonlinearity)
            )