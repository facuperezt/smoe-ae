from typing import List, Tuple
import torch
from torch.nn.modules import Module

from models.components.conv_blocks.conv_blocks import BatchNormConvBlock, ConvBlock, CustomLastLayerActivations, DeepResidualDownsamplingConvBlock, LinearBlock, ResidualDownsamplingConvBlock, ShiftedSigmoid, SwishBatchNormConvBlock, SwishConvBlock

__all__ = [
    'VanillaVAE',
    'KernelsOutsideVAE',
    'NegativeExpertsVAE',
    'KernelsOutsideNegativeExpertsVAE',
    'ResidualVAE',
    'ResidualDownsamplingVAE',
]

class VAE(torch.nn.Module):
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

        self.fc_mu = torch.nn.Linear(in_channels, self.latent_dim)
        self.fc_var = torch.nn.Linear(in_channels, self.latent_dim)

        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
    def _build_conv_layers(self, in_channels: int, hidden_dims: List[int], conv_block: int, **conv_args) -> torch.nn.Module:
        conv_modules = []
        for h_dim in hidden_dims:
            layer = conv_block(in_channels, h_dim, **conv_args)
            conv_modules.append(layer)
            in_channels = h_dim
        return torch.nn.Sequential(*conv_modules), in_channels
    
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
        for layer in self.encoder:
            result = layer(result)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        out = eps * std + mu
        out = self.output_nonlinearities(out)
        return out
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

class KernelsInsideVAE(VAE):
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

class KernelsOutsideVAE(VAE):
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
            (ShiftedSigmoid(), torch.nn.Sigmoid(), torch.nn.Identity())
            )
        
class NegativeExpertsVAE(VAE):
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
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )

class KernelsOutsideNegativeExpertsVAE(VAE):
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
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        