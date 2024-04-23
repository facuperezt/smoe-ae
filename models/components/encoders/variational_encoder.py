from typing import List, Tuple
import torch

__all__ = [
    'VanillaVAE',
    'KernelsOutsideVAE',
    'NegativeExpertsVAE',
    'KernelsOutsideNegativeExpertsVAE',
    'ResidualVAE',
    'ResidualDownsamplingVAE',
]

class CustomLastLayerActivations(torch.nn.Module):
    def __init__(self, group_sizes: Tuple[int], activations: Tuple):
        super().__init__()
        self.group_sizes = group_sizes
        self.group_activation = activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        _base = 0
        for group_size, act in zip(self.group_sizes, self.group_activation):
            out.append(act(x[:, _base:_base + group_size]))
            _base += group_size
        return torch.cat(out, dim=1)
    
class ShiftedSigmoid(torch.nn.Module):
    def __init__(self, shift: float = -1, scale: float = 3):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale + self.shift

class ResidualDownsamplingConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, curr_block_size: int, kernel_size: tuple[int, int] = (3, 3), min_block_size: int = 2):
        super().__init__()
        if curr_block_size >= min_block_size/2 and in_channels < out_channels:
            stride = 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

class ResidualBatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.downsample(_x) + x
        x = self.relu(x)
        return x

class BatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class LinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc(x))

class VAE(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 conv_block: torch.nn.Module = BatchNormConvBlock,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = 7 * n_kernels
        if hidden_dims is None:
            # Same as Elvira's model?
            # hidden_dims = [16, 32, 64, 128, 256, 512, 1024]
            hidden_dims = [16, 32, 64]
            
        self.conv = self._build_conv_layers(in_channels, hidden_dims, conv_block, kwargs.get("conv_args", {}))

        # conv_modules = []
        # # Build Encoder
        # for h_dim in hidden_dims:

        #     layer = conv_block(in_channels, h_dim)

        #     conv_modules.append(layer)
        #     in_channels = h_dim

        # self.conv = torch.nn.Sequential(*conv_modules)

        in_channels *= block_size**2
        lin_modules = []
        for h_dim in hidden_dims[2:][::-1]:

            layer = LinearBlock(in_channels, h_dim)

            lin_modules.append(layer)
            in_channels = h_dim

        self.lin = torch.nn.Sequential(*lin_modules)

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
        return torch.nn.Sequential(*conv_modules)

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
    
    def forward(self, x: torch.Tensor, **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
class VanillaVAE(VAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, BatchNormConvBlock, **kwargs)

class KernelsOutsideVAE(VAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, BatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, BatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, BatchNormConvBlock, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        
class ResidualVAE(VAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
            super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualBatchNormConvBlock, **kwargs)

class ResidualDownsamplingVAE(VAE):
    def __init__(self,
                in_channels: int = 1,
                n_kernels: int = 4,
                block_size: int = 16,
                hidden_dims: List = None,
                **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualDownsamplingConvBlock, **kwargs)
