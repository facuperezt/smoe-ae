from typing import List, Tuple
import torch

from .variational_encoder import CustomLastLayerActivations, LinearBlock, ShiftedSigmoid

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
#################################### CONV BLOCKS ############################################
############################################################################################

class ResidualDownsamplingConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, curr_block_size: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, min_block_size: int = 2):
        super().__init__()
        assert padding == kernel_size[0] // 2 if type(kernel_size) == tuple else kernel_size // 2 == padding
        if curr_block_size//2 >= min_block_size and in_channels < out_channels:
            stride = 2
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

class DeepResidualDownsamplingConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, curr_block_size: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, min_block_size: int = 2):
        super().__init__()
        assert padding == kernel_size[0] // 2 if type(kernel_size) == tuple else kernel_size // 2 == padding
        if curr_block_size//2 >= min_block_size and in_channels < out_channels and in_channels != 1:
            stride = 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.downsample(_x) + x
        x = self.relu(x)
        return x

class ResidualBatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DeepResidualBatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

############################################################################################
#################################### ENCODERS ##############################################
############################################################################################

############################
### WITHOUT DOWNSAMPLING ###
############################

class ResidualVAE(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 conv_block: torch.nn.Module = ResidualBatchNormConvBlock,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = 7 * n_kernels
        if hidden_dims is None:
            # Same as Elvira's model?
            # hidden_dims = [16, 32, 64, 128, 256, 512, 1024]
            hidden_dims = [16, 32, 64]

        self.conv, in_channels, block_size = self._build_conv_layers(in_channels, hidden_dims, conv_block=conv_block, curr_block_size=block_size)
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
        
    def _build_conv_layers(self, in_channels: int, hidden_dims: List[int], conv_block: ResidualDownsamplingConvBlock, curr_block_size: int, min_block_size: int = 2) -> torch.nn.Sequential:
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
        out = mu
        out += eps * std
        out = self.output_nonlinearities(out)
        return out
    
    def forward(self, x: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var
    
class ResidualKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualBatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualBatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualBatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualBatchNormConvBlock, **kwargs)

class DeepResidualKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualBatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualBatchNormConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualBatchNormConvBlock, **kwargs)
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
                **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualDownsamplingConvBlock, **kwargs)

class ResidualDownsamplingKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualDownsamplingConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualDownsamplingConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, ResidualDownsamplingConvBlock, **kwargs)
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
                **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualDownsamplingConvBlock, **kwargs)

class DeepResidualDownsamplingKernelsOutsideVAE(ResidualVAE):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualDownsamplingConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualDownsamplingConvBlock, **kwargs)
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
                 **kwargs) -> None:
        super().__init__(in_channels, n_kernels, block_size, hidden_dims, DeepResidualDownsamplingConvBlock, **kwargs)
        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (ShiftedSigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        