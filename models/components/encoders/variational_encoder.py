from typing import List, Tuple
import torch

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

class VanillaVAE(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 16,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = 7 * n_kernels

        conv_modules = []
        if hidden_dims is None:
            # Same as Elvira's model?
            hidden_dims = [16, 32, 64, 128, 256, 512, 1024]

        # Build Encoder
        for h_dim in hidden_dims:
            conv_modules.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 1, padding  = 1),
                    torch.nn.BatchNorm2d(h_dim),
                    torch.nn.LeakyReLU())
            )
            in_channels = h_dim

        self.conv = torch.nn.Sequential(*conv_modules)

        in_channels *= block_size**2
        lin_modules = []
        for h_dim in hidden_dims[2:][::-1]:
            lin_modules.append(
                torch.nn.Sequential(
                    torch.nn.Linear(in_channels, out_features=h_dim),
                    torch.nn.ReLU())
            )
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
            (ShiftedSigmoid(shift=-1, scale=3), torch.nn.Tanh(), torch.nn.Identity())
            )

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
    
    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        if len(input.shape) == 3:
            input = input[:, None, :, :]
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var