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

class EncoderMLP(torch.nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 n_kernels: int = 4,
                 block_size: int = 32,
                 n_hidden_layers: int = 4,
                 device: str = "cpu",
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = 7 * n_kernels

        in_channels = block_size**2

        force_hidden_sizes = kwargs.get("force_hidden_sizes", None)
        if force_hidden_sizes is None:
            lin_modules = []
            for i in range(n_hidden_layers):
                h_dim = max(self.latent_dim, in_channels//2)
                lin_modules.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_features=h_dim),
                        torch.nn.ReLU())
                )
                in_channels = h_dim
        else:
            lin_modules = []
            for h_dim in force_hidden_sizes + [self.latent_dim]:
                lin_modules.append(
                    torch.nn.Sequential(
                        torch.nn.Linear(in_channels, out_features=h_dim),
                        torch.nn.ReLU())
                )
                in_channels = h_dim

        self.lin = torch.nn.Sequential(*lin_modules)

        self.output_nonlinearities = CustomLastLayerActivations(
            (2*n_kernels, 1*n_kernels, 4*n_kernels),
            (torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.Identity())
            )
        
        self.lin = self.lin.to(device)
        

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        x = self.lin(x)
        x = self.output_nonlinearities(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        x = self.encode(x)
        return x