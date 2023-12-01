import torch
from autoencoder import MixedActivation
from typing import Optional, Dict

class GlobalMeanOptimizer(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, n_channels: int = 1, img_size: int = 512, network_architecture: Optional[Dict] = None):
        """ 
        Global Optimization for the SMoE description.
        """
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.n_channels = n_channels
        self.img_size = img_size

        nr_smoe_params = 3*n_kernels + n_kernels**2
        network_architecture = [2*nr_smoe_params, 2*nr_smoe_params, nr_smoe_params]

        ### Dense layers ###
        global_smoe_optimizer = []
        for out_features, in_features in zip(
            network_architecture[1:],
            network_architecture[:-1]
            ):
            global_smoe_optimizer.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            global_smoe_optimizer.append(torch.nn.LeakyReLU())
        global_smoe_optimizer.append(torch.nn.Linear(network_architecture[-1], [3*n_kernels + n_kernels**2], dtype=torch.float32))
        global_smoe_optimizer.append(MixedActivation())

        self.global_smoe_optimizer = torch.nn.Sequential(*global_smoe_optimizer)

    def forward(self, x_smoe: torch.Tensor, x_comb: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass of the GlobalMeanOptimizer model.
        """
        x = x_comb + x_smoe
        x = self.global_smoe_optimizer(x)

        return x
