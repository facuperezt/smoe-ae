from dataclasses import dataclass, field
import enum
from typing import Optional, List, Union, Dict
import torch

__all__ = [
    'TorchSMoE_AE'
]

_base_network_architecture = {
    "conv": {
        "channel_sizes": [32, 256, 512],
    },
    "lin": {
        "feature_sizes": [256, 128],
    },
    "smoe": {
        "feature_sizes": [64],
    },
    "combiner": {
        "feature_sizes": [64],
    },
}

class MixedActivation(torch.nn.Module):
    def __init__(self, n_kernels: int = 4):
        super().__init__()
        self.n_kernels = n_kernels
        self.sigmoid = torch.nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sigmoid to the first 3*n_kernels components of x
        which are:
        0*n_kernels to 1*n_kernels: x-position
        1*n_kernels to 2*n_kernels: y-position
        2*n_kernels to 3*n_kernels: experts
        """
        x[:, :3*self.n_kernels] = self.sigmoid(x[:, :3*self.n_kernels])

        return x

class TorchSMoE_AE(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, n_channels: int = 1, img_size: int = 512, network_architecture: Optional[Dict] = None):
        """
        Initializes the Conv2d and Linear (Dense in tensorflow) layers according to AE_SMoE paper.
        """
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.n_channels = n_channels
        self.img_size = img_size

        if network_architecture is None:
            network_architecture = _base_network_architecture

        network_architecture["conv"]["channel_sizes"] = [n_channels] + network_architecture["conv"]["channel_sizes"]
        network_architecture["lin"]["feature_sizes"] = [network_architecture["conv"]["channel_sizes"][-1] * block_size**2] + network_architecture["lin"]["feature_sizes"]
        network_architecture["smoe"]["feature_sizes"] = [network_architecture["lin"]["feature_sizes"][-1]] + network_architecture["smoe"]["feature_sizes"]
        network_architecture["combiner"]["feature_sizes"] = [network_architecture["lin"]["feature_sizes"][-1]] + network_architecture["combiner"]["feature_sizes"]
        
        ### Convolutional layers ###
        conv_layers = []
        for out_channels, in_channels in zip(
            network_architecture["conv"]["channel_sizes"][1:],
            network_architecture["conv"]["channel_sizes"][:-1]
            ):
            conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, dtype=torch.float32))
            conv_layers.append(torch.nn.LeakyReLU())
        conv_layers.append(torch.nn.Flatten())

        ### Dense layers ###
        means_dense_layers = []
        for out_features, in_features in zip(
            network_architecture["lin"]["feature_sizes"][1:],
            network_architecture["lin"]["feature_sizes"][:-1]
            ):
            means_dense_layers.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            means_dense_layers.append(torch.nn.ReLU())

        ### SMoE layers ###
        fc_smoe_descriptions = []
        for out_features, in_features in zip(
            network_architecture["smoe"]["feature_sizes"][1:],
            network_architecture["smoe"]["feature_sizes"][:-1]
            ):
            fc_smoe_descriptions.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            fc_smoe_descriptions.append(torch.nn.ReLU())
        fc_smoe_descriptions.append(torch.nn.Linear(network_architecture["smoe"]["feature_sizes"][-1], 3*n_kernels + n_kernels**2, dtype=torch.float32))
        fc_smoe_descriptions.append(MixedActivation(n_kernels))

        ### Combiner layers ###
        fc_global_infos = []
        for out_features, in_features in zip(
            network_architecture["combiner"]["feature_sizes"][1:],
            network_architecture["combiner"]["feature_sizes"][:-1]
            ):
            fc_global_infos.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            fc_global_infos.append(torch.nn.LeakyReLU())
        fc_global_infos.append(torch.nn.Linear(network_architecture["combiner"]["feature_sizes"][-1], 3*n_kernels + n_kernels**2, dtype=torch.float32))
        fc_global_infos.append(torch.nn.Tanh())

        self.conv = torch.nn.Sequential(*conv_layers)
        self.lin = torch.nn.Sequential(*means_dense_layers)
        self.fc_smoe_descriptions = torch.nn.Sequential(*fc_smoe_descriptions)
        self.fc_global_infos = torch.nn.Sequential(*fc_global_infos)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass of the AE_SMoE model.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.lin(x)
        x_smoe = self.fc_smoe_descriptions(x)
        x_comb = self.fc_global_infos(x)

        return x_smoe, x_comb
