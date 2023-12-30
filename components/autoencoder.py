from dataclasses import dataclass, field
import enum
import pickle
from typing import Optional, List, Union, Dict
import torch
from .elvira_helpers import PermuteAndFlatten, TorchSMoE_clipper


__all__ = [
    'TorchSMoE_AE',
    'TorchSMoE_AE_Elvira',
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

class ConvBlock(torch.nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int, ker_size = (3,3), padd = 1, stride = 1, add_batchnorm: bool = True):
        super(ConvBlock,self).__init__()
        self.add_module('conv',torch.nn.Conv2d(in_channel ,out_channel,kernel_size=ker_size,stride=stride,padding=padd)),
        if add_batchnorm:
            self.add_module('norm',torch.nn.BatchNorm2d(out_channel)),
        self.add_module('LeakyRelu',torch.nn.LeakyReLU(0.2, inplace=True))

class LinearBlock(torch.nn.Sequential):
    def __init__(self, in_features, out_features, add_batchnorm: bool = True, dropout: float = 0.35):
        super(LinearBlock,self).__init__()
        self.add_module('linear',torch.nn.Linear(in_features, out_features, dtype=torch.float32)),
        if add_batchnorm:
            self.add_module('norm',torch.nn.BatchNorm1d(out_features)),
        self.add_module('ReLU',torch.nn.ReLU()),
        self.add_module('Dropout',torch.nn.Dropout(dropout))


class TorchSMoE_AE_Elvira(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, load_tf_model: bool = False, **kwargs):
        """
        Initializes the Conv2d and Linear (Dense in tensorflow) layers according to AE_SMoE paper.
        """
        super().__init__()
        conv_layers = []
        for out_channels, in_channels in zip([16, 32, 64, 128, 256, 512, 1024], [1, 16, 32, 64, 128, 256, 512]):
            conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, dtype=torch.float32))
            conv_layers.append(torch.nn.ReLU())
        conv_layers.append(PermuteAndFlatten())
        dense_layers = []
        for out_features, in_features in zip([1024, 512, 256, 128, 64], [1024*block_size**2, 1024, 512, 256, 128]):
            dense_layers.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            dense_layers.append(torch.nn.ReLU())
        dense_layers.append(torch.nn.Linear(64, 28, dtype=torch.float32))
        # dense_layers.append(MixedActivation(n_kernels))

        self.conv = torch.nn.Sequential(*conv_layers)
        self.lin = torch.nn.Sequential(*dense_layers)

        if load_tf_model:
            self.load_from_tf_smoe(f"models/saves/elvira_checkpoints/tf_smoe_weights_and_biases_{block_size}x{block_size}.pkl")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        # for conv in self.conv:
        #     x = conv(x)
        x = self.lin(x)
        # for lin in self.lin:
        #     x = lin(x)
        return x

    def load_from_tf_smoe(self, path_to_pkl: str) -> None:
        with open(path_to_pkl, "rb") as f:
            d = pickle.load(f)
        conv = d["conv"]
        lin = d["lin"]

        for layer, wandb in zip(self.conv[::2], conv.values()):
            weights, biases = wandb["weight"], wandb["bias"]
            layer.weight.data = weights.clone()
            layer.bias.data = biases.clone()

        for layer, wandb in zip(self.lin[::2], lin.values()):
            weights, biases = wandb["weight"], wandb["bias"]
            layer.weight.data = weights.clone()
            layer.bias.data = biases.clone()


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
        _batchnorm = network_architecture["conv"]["add_batchnorm"]
        for out_channels, in_channels in zip(
            network_architecture["conv"]["channel_sizes"][1:],
            network_architecture["conv"]["channel_sizes"][:-1]
            ):
            conv_layers.append(ConvBlock(in_channels, out_channels, ker_size=(3,3), padd=1, add_batchnorm=_batchnorm))
        conv_layers.append(torch.nn.Flatten())

        ### Dense layers ###
        means_dense_layers = []
        _batchnorm = network_architecture["lin"]["add_batchnorm"] if "add_batchnorm" in network_architecture["lin"].keys() else False
        _dropout = network_architecture["lin"]["dropout"] if "dropout" in network_architecture["lin"].keys() else 0.35
        for out_features, in_features in zip(
            network_architecture["lin"]["feature_sizes"][1:],
            network_architecture["lin"]["feature_sizes"][:-1]
            ):
            means_dense_layers.append(LinearBlock(in_features, out_features, add_batchnorm=_batchnorm, dropout=_dropout))

        ### SMoE layers ###
        fc_smoe_descriptions = []
        _batchnorm = network_architecture["smoe"]["add_batchnorm"] if "add_batchnorm" in network_architecture["smoe"].keys() else False
        _dropout = network_architecture["smoe"]["dropout"] if "dropout" in network_architecture["smoe"].keys() else 0.35
        for out_features, in_features in zip(
            network_architecture["smoe"]["feature_sizes"][1:],
            network_architecture["smoe"]["feature_sizes"][:-1]
            ):
            fc_smoe_descriptions.append(LinearBlock(in_features, out_features, add_batchnorm=_batchnorm, dropout=_dropout))
        fc_smoe_descriptions.append(torch.nn.Linear(network_architecture["smoe"]["feature_sizes"][-1], 3*n_kernels + n_kernels**2, dtype=torch.float32))
        fc_smoe_descriptions.append(MixedActivation(n_kernels))  # Final output corresponds to the postition, weight and variance of each

        ### Combiner layers ###
        fc_global_infos = []
        _batchnorm = network_architecture["combiner"]["add_batchnorm"] if "add_batchnorm" in network_architecture["combiner"].keys() else False
        _dropout = network_architecture["combiner"]["dropout"] if "dropout" in network_architecture["combiner"].keys() else 0.35
        for out_features, in_features in zip(
            network_architecture["combiner"]["feature_sizes"][1:],
            network_architecture["combiner"]["feature_sizes"][:-1]
            ):
            fc_global_infos.append(LinearBlock(in_features, out_features, add_batchnorm=_batchnorm, dropout=_dropout))
        fc_global_infos.append(torch.nn.Linear(network_architecture["combiner"]["feature_sizes"][-1], 3*n_kernels + n_kernels**2, dtype=torch.float32))
        fc_global_infos.append(torch.nn.Tanh())  # This output corresponds to an internal representation of the blocks learned by the autoencoder

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
