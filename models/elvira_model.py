import torch
import matplotlib.pyplot as plt

from components import TorchSMoE_AE_Elvira as ElviraAE, TorchSMoE_clipper as Clipper, TorchSMoE_SMoE as SMoE, MixedLossFunction, Img2Block, Block2Img

from utils.cfg_file_parser import parse_cfg_file

# class ElviraModel(torch.nn.Module):
#     def __init__(self, config_file_path: str, device: torch.device = torch.device("cpu")):
#         super().__init__()
#         self.cfg = parse_cfg_file(config_file_path)
#         self.img2block = Img2Block(**self.cfg['img2block']).to(device)
#         self.ae = ElviraAE(**self.cfg['ae']).to(device)
#         self.clipper = Clipper(**self.cfg['clipper']).to(device)
#         self.smoe = SMoE(**self.cfg['smoe']).to(device)
#         self.block2img = Block2Img(**self.cfg['block2img']).to(device)
#         self.loss_function = MixedLossFunction(**self.cfg['loss_function']).to(device)

#     def forward(self, x: torch.Tensor):
#         x_blocked: torch.Tensor = self.img2block(x)
#         x_smoe: torch.Tensor
#         x_smoe = self.ae(x_blocked)
#         x = self.clipper(x_smoe)
#         x = self.smoe(x_smoe)
#         x = self.block2img(x)

#         return x
    
#     def loss(self, x, y, return_separate_losses: bool = False):
#         if return_separate_losses:
#             return {
#                 "e2e_loss": self.loss_function(x, y),
#                 }
#         else:
#             return {"e2e_loss": sum(self.loss_function(x, y).values())}

#     def visualize_output(self, img: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
#         try:
#             imgs = iter(img)
#         except TypeError:
#             img = img.cpu().detach().numpy()
#             plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
#         else:
#             print("Too many images to visualize, only the first one will be shown.")
#             img = img[0].cpu().detach().numpy()
#             plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax) 

import itertools
import os
import pickle
from typing import Union
from PIL import Image

import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import sliding_window

class PermuteAndFlatten(torch.nn.Module):
    """
    Due to some weirdness in the training of the Keras model, it's flatten operation is "channels_first"
    which is wrong, because the actual format of the tensors is "channels_last", that does not affect the 
    training, but it is the reason why in PyTorch we need to permute the tensor in this weird way in order
    to be able to reuse the weights of the Keras implementation.
    """
    def __init__(self, start_dim: int = 1, end_dim: int = -1):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim, end_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 4:
            x = torch.permute(x, (0, 2, 3, 1))
        elif len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.flatten(x)
        return x

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
    def __init__(self, n_kernels: int = 4, block_size: int = 8, load_tf_model: bool = False):
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
            self.load_from_tf_smoe(f"saved_weights/tf_smoe_weights_and_biases_{block_size}x{block_size}.pkl")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        # x = self.conv(x)
        for conv in self.conv:
            x = conv(x)
        # x = self.lin(x)
        for lin in self.lin:
            x = lin(x)
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

class TorchSMoE_clipper(torch.nn.Module):
    """
    The center and nus clipping layer
    """
    def __init__(self, n_kernels: int = 4):
        super().__init__()
        self.n_kernels = n_kernels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        As_offset = 3 * self.n_kernels
        center_nus=x[:,0:As_offset]
        As=x[:,As_offset:]
        center_nus=torch.clip(center_nus, min=0.0, max=1.0)
        return torch.cat([center_nus,As],axis=1)

class TorchSMoE_SMoE(torch.nn.Module):
    """
    SMoE implemented in PyTorch, which allows for the gradients to be calculated with autograd
    """
    def __init__(self, n_kernels: int = 4, block_size: int = 8):
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        pass

    def forward(self, x: torch.Tensor, use_numpy: bool = False) -> np.ndarray:
        if use_numpy is False:
            return self.torch_smoe(x)
        else:
            print("Using np_smoe")
            return self.np_smoe(x.detach().numpy())


    def torch_smoe(self, arr):
        block_size = self.block_size
        kernel_num = self.n_kernels

        x = torch.linspace(0, 1, block_size, dtype=torch.float32)
        y = torch.linspace(0, 1, block_size, dtype=torch.float32)
        domain_init = torch.tensor(np.array(np.meshgrid(x, y)).T.reshape([block_size ** 2, 2]), dtype=torch.float32)
        center_x = arr[:, :kernel_num]
        center_y = arr[:, kernel_num:2 * kernel_num]
        A_NN = arr[:, 3 * kernel_num:].reshape([-1, kernel_num, 2, 2])
        A_NN = torch.tril(A_NN)
        nue_e = arr[:, 2 * kernel_num:3 * kernel_num]
        shape_x = center_x.shape
        reshape_x = center_x.view(shape_x[0], kernel_num, 1)
        reshape_y = center_y.view(shape_x[0], kernel_num, 1)
        centers = torch.cat([reshape_x, reshape_y], dim=2).view(shape_x[0], kernel_num, 2)

        musX = centers.unsqueeze(2)
        domain_exp = torch.unsqueeze(torch.unsqueeze(domain_init, dim=0), dim=0).expand(musX.shape[0], musX.shape[1], block_size * block_size, 2)
        x_sub_mu = (domain_exp - musX).unsqueeze(-1)
        n_exp = torch.exp(-0.5 * torch.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu, A_NN, A_NN, x_sub_mu))

        n_w_norm = n_exp.sum(dim=1, keepdim=True)
        n_w_norm = torch.clamp(n_w_norm, min=1e-8)

        w_e_op = n_exp / n_w_norm

        res = (w_e_op * nue_e.unsqueeze(-1)).sum(dim=1)
        res = torch.clamp(res, min=0, max=1)
        res = res.view(-1, block_size, block_size)

        return res

    @staticmethod
    def np_smoe(arr: np.ndarray) -> np.ndarray:
        #essentially smoe in numpy
        block_size = 8
        kernel_num = 4

        x = np.linspace(0, 1, block_size).astype(dtype=np.float32)
        y = np.linspace(0, 1, block_size).astype(dtype=np.float32)
        domain_init = np.array(np.meshgrid(x, y)).T
        domain_init = domain_init.reshape([block_size ** 2, 2])
        center_x = arr[:, 0:kernel_num]
        center_y = arr[:, kernel_num:2 * kernel_num]
        A_NN = arr[:, 3 * kernel_num:]
        A_NN = np.reshape(A_NN, [-1, kernel_num, 2, 2])
        A_NN = np.tril(A_NN)
        nue_e=arr[:,2*kernel_num:3*kernel_num]
        shape_x = np.shape(center_x)
        reshape_x = np.reshape(center_x, (shape_x[0], kernel_num, 1))
        reshape_y = np.reshape(center_y, (shape_x[0], kernel_num, 1))
        centers = np.reshape(np.concatenate([reshape_x, reshape_y], axis=2), (shape_x[0], kernel_num, 2))

        musX = np.expand_dims(centers, axis=2)
        domain_exp = np.tile(np.expand_dims(np.expand_dims(domain_init, axis=0), axis=0),
                            (np.shape(musX)[0], np.shape(musX)[1], 1, 1))
        x_sub_mu = np.expand_dims(domain_exp.astype(dtype=np.float32) - musX.astype(dtype=np.float32),
                                axis=-1)
        n_exp = np.exp(
            np.negative(0.5 * np.einsum('abcli,ablm,abnm,abcnj->abc', x_sub_mu, A_NN, A_NN, x_sub_mu)))

        n_w_norm = np.sum(n_exp, axis=1, keepdims=True)
        n_w_norm = np.maximum(10e-8, n_w_norm)

        w_e_op = np.divide(n_exp, n_w_norm)

        res = np.sum(w_e_op * np.expand_dims(nue_e.astype(dtype=np.float32), axis=-1), axis=1)
        res = np.minimum(np.maximum(res, 0), 1)
        res = np.reshape(res, (-1, block_size, block_size))
        # -----------------------------------------------------------------

        return res
    
class ElviraModel(torch.nn.Module):
    def __init__(self, img_size: int = 512, n_kernels: int = 4, block_size: int = 16, load_tf_model: bool = False, **kwargs):
        super().__init__()
        self.img_size = img_size
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.ae = TorchSMoE_AE(n_kernels=n_kernels, block_size=block_size, load_tf_model=load_tf_model)
        self.clipper = TorchSMoE_clipper(n_kernels=n_kernels)
        self.smoe = TorchSMoE_SMoE(n_kernels=n_kernels, block_size=block_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.img_to_blocks(x)
        x = self.ae(x)
        x = self.clipper(x)
        # x = x.to(torch.device("cpu"))
        x = self.smoe(x)
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.blocks_to_img(x)
        return x
    
    def img_to_blocks(self, img_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        img_input = img_input.squeeze()
        return torch.tensor(sliding_window(np.asarray(img_input), 2*[self.block_size], 2*[self.block_size], False)).flatten(0, -3).reshape(-1, 1, self.block_size, self.block_size)

    def blocks_to_img(self, blocked_input: torch.Tensor) -> torch.Tensor:
        reshape_size = (int(self.img_size/self.block_size), int(self.img_size/self.block_size), self.block_size, self.block_size)
        return blocked_input.reshape(reshape_size).permute(0, 2, 1, 3).reshape(self.img_size, self.img_size)

    def visualize_output(self, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
        img = self.blocks_to_img(blocked_output).detach().numpy()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    