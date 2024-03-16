#%%
import json
import sys
import os
import itertools
import pickle
import time
from typing import Union
from PIL import Image
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import sliding_window, sliding_window_torch, parse_cfg_file, get_gpu_memory_usage
from components import MixedLossFunction, GDN, RDFTConv


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
            pass
            # x = torch.permute(x, (0, 2, 3, 1))
        elif len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.flatten(x)
        return x

class MixedActivation(torch.nn.Module):
    def __init__(self, n_kernels: int = 4):
        super().__init__()
        self.n_kernels = n_kernels
        self.sigmoid = torch.nn.Sigmoid()
        self.softplus = torch.nn.Softplus()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies sigmoid to the first 3*n_kernels components of x
        which are:
        0*n_kernels to 1*n_kernels: x-position
        1*n_kernels to 2*n_kernels: y-position
        2*n_kernels to 3*n_kernels: experts
        """
        x[:, :3*self.n_kernels] = self.sigmoid(x[:, :3*self.n_kernels])
        x[:, 3*self.n_kernels:5*self.n_kernels] = torch.nn.functional.softplus(x[:, 3*self.n_kernels:5*self.n_kernels])

        return x
    
class ExtraLayerPerClass(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, group_sizes: Tuple[int, int, int] = (3, 2, 1), group_activation: Tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module] = None, device: torch.device = torch.device("cpu")):
        super().__init__()
        if group_activation is None:
            group_activation = (torch.nn.Sigmoid(), torch.nn.Softplus(), None)
        self.n_kernels = n_kernels
        self.group_sizes = group_sizes
        self.group_activation = group_activation
        self.group_activation = torch.nn.ModuleList([act if act is not None else torch.nn.Identity() for act in group_activation])

        # Build a linear layer for each group
        self.linears = torch.nn.ModuleList([torch.nn.Linear(group_size*n_kernels, group_size*n_kernels, dtype=torch.float32) for group_size in group_sizes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        _base = 0
        for i, (group_size, linear, act) in enumerate(zip(self.group_sizes, self.linears, self.group_activation)):
            out.append(act(linear(x[:, _base:_base + group_size*self.n_kernels])))
            _base += group_size*self.n_kernels
        return torch.cat(out, dim=1)

class TorchSMoE_AE(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, load_tf_model: bool = False, device: torch.device = torch.device("cpu")):
        """
        Initializes the Conv2d and Linear (Dense in tensorflow) layers according to AE_SMoE paper.
        """
        super().__init__()
        self.n_kernels = n_kernels
        corr_layers = []
        for i, (out_channels, in_chanenls) in enumerate(zip([16, 32], [1, 16])):
            corr_layers.append(RDFTConv(in_chanenls, out_channels, kernel_size=(3, 3), freq_domain_size=(2, 3), stride=2, padding=1, device=device, name=f"layer_{i}"))
            corr_layers.append(GDN(out_channels, device=device, name=f"layer_{i}:GDN"))
        conv_layers = []
        for i, (out_channels, in_channels) in enumerate(zip([64, 128, 256, 512, 1024], [32, 64, 128, 256, 512])):
            name = "conv2d" + "" if i == 0 else f"_{i}"
            conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, dtype=torch.float32))
            conv_layers[-1].name = name
            conv_layers.append(GDN(out_channels, device=device, name=name+":GDN"))
        conv_layers.append(PermuteAndFlatten())
        dense_layers = []
        for i, (out_features, in_features) in enumerate(zip([1024, 512, 256, 128, 64, 24], [1024*2**2, 1024, 512, 256, 128, 64])):
            name = "dense" + "" if i == 0 else f"_{i}"
            dense_layers.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            dense_layers[-1].name = name
            dense_layers.append(torch.nn.ReLU())
        extra_layers = []
        extra_layers.append(ExtraLayerPerClass(n_kernels, group_sizes=(3, 2, 1), group_activation=(torch.nn.Sigmoid(), torch.nn.Softplus(), None), device=device))

        self.corr = torch.nn.Sequential(*corr_layers)
        self.conv = torch.nn.Sequential(*conv_layers)
        self.lin = torch.nn.Sequential(*dense_layers)
        self.extra = torch.nn.Sequential(*extra_layers)

        if load_tf_model:
            self.load_from_tf_smoe(f"models/saves/tarek_checkpoints/tarek_tf_smoe_weights_and_biases_{block_size}x{block_size}_c.pkl")

    def forward(self, x: torch.Tensor, ret_i: int = 100) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        verbose = False
        i = -1
        for corr in self.corr:
            i += 1
            if verbose:
                print("before corr")
                print(get_gpu_memory_usage(self.corr[0].weight.device))
            x = corr(x)
            if ret_i == i:
                return x
        for conv in self.conv:
            i += 1
            if verbose:
                print("before conv")
                print(get_gpu_memory_usage(self.conv[0].weight.device))
            x = conv(x)
            if ret_i == i:
                return x
        for lin in self.lin:
            i += 1
            if verbose:
                print("before lin")
                print(get_gpu_memory_usage(self.lin[0].weight.device))
            x = lin(x)
            if ret_i == i:
                return x
        for extra in self.extra:
            i += 1
            if verbose:
                print("before extra")
                print(get_gpu_memory_usage(self.extra[0].linears[0].weight.device))
            x = extra(x)
            if ret_i == i:
                return x
        return x
    
    def load_from_tf_smoe(self, path_to_pkl: str) -> None:
        with open(path_to_pkl, "rb") as f:
            d = pickle.load(f)

        for i, (corr, gdn) in enumerate(zip(self.corr[0::2], self.corr[1::2])):
            layer = d[f"layer_{i+1}"]
            corr: RDFTConv
            gdn: GDN
            k_real, k_imag, bias = layer["kernel_real:0"].transpose(1, 0, 3, 2), layer["kernel_imag:0"].transpose(1, 0, 3, 2), layer["bias:0"]
            g_beta, g_gamma = layer[f"gdn_{i+1}_reparam_beta:0"], layer[f"gdn_{i+1}_reparam_gamma:0"]

            assert corr.kernel_real.data.shape == k_real.shape 
            corr.kernel_real.data = torch.tensor(k_real)
            assert corr.kernel_imag.data.shape == k_imag.shape
            corr.kernel_imag.data = torch.tensor(k_imag)
            assert corr.bias.data.shape == bias.shape
            corr.bias.data = torch.tensor(bias)
            assert gdn.beta.data.shape == g_beta.shape
            gdn.beta.data = torch.tensor(g_beta)
            assert gdn.gamma.data.shape == g_gamma.shape
            gdn.gamma.data = torch.tensor(g_gamma)

        for i, (conv, gdn) in enumerate(zip(self.conv[0::2], self.conv[1::2]), 10):
            _i = "" if i == 0 else f"_{i}"
            layer = d[f"conv2d{_i}"]
            conv: torch.nn.Conv2d
            gdn: GDN
            kernel, bias = layer["kernel:0"].transpose(3,2,0,1), layer["bias:0"]
            g_beta, g_gamma = layer[f"gdn_{i-7}_reparam_beta:0"], layer[f"gdn_{i-7}_reparam_gamma:0"]

            assert conv.weight.data.shape == kernel.shape
            conv.weight.data = torch.tensor(kernel)
            assert conv.bias.data.shape == bias.shape
            conv.bias.data = torch.tensor(bias)
            assert gdn.beta.data.shape == g_beta.shape
            gdn.beta.data = torch.tensor(g_beta)
            assert gdn.gamma.data.shape == g_gamma.shape
            gdn.gamma.data = torch.tensor(g_gamma)

        i = 18
        for lin in self.lin:
            _i = "" if i == 0 else f"_{i}"
            layer = d[f"dense{_i}"]
            if not isinstance(lin, torch.nn.Linear): continue
            i += 1
            lin: torch.nn.Linear
            kernel, bias = layer["kernel:0"].swapaxes(0, 1), layer["bias:0"]

            assert lin.weight.data.shape == kernel.shape
            lin.weight.data = torch.tensor(kernel)
            assert lin.bias.data.shape == bias.shape
            lin.bias.data = torch.tensor(bias)

    def load_from_tf_smoe_flat(self, path_to_pkl: str) -> None:
        with open(path_to_pkl, "rb") as f:
            d = pickle.load(f)

        for layer in self.corr[::2]:
            for name, tf_data in d["corr"]:
                layer: RDFTConv
                layer_map = {
                    "bias": layer.bias.data,
                    "imag": layer.kernel_real.data,
                    "real": layer.kernel_real.data,
                }
                for n in layer_map:
                    if n not in name:
                        continue
                    pt_data = layer_map[n]
                if not pt_data.shape == tf_data.shape:
                    continue
                pt_data = torch.tensor(tf_data)

        for layer in self.conv[::2]:
            for name, tf_data in d["conv"]:
                layer: torch.nn.Conv2d | GDN
                if isinstance(layer, torch.nn.Conv2d):
                    layer_map = {
                        "kernel": layer.weight.data,
                        "bias": layer.bias.data,
                    }
                    if "kernel" in name:
                        tf_data = tf_data.transpose(3, 2, 1, 0)
                elif isinstance(layer, GDN):
                    layer_map = {
                        "beta": layer.beta.data,
                        "gamma": layer.gamma.data,
                    }
                else:
                    pass
                for n in layer_map:
                    if n not in name:
                        continue
                    pt_data = layer_map[n]
                
                if not pt_data.shape == tf_data.shape:
                    continue
                pt_data = torch.tensor(tf_data)

        for layer in self.lin[::2]:
            for name, tf_data in d["lin"]:
                tf_data: np.ndarray
                layer: torch.nn.Linear
                layer_map = {
                    "kernel": layer.weight.data,
                    "bias": layer.bias.data,
                }
                for n in layer_map:
                    if n not in name:
                        continue
                    pt_data = layer_map[n]
                if not pt_data.shape == tf_data.shape:
                    continue
                pt_data = torch.tensor(tf_data)

class TorchSMoE_clipper(torch.nn.Module):
    """
    The center and nus clipping layer
    """
    def __init__(
            self,
            n_kernels: int = 4,
            group_sizes: Tuple[int, int, int] = (3, 2, 1),
            clip_borders: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = None,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        self.group_sizes = group_sizes
        if clip_borders is None:
            clip_borders = ((None, None), (0., 50.), (-50., 50.))
        self.clip_borders = clip_borders


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        _base = 0
        for i, (group_size, (low, high)) in enumerate(zip(self.group_sizes, self.clip_borders)):
            if low is None and high is None:
                out.append(x[:, _base:_base + group_size*self.n_kernels])
            else:
                out.append(torch.clip(x[:, _base:_base + group_size*self.n_kernels], low, high))
            _base += group_size*self.n_kernels
        return torch.cat(out, dim=1)

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x[:, 3*self.n_kernels:5*self.n_kernels] = torch.clip(x[:, 3*self.n_kernels:5*self.n_kernels], 0.0, 50)
    #     x[:, 5*self.n_kernels:] = torch.clip(x[:, 5*self.n_kernels:], -50, 50)
    #     return x
    
class TorchSMoE_SMoE(torch.nn.Module):
    """
    SMoE implemented in PyTorch, which allows for the gradients to be calculated with autograd
    """
    def __init__(self, n_kernels: int = 4, block_size: int = 8, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        x = torch.linspace(0, 1, block_size, dtype=torch.float32)
        y = torch.linspace(0, 1, block_size, dtype=torch.float32)
        self.domain_init = torch.tensor(np.array(np.meshgrid(x, y)).T.reshape([block_size ** 2, 2]), dtype=torch.float32, device=device)
        pass

    def forward(self, x: torch.Tensor, use_numpy: bool = False) -> np.ndarray:
        if use_numpy is False:
            return self.torch_smoe(x)
        else:
            print("Using np_smoe")
            return self.np_smoe(x.detach().numpy())


    def torch_smoe(self, arr):
        """Tarek's version"""
        block_size = self.block_size
        num_kernels = self.n_kernels

        domain_init = self.domain_init.clone()
        center_x = arr[:, :num_kernels]
        center_y = arr[:, num_kernels:2 * num_kernels]
        # A_NN = arr[:, 3 * kernel_num:].reshape([-1, kernel_num, 2, 2])
        # A_NN = torch.tril(A_NN)

        chol_diag = arr[:, 3*num_kernels:5*num_kernels]
        chol_diag = torch.reshape(chol_diag, (-1, num_kernels, 2))
        chol_offdiag = arr[:, 5*num_kernels:]
        chol_offdiag = torch.reshape(chol_offdiag, [-1, num_kernels, 1])
        zeros = torch.zeros_like(chol_offdiag)
        upper = torch.concatenate([chol_diag[:, :, :1], zeros], axis=-1)
        lower = torch.concatenate([chol_offdiag, chol_diag[:, :, 1:]], axis=-1)
        chol = torch.concatenate([upper, lower], axis=-1)
        chol = torch.reshape(chol, (-1, num_kernels, 2, 2))
        A_NN = chol 

        nue_e = arr[:, 2 * num_kernels:3 * num_kernels]
        shape_x = center_x.shape
        reshape_x = center_x.view(shape_x[0], num_kernels, 1)
        reshape_y = center_y.view(shape_x[0], num_kernels, 1)
        centers = torch.cat([reshape_x, reshape_y], dim=2).view(shape_x[0], num_kernels, 2)

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
    
class TorchSMoE(torch.nn.Module):
    def __init__(self, img_size: int = 512, n_kernels: int = 4, block_size: int = 8, load_tf_model: bool = False, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.img_size = img_size
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.device = device
        self.ae = TorchSMoE_AE(n_kernels=n_kernels, block_size=block_size, load_tf_model=load_tf_model, device=device)
        self.clipper = TorchSMoE_clipper(n_kernels=n_kernels)
        self.smoe = TorchSMoE_SMoE(n_kernels=n_kernels, block_size=block_size, device=device)

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        x_shape = x.shape
        if verbose:
            print("forward pass")
        if x.squeeze().ndim == 2:
            x = self.img_to_blocks(x)
        if verbose:
            print("before ae")
            print(get_gpu_memory_usage(self.device))
        x = self.ae(x)
        if verbose:
            print("before clipper")
            print(get_gpu_memory_usage(self.device))
        x = self.clipper(x)
        if verbose:
            print("before smoe")
            print(get_gpu_memory_usage(self.device))
        x = self.smoe(x)
        if verbose:
            print("before blocks_to_img")
            print(get_gpu_memory_usage(self.device))
        x = self.blocks_to_img(x)
        x = x.view(x_shape)
        return x
    
    def img_to_blocks(self, img_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return sliding_window_torch(img_input.squeeze(), 2*[self.block_size], 2*[self.block_size], False).flatten(0, -3).reshape(-1, 1, self.block_size, self.block_size)

    def blocks_to_img(self, blocked_input: torch.Tensor) -> torch.Tensor:
        reshape_size = (int(self.img_size/self.block_size), int(self.img_size/self.block_size), self.block_size, self.block_size)
        return blocked_input.reshape(reshape_size).permute(0, 2, 1, 3).reshape(self.img_size, self.img_size)

    def visualize_output(self, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1., ax: plt.Axes = None) -> None:
        if blocked_output.squeeze().ndim > 2:
            img = self.blocks_to_img(blocked_output).cpu().detach().numpy()
        else:
            img = blocked_output.squeeze().cpu().detach().numpy()
        if ax is None:
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.show()
        else:
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)


class TarekModel(TorchSMoE):
    def __init__(self, config_file_path: str, device: torch.device = torch.device("cpu"), verbose: bool = False):
        cfg = parse_cfg_file(config_file_path)
        super().__init__(img_size=cfg["ae"]["img_size"], n_kernels=cfg["ae"]["n_kernels"], block_size=cfg["ae"]["block_size"], load_tf_model=cfg["ae"]["load_tf_model"], device=device)
        self.cfg = cfg
        if verbose:
            print("before ae")
            print(get_gpu_memory_usage(self.device))
        self.ae = self.ae.to(device)
        if verbose:
            print("before clipper")
            print(get_gpu_memory_usage(self.device))
        self.clipper = self.clipper.to(device)
        if verbose:
            print("before smoe")
            print(get_gpu_memory_usage(self.device))
        self.smoe = self.smoe.to(device)
        if verbose:
            print("before loss")
            print(get_gpu_memory_usage(self.device))
        self.device = device
        self.loss_function = MixedLossFunction(**self.cfg['loss_function']).to(device)
        if verbose:
            print("after everything")
            print(get_gpu_memory_usage(self.device))

    def loss(self, x, y, return_separate_losses: bool = False):
        # MSE pytorch
        return torch.pow(x - y, 2).mean()
        
    def embed_artifacts(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input is a tensor of shape (n, c, w, h)
        Output is a tensor of shape (n, 1, w, h)
        """
        x_device = x.device
        if x.ndim < 3:
            x = x[None, :, :]
        if x.ndim < 4:
            x = x[None, :, :, :]

        was_rgb = False
        # If image is RGB, convert to grayscale using cv2
        if x.shape[1] == 3:
            was_rgb = True
            x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
            x = np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x])
            if x.ndim < 4:
                x = x[:, :, :, None]
            x = torch.tensor(x).permute(0, 3, 1, 2).float()

        w, h = x.shape[-2:]
        if (w, h) != (self.img_size, self.img_size):
            x = torch.nn.functional.interpolate(x, (self.img_size, self.img_size), mode='bilinear', align_corners=True)
        y = self.forward(x)
        if (w, h) != (self.img_size, self.img_size):
            y = torch.nn.functional.interpolate(y, (w, h), mode='bilinear', align_corners=True)

        if was_rgb:
            y = y.permute(0, 2, 3, 1).detach().cpu().numpy()
            y = np.asarray([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in y])
            y = torch.tensor(y).permute(0, 3, 1, 2).float()

        return y.to(x_device)
    

    def embed_artifacts_without_resize(self, x: torch.Tensor, scale: int = None) -> torch.Tensor:
        """
        Input is a tensor of shape (n, c, w, h)
        Output is a tensor of shape (n, 1, w, h)
        """
        if scale is not None:
            start = time.time()
        x_device = x.device
        if x.ndim < 3:
            x = x[None, :, :]
        if x.ndim < 4:
            x = x[None, :, :, :]

        was_rgb = False
        # If image is RGB, convert to grayscale using cv2
        if x.shape[1] == 3:
            was_rgb = True
            x = x.permute(0, 2, 3, 1).detach().cpu().numpy()
            x = np.asarray([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in x])
            if x.ndim < 4:
                x = x[:, :, :, None]
            x = torch.tensor(x).permute(0, 3, 1, 2).float().to(x_device)

        w, h = x.shape[-2:]
        ratio = w/h
        if ratio > 1:
            new_w = min(w, self.img_size)
            new_h = new_w
        else:
            new_h = min(h, self.img_size)
            new_w = new_h
        new_w, new_h = min(w, self.img_size), min(h, self.img_size)
        new_w -= new_w % self.block_size
        new_h -= new_h % self.block_size
        x = torch.nn.functional.interpolate(x, (new_w, new_h), mode='bilinear', align_corners=True)
        old_img_size = self.img_size
        self.img_size = new_w
        y = self.forward(x)
        self.img_size = old_img_size

        if was_rgb:
            y = y.permute(0, 2, 3, 1).detach().cpu().numpy()
            y = np.asarray([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in y])
            y = torch.tensor(y).permute(0, 3, 1, 2).float()

        if scale is not None:
            print(f"Time for embed_artifacts_without_resize at scale [{scale}]: {time.time()-start}")
        return y.to(x_device) 

    def eval(self):
        self.ae.eval()
        self.clipper.eval()
        self.smoe.eval()
        self.loss_function.eval()
        super().eval()

if __name__ == "__main__":
    g = []
    block_size = 16
    n_kernels = 4
    tsmoe = TorchSMoE(n_kernels=n_kernels, block_size=block_size, load_tf_model=True)
    tsmoe.eval()
    valid_path = f"./data/professional_photos/valid/"
    for pic in os.listdir(valid_path):
        with open(f"data/elvira/images/blocked/{block_size}x{block_size}/lena.pckl", "rb") as f:
            blocked_img = pickle.load(f)["block"]
            blocked_img = torch.tensor(blocked_img).float()[:, None, :, :]
        # img = Image.open(os.path.join(valid_path, pic)).convert('L')
        # img = torch.tensor(np.asarray(img))[:512, :512]/255.
        # img = tsmoe.img_to_blocks(img)[:, None, :, :]
        full_img = tsmoe.blocks_to_img(blocked_img)
        re_blocked_img = tsmoe.img_to_blocks(full_img)
        for rbi, bi in zip(re_blocked_img, blocked_img):
            if not torch.allclose(rbi, bi):
                print("Not equal")
                print(rbi)
                print(bi)
                break
        out = tsmoe(blocked_img)
        tsmoe.visualize_output(out,)
        plt.show()
        break
    # for block_size in [8, 16]:
    #     test_path=f'./images/blocked/{block_size}x{block_size}/'
    #     for (dirpath, dirnames, filenames) in os.walk(test_path):
    #         g.extend(filenames)
    #         break

    #     tsmoe = TorchSMoE(n_kernels=n_kernels, block_size=block_size, load_tf_model=True)
    #     #Go through all 4 testimages
    #     for i in range(4):
    #         filename = g[i]
    #         data = pickle.load(open(test_path + filename, "rb"))
    #         # mask_test=data.get('mask').tolist()
    #         block_test = data.get('block').tolist()
    #         for perm in itertools.permutations([0,1,2,3]):
    #         #     plt.figure()
    #         #     plt.title(perm)
    #         #     tsmoe.visualize_output(torch.tensor(block_test))
    #         #     break
    #             tsmoe.visualize_output(tsmoe.img_to_blocks(tsmoe.blocks_to_img(torch.tensor(block_test))))
    #         # break
    #         test_image_array = torch.asarray(block_test)[:, None, :, :]
    #         # print(test_image_array.shape)
    #         out = tsmoe(test_image_array)
    #         tsmoe.visualize_output(out)
    #         break
# %%
