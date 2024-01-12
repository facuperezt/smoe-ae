#%%
import json
import sys
import os
import itertools
import pickle
from typing import Union
from PIL import Image
import cv2

import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import sliding_window, sliding_window_torch, parse_cfg_file, get_gpu_memory_usage
from components import MixedLossFunction


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
            self.load_from_tf_smoe(f"models/saves/elvira_checkpoints/tf_smoe_weights_and_biases_{block_size}x{block_size}.pkl")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        verbose = False
        for conv in self.conv:
            if verbose:
                print("before conv")
                print(get_gpu_memory_usage(self.conv[0].weight.device))
            x = conv(x)
        for lin in self.lin:
            if verbose:
                print("before lin")
                print(get_gpu_memory_usage(self.lin[0].weight.device))
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
        block_size = self.block_size
        kernel_num = self.n_kernels

        domain_init = self.domain_init.clone()
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
    
class TorchSMoE(torch.nn.Module):
    def __init__(self, img_size: int = 512, n_kernels: int = 4, block_size: int = 8, load_tf_model: bool = False, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.img_size = img_size
        self.n_kernels = n_kernels
        self.block_size = block_size
        self.device = device
        self.ae = TorchSMoE_AE(n_kernels=n_kernels, block_size=block_size, load_tf_model=load_tf_model)
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


class Elvira2(TorchSMoE):
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
        if return_separate_losses:
            return {
                "e2e_loss": self.loss_function(x, y),
                }
        else:
            return {"e2e_loss": sum(self.loss_function(x, y).values())}
        
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
    

    def embed_artifacts_without_resize(self, x: torch.Tensor) -> torch.Tensor:
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
        self.img_size = new_w
        y = self.forward(x)

        if was_rgb:
            y = y.permute(0, 2, 3, 1).detach().cpu().numpy()
            y = np.asarray([cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) for img in y])
            y = torch.tensor(y).permute(0, 3, 1, 2).float()

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
