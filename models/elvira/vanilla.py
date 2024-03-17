import pickle
import torch
import os

from models.components.encoders.normal_encoder import ElviraVanillaAE
from models.components.decoders.smoe_decoder import VanillaSMoE

from utils import Img2Block, Block2Img

__all__ = [
    'Vanilla',
]

class Clipper(torch.nn.Module):
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

class Vanilla(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, load_tf_model: bool = False, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.img2block = Img2Block(block_size, img_size)
        self.encoder = ElviraVanillaAE(n_kernels, block_size)
        self.clipper = Clipper(n_kernels)
        self.decoder = VanillaSMoE(n_kernels, block_size, device=device)
        self.block2img = Block2Img(block_size, img_size)

        if load_tf_model:
            path = os.path.join(os.path.dirname(__file__), "checkpoints", f"vanilla_{n_kernels}-{block_size}.pkl")
            self.load_from_tf_smoe(path)
        
        self.encoder.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.img2block(x)
        x = self.encoder(x)
        x = self.clipper(x)
        x = self.decoder(x)
        x = self.block2img(x)
        return x
    
    def load_from_tf_smoe(self, path_to_pkl: str) -> None:
        with open(path_to_pkl, "rb") as f:
            d = pickle.load(f)
        conv = d["conv"]
        lin = d["lin"]

        for layer, wandb in zip(self.encoder.conv[::2], conv.values()):
            weights, biases = wandb["weight"], wandb["bias"]
            assert weights.shape == layer.weight.shape
            assert biases.shape == layer.bias.shape
            layer.weight.data = weights.clone()
            layer.bias.data = biases.clone()

        for layer, wandb in zip(self.encoder.lin[::2], lin.values()):
            weights, biases = wandb["weight"], wandb["bias"]
            assert weights.shape == layer.weight.shape
            assert biases.shape == layer.bias.shape
            layer.weight.data = weights.clone()
            layer.bias.data = biases.clone()