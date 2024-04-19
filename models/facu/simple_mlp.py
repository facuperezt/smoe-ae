import torch

from models.components.encoders.mlp_encoder import EncoderMLP
from models.components.decoders.smoe_decoder import VanillaSMoE

from utils import Img2Block, Block2Img

__all__ = [
    'SimpleMLP',
]

class SimpleMLP(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 32, device: str = "cuda", **kwargs):
        super().__init__()
        self.device = device
        self.img2block = Img2Block(block_size, img_size)
        self.encoder = EncoderMLP(in_channels=1, n_kernels=n_kernels, block_size=block_size, n_hidden_layers=8, device=device, **kwargs)
        self.decoder = VanillaSMoE(n_kernels, block_size, device=device)
        self.block2img = Block2Img(block_size, img_size)

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        x = _x.clone()
        x = self.img2block(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.block2img(x).squeeze()
        return x, _x, None, None  # Just return same size because lazy
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        recons = args[0]
        input = args[1]

        recons_loss = torch.nn.functional.mse_loss(recons, input)

        loss = recons_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach()}
