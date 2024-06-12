from abc import abstractmethod
import torch
from typing import Optional, Tuple, Dict
from utils import Img2Block, Block2Img

class CAE_Abstract(torch.nn.Module):
    def __init__(self, n_kernels: int, block_size: int, img_size: int, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.n_kernels = n_kernels
        self.img2block = Img2Block(block_size, img_size)
        self.block2img = Block2Img(block_size, img_size)
        self.kld_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self._encoder = None
        self._decoder = None

    def forward(self, _x: torch.Tensor, return_all: bool = False, input_is_img: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _x.clone()
        if input_is_img:
            x = self.img2block(x)
        z = self.encoder(x)
        x = self.decoder(z)
        if input_is_img:
            x = self.block2img(x)
        if not return_all:
            return x
        return x, _x, z
    
    @property
    def encoder(self):
        if self._encoder is None:
            raise NotImplementedError("Encoder not implemented")
        return self._encoder

    @property
    def decoder(self):
        if self._decoder is None:
            raise NotImplementedError("Decoder not implemented")
        return self._decoder

    def loss_function(self,
                      *args,
                      **kwargs) -> Dict[str, torch.Tensor]:
        
        recons = args[0]
        input = args[1]
        latent = args[2]
        original_latent = args[3]

        losses = {}
        loss = 0
        recons_loss = (recons - input).abs().mean()
        min_recons_loss = torch.nn.functional.mse_loss(recons.flatten(start_dim=1).min(dim=1).values, input.flatten(start_dim=1).min(dim=1).values)
        max_recons_loss = torch.nn.functional.mse_loss(recons.flatten(start_dim=1).max(dim=1).values, input.flatten(start_dim=1).max(dim=1).values)
        losses = {**losses, "Reconstruction_Loss": recons_loss.detach(), "min_recons_loss": min_recons_loss.detach(), "max_recons_loss": max_recons_loss.detach()}
        loss = recons_loss + min_recons_loss*0.25 + max_recons_loss*0.25
        if kwargs.get("n_kernels", False):
            kernel_position_loss = torch.nn.functional.mse_loss(latent[:, 2*kwargs["n_kernels"]], original_latent[:, 2*kwargs["n_kernels"]])
            losses = {**losses, "kernel_position_loss": kernel_position_loss.detach()}
            loss += kernel_position_loss
        # latent_loss = torch.nn.functional.mse_loss(latent, original_latent)
        # losses = {**losses, "latent_loss": latent_loss.detach()}
        # loss += latent_loss
        return {"loss": loss, **losses}