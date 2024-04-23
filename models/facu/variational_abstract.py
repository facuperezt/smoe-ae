from abc import abstractmethod
import torch
from typing import Optional
from utils import Img2Block, Block2Img

class VAE_Abstract(torch.nn.Module):
    def __init__(self, block_size: int, img_size: int, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.img2block = Img2Block(block_size, img_size)
        self.block2img = Block2Img(block_size, img_size)
        self._encoder = None
        self._decoder = None

    def forward(self, _x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = _x.clone()
        x = self.img2block(x)
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        x = self.block2img(x)
        return x, _x, mu, log_var
    
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
                      **kwargs) -> dict[str, torch.Tensor]:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs.get('kld_weight', 0.00025) # Account for the minibatch samples from the dataset
        recons_loss = torch.nn.functional.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}