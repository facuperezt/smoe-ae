import torch

from models.components.encoders.variational_encoder import VanillaVAE
from models.components.decoders.smoe_decoder import VanillaSMoE

from utils import Img2Block, Block2Img

__all__ = [
    'VAE',
]

class VAE(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, img_size: int = 512, load_tf_model: bool = False, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.img2block = Img2Block(block_size, img_size)
        self.encoder = VanillaVAE(1, n_kernels, block_size, None).to(device)
        self.decoder = VanillaSMoE(n_kernels, block_size, device=device)
        self.block2img = Block2Img(block_size, img_size)

    def forward(self, _x: torch.Tensor) -> torch.Tensor:
        x = _x.clone()
        x = self.img2block(x)
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        x = self.block2img(x)
        return x, _x, mu, log_var
    
    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
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
