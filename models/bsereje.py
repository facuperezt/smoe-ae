import json
from matplotlib import pyplot as plt
import torch
from components import TorchSMoE_SMoE as SMoE, OffsetBlock, GlobalMeanOptimizer, TorchSMoE_VAE as AE,\
        Img2Block, Block2Img, MixedLossFunction, PositionalEncodingPermute1D as PositionalEncoding

from utils.cfg_file_parser import parse_cfg_file


class SampleFromDistribution(torch.nn.Module):
    def __init__(self, dist: str = "Normal"):
        super().__init__()
        self.dist = torch.normal

    def forward(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(logvar/2)  # log-var trick
        return self.dist(mean, std)

class BserejePipeline(torch.nn.Module):
    def __init__(self, config_file_path: str, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.cfg = parse_cfg_file(config_file_path)
        self.img2block = Img2Block(**self.cfg['img2block']).to(device)
        self.ae = AE(**self.cfg['ae']).to(device)
        self.sample = SampleFromDistribution("Normal")
        self.smoe = SMoE(**self.cfg['smoe']).to(device)
        self.block2img = Block2Img(**self.cfg['block2img']).to(device)
        

    def forward(self, x: torch.Tensor):
        x_blocked: torch.Tensor = self.img2block(x)
        z_mean: torch.Tensor
        z_logvar: torch.Tensor
        z_mean, z_logvar = self.ae(x_blocked)
        x = self.sample(z_mean, z_logvar)
        x = self.smoe(x)
        x = self.block2img(x)

        return x, z_mean, z_logvar
    
    def loss(self, x, y, z_mean, z_log_var) -> torch.Tensor:
        """
        params:
        x <-- Reconstructed image
        y <-- Original image
        z <-- latent space distribution parameters
        """
        reconstruction_loss = torch.nn.functional.mse_loss(x, y)  # MSE
        kl_div = -0.5 * torch.sum(
            1 + z_log_var - z_mean**2 - torch.exp(z_log_var),  # 1 + log(sigma^2) - mu^2 - sigma^2
            dim=-1
        ).mean()

        return 10*reconstruction_loss + kl_div

    def old_loss(self, x, y, return_separate_losses: bool = False):
        if return_separate_losses:
            dim0 = self.x_blocked.shape[0]
            return {
                "e2e_loss": self.loss_function(x, y),
                "blockwise_loss": {
                    key: value.sum(dim=[1,2]).reshape(int(dim0**0.5), int(dim0**0.5)) 
                    for key, value in self.blockwise_loss_function(self.x_smoe_reconst, self.x_blocked).items()
                    }
            }
        return {"e2e_loss": sum(self.loss_function(x, y).values()), "blockwise_loss": sum(self.blockwise_loss_function(self.x_smoe_reconst, self.x_blocked).values())}
    
    def visualize_output(self, img: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
        try:
            imgs = iter(img)
        except TypeError:
            img = img.cpu().detach().numpy()
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            print("Too many images to visualize, only the first one will be shown.")
            img = img[0].cpu().detach().numpy()
            plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax) 