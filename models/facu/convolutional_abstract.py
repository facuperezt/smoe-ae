from abc import abstractmethod
import torch
from typing import Optional, Tuple, Dict
from utils import Img2Block, Block2Img

from torch_linear_assignment import batch_linear_assignment

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

    def plot_points(self, X, Y, color="red"):
        from matplotlib import pyplot as plt
        plt.figure()
        X1, Y1 = X[:, 0], X[:, 1]
        X2, Y2 = Y[:, 0], Y[:, 1]
        print(X1, Y1, X2, Y2)
        X1, Y1 = X1.detach().cpu().numpy(), Y1.detach().cpu().numpy()
        X2, Y2 = X2.detach().cpu().numpy(), Y2.detach().cpu().numpy()
        for i in range(X1.size):
            print(X1[i], Y1[i], X2[i], Y2[i])
            plt.plot(X1[i], Y1[i], 'o', color=color)
            plt.plot(X2[i], Y2[i], 'x', color=color)
            plt.plot([X1[i], X2[i]], [Y1[i], Y2[i]], color=color)
                    

    def best_match(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            pairwise_distances = torch.cdist(y, x, p=2)
            best_match = batch_linear_assignment(pairwise_distances)
            x = x[torch.arange(x.size(0))[:, None], best_match]

            # chosen_indices = -torch.ones((y.size(0), n_kernels, 2), dtype=torch.long, device=y.device)
            
            # # get the minimum distance for each kernel, cancel the row and column and repeat
            # for i in range(n_kernels):
            #     min_ind = torch.cat([(pairwise_distances[i]==torch.min(pairwise_distances[i])).nonzero() for i in range(pairwise_distances.size(0))], dim=0)
            #     chosen_indices[:, i] = min_ind
            #     pairwise_distances[torch.arange(pairwise_distances.size(0)), min_ind[:, 0], :] = 30
            #     pairwise_distances[torch.arange(pairwise_distances.size(0)), :, min_ind[:, 1]] = 30

            # # reorder the indices
            # x = x[torch.arange(x.size(0))[:, None], chosen_indices[:, :, 1]]
            # y = y[torch.arange(y.size(0))[:, None], chosen_indices[:, :, 0]]

            return x, y

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
            n_kernels = kwargs["n_kernels"]
            xy_orig = original_latent[:, :2*n_kernels]
            xy_orig = xy_orig.T.reshape(2, n_kernels, -1).transpose(0, -1)
            xy_recon = latent[:, :2*n_kernels]
            xy_recon = xy_recon.T.reshape(2, n_kernels, -1).transpose(0, -1)
            xy_orig, xy_recon = self.best_match(xy_orig, xy_recon)
            distances = torch.pow(xy_orig - xy_recon, 2).sum(dim=-1).mean()
            kernel_position_loss = distances.mean() # torch.nn.functional.mse_loss(latent[:, 2*kwargs["n_kernels"]:], original_latent[:, 2*kwargs["n_kernels"]:])
            losses = {**losses, "kernel_position_loss": kernel_position_loss.detach()}
            loss += kernel_position_loss*0.1

            # minimize the total expert value, because we don't know the order of the experts
            nu_orig = original_latent[:, 2*n_kernels:3*n_kernels].sum(dim=-1)
            nu_recon = latent[:, 2*n_kernels:3*n_kernels].sum(dim=-1)
            kernel_expert_loss = torch.nn.functional.mse_loss(nu_recon, nu_orig)
            losses = {**losses, "kernel_expert_loss": kernel_expert_loss.detach()}
            loss += kernel_expert_loss*0.05

        # latent_loss = torch.nn.functional.mse_loss(latent, original_latent)
        # losses = {**losses, "latent_loss": latent_loss.detach()}
        # loss += latent_loss
        return {"loss": loss, **losses}