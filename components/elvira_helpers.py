import torch

__all__ = [
    "PermuteAndFlatten",
    "TorchSMoE_clipper"
]

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