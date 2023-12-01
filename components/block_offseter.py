#%%
import torch

__all__ = [
    'OffsetBlock'
]

class OffsetBlock(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8, n_channels: int = 1, img_size: int = 512):
        """
        Offset Block for the SMoE description.
        """
        super().__init__()
        self.block_size = block_size
        self.n_channels = n_channels
        self.n_kernels = n_kernels
        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the forward pass of the OffsetBlock model.
        """
        base_mask = torch.zeros(x.shape[0])
        x_coord_mask = base_mask.clone()
        x_coord_mask[0:self.n_kernels] = 1
        y_coord_mask = base_mask.clone()
        y_coord_mask[self.n_kernels:2*self.n_kernels] = 1
        expert_mask = base_mask.clone()
        expert_mask[2*self.n_kernels:3*self.n_kernels] = 1
        A_mask = base_mask.clone()
        A_mask[3*self.n_kernels:] = 1

        image_location = torch.arange(self.img_size**2, dtype=torch.int32)
        x[:, x_coord_mask] = x[:, x_coord_mask] + self.block_size * (image_location % self.block_size)
        x[:, y_coord_mask] = x[:, y_coord_mask] + self.block_size * (image_location // self.block_size)
        
        return x
