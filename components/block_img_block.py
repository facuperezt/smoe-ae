import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from utils import sliding_window

__all__ = [
    'Img2Block',
    'Block2Img'
]

class BlockImgBlock(torch.nn.Module):
    def __init__(self, block_size: int, img_size: int, require_grad: bool = False):
        super().__init__()
        self.block_size = block_size
        self.img_size = img_size
        self.require_grad = require_grad

    def img_to_blocks(self, img_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        device = img_input.device
        return torch.tensor(
                sliding_window(np.asarray(img_input.detach().squeeze().cpu()), 2*[self.block_size], 2*[self.block_size], False),
                requires_grad=self.require_grad,
                dtype=torch.float32,
            ).flatten(0, -3).to(device)

    def blocks_to_img(self, blocked_input: torch.Tensor) -> torch.Tensor:
        reshape_size = (int(self.img_size/self.block_size), int(self.img_size/self.block_size), self.block_size, self.block_size)
        return blocked_input.reshape(reshape_size).permute(0, 2, 1, 3).reshape(1, self.img_size, self.img_size)

    def visualize_output(self, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
        img = self.blocks_to_img(blocked_output).detach().numpy()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)

class Img2Block(BlockImgBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.img_to_blocks(x)
    
class Block2Img(BlockImgBlock):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks_to_img(x)