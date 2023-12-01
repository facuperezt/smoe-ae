import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from utils import sliding_window

class BlockImgBlock(torch.nn.Module):
    def img_to_blocks(self, img_input: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        return torch.tensor(sliding_window(np.asarray(img_input), 2*[self.block_size], 2*[self.block_size], False))

    def blocks_to_img(self, blocked_input: torch.Tensor) -> torch.Tensor:
        reshape_size = (int(self.img_size/self.block_size), int(self.img_size/self.block_size), self.block_size, self.block_size)
        return blocked_input.reshape(reshape_size).permute(0, 2, 1, 3).reshape(self.img_size, self.img_size)

    def visualize_output(self, blocked_output: torch.Tensor, cmap: str = 'gray', vmin: float = 0., vmax: float = 1.) -> None:
        img = self.blocks_to_img(blocked_output).detach().numpy()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)