import torch

__all__ = [
    'ElviraVanillaAE',
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

class ElviraVanillaAE(torch.nn.Module):
    def __init__(self, n_kernels: int = 4, block_size: int = 8):
        """
        Initializes the Conv2d and Linear (Dense in tensorflow) layers according to AE_SMoE paper.
        """
        super().__init__()
        self.n_kernels = n_kernels
        self.block_size = block_size
        conv_layers = []
        for out_channels, in_channels in zip([16, 32, 64, 128, 256, 512, 1024], [1, 16, 32, 64, 128, 256, 512]):
            conv_layers.append(torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1, dtype=torch.float32))
            conv_layers.append(torch.nn.ReLU())
        conv_layers.append(PermuteAndFlatten())
        
        dense_layers = []
        for out_features, in_features in zip([1024, 512, 256, 128, 64], [1024*block_size**2, 1024, 512, 256, 128]):
            dense_layers.append(torch.nn.Linear(in_features, out_features, dtype=torch.float32))
            dense_layers.append(torch.nn.ReLU())
        dense_layers.append(torch.nn.Linear(64, 7*n_kernels, dtype=torch.float32))

        self.conv = torch.nn.Sequential(*conv_layers)
        self.lin = torch.nn.Sequential(*dense_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assumes x is of shape (batch_size, n_ch, height, width)
        """
        if len(x.shape) == 3:
            x = x[:, None, :, :]
        x = self.conv(x)
        x = self.lin(x)
        return x