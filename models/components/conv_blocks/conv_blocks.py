from logging import warning
from typing import Literal, Tuple, Union
import torch

__all__ = [
    'ResidualDownsamplingConvBlock',
    'DeepResidualDownsamplingConvBlock',
    'ResidualConvBlock',
    'DeepResidualConvBlock',
    'CustomLastLayerActivations',
    'ShiftedSigmoid',
    'BatchNormConvBlock',
    'LinearBlock',
    'ConvBlock',
    'SwishConvBlock',
    'SwishBatchNormConvBlock',
    'GeneralConvBlock',
    'GeneralLinearBlock'
]

############################################################################################
#################################### CONV BLOCKS ############################################
############################################################################################

class ResidualDownsamplingConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, curr_block_size: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, min_block_size: int = 2, batch_norm: bool = True):
        super().__init__()
        assert padding == kernel_size[0] // 2 if type(kernel_size) == tuple else kernel_size // 2 == padding
        if curr_block_size//2 >= min_block_size and in_channels < out_channels:
            stride = 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.LeakyReLU()
        else:
            self.bn = torch.nn.Identity()
            self.relu = torch.nn.ReLU()
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.downsample(_x) + x
        x = self.relu(x)
        return x

class DeepResidualDownsamplingConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, curr_block_size: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, min_block_size: int = 2, batch_norm: bool = True):
        super().__init__()
        assert padding == kernel_size[0] // 2 if type(kernel_size) == tuple else kernel_size // 2 == padding
        if curr_block_size//2 >= min_block_size and in_channels < out_channels and in_channels != 1:
            stride = 2
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.LeakyReLU()
        else:
            self.bn = torch.nn.Identity()
            self.relu = torch.nn.ReLU()
        self.downsample = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=not batch_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.downsample(_x) + x
        x = self.relu(x)
        return x

class ResidualConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, batch_norm: bool = True, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.LeakyReLU()
        else:
            self.bn = torch.nn.Identity()
            self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class DeepResidualConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, batch_norm: bool = True, **kwargs):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=not batch_norm)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        self.conv3 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=not batch_norm)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
            self.relu = torch.nn.LeakyReLU()
        else:
            self.bn = torch.nn.Identity()
            self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class CustomLastLayerActivations(torch.nn.Module):
    def __init__(self, group_sizes: Tuple[int], activations: Tuple):
        super().__init__()
        self.group_sizes = group_sizes
        self.group_activation = activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = []
        _base = 0
        for group_size, act in zip(self.group_sizes, self.group_activation):
            out.append(act(x[:, _base:_base + group_size]))
            _base += group_size
        return torch.cat(out, dim=1)
    
class ShiftedSigmoid(torch.nn.Module):
    def __init__(self, shift: float = -1, scale: float = 2):
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) * self.scale + self.shift

class BatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))

class LinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc = torch.nn.Linear(in_features, out_features)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.fc(x))
    
class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))

class SwishConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.swish = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.conv(x))
    
class SwishBatchNormConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3), stride: int = 1, padding: int = 1, bias: bool = False):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.swish = torch.nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swish(self.bn(self.conv(x)))
    
class GeneralConvBlock(torch.nn.Module):
    """
    General Convolutional Block with the following order:
    l: Layer
    b: BatchNorm
    a: Activation
    d: Dropout
    r: Residual
    
    Args:
    in_channels (int): Number of input channels
    out_channels (int): Number of output channels
    kernel_size (Tuple[int, int]): Kernel size
    stride (int): Stride
    padding (int): Padding
    batch_norm (bool): Whether to use batch normalization
    bias (bool): Whether to use bias
    residual (bool): Whether to use residual connection
    dropout (float): Dropout rate
    order (str): Order of operations
    activation (str): Activation function, one of "relu", "swish", "lrelu". Alternatively, pass an initialized Module like torch.nn.ReLU()

    Returns:
    torch.Tensor: Output tensor
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, int] = (3, 3),
                 stride: int = 1,
                 padding: int = 1,
                 batch_norm: bool = False,
                 bias: bool = False,
                 residual: bool = True,
                 dropout: float = 0.0,
                 order: str = "lbadr",
                 activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
                 ):
        super().__init__()
        if type(activation) == str:
            activation = {
                "relu": torch.nn.ReLU(),
                "swish": torch.nn.SiLU(),
                "lrelu": torch.nn.LeakyReLU()
            }[activation]
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(out_channels)
        else:
            self.bn = torch.nn.Identity()
        self.activation = activation
        if not residual:
            order = order.replace("r", "")
        if residual and in_channels != out_channels:
            self.downsample_residual = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample_residual = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.order = order
        self._apply_order_dict = {
            "l": self.conv,
            "b": self.bn,
            "a": self.activation,
            "d": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        for order in self.order:
            if order == "r":
                _x = self.downsample_residual(_x)
                x = x + _x
            else:
                x = self._apply_order_dict[order](x)
        return x
    
class GeneralLinearBlock(torch.nn.Module):
    """
    General Linear Block with the following order:
    l: Layer
    b: BatchNorm
    a: Activation
    d: Dropout
    r: Residual
    
    Args:
    in_features (int): Number of input features
    out_features (int): Number of output features
    batch_norm (bool): Whether to use batch normalization
    bias (bool): Whether to use bias
    residual (bool): Whether to use residual connection
    dropout (float): Dropout rate
    order (str): Order of operations
    activation (str): Activation function, one of "relu", "swish", "lrelu". Alternatively, pass an initialized Module like torch.nn.ReLU()
    
    Returns:
    torch.Tensor: Output tensor
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 batch_norm: bool = False,
                 bias: bool = False,
                 residual: bool = False,
                 dropout: float = 0.0,
                 order: str = "lbadr",
                 activation: Union[torch.nn.Module, Literal["relu", "swish", "lrelu"]] = "relu",
                 ):
        super().__init__()
        if type(activation) == str:
            activation = {
                "relu": torch.nn.ReLU(),
                "swish": torch.nn.SiLU(),
                "lrelu": torch.nn.LeakyReLU()
            }[activation]
        self.fc = torch.nn.Linear(in_features, out_features, bias=bias)

        if batch_norm:
            self.bn = torch.nn.BatchNorm1d(out_features)
        else:
            self.bn = torch.nn.Identity()
        self.dropout = torch.nn.Dropout(dropout)
        self.activation = activation
        if residual and in_features != out_features:
            warning("Residual connection is not possible with different input and output features. Disabling residual connection.")
            residual = False
        if not residual:
            order = order.replace("r", "")
        self.order = order
        self._apply_order_dict = {
            "l": self.fc,
            "b": self.bn,
            "a": self.activation,
            "d": self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = x
        for order in self.order:
            if order == "r":
                x = x + _x
            else:
                x = self._apply_order_dict[order](x)
        return x
