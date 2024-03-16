import torch

class RDFTConv(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int] = (3, 3), freq_domain_size: tuple[int, int] = (3, 2), corr: bool = True,
                 stride: int = 2, padding: int = 1, device: torch.device = "cpu", name: str = ""):
        """
        I am fking guessing how this works in TF under the hood. But this in theory should work.
        """
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.freq_size = freq_domain_size
        self.stride = stride
        self.padding = padding
        self.corr = corr
        self.device = device
        self.kernel_real = torch.nn.Parameter(torch.randn((out_channels, in_channels, *freq_domain_size)).to(device))
        self.kernel_imag = torch.nn.Parameter(torch.randn((out_channels, in_channels, *freq_domain_size)).to(device))
        self.bias = torch.nn.Parameter(torch.randn(out_channels).to(device))

    def forward(self, x):
        kernel = torch.fft.irfft2(torch.complex(self.kernel_real, self.kernel_imag).to(self.device), self.kernel_size)
        if not self.corr:
            kernel = kernel.flip(-2, -1)
        x = torch.conv2d(x, kernel, self.bias, self.stride, padding=self.padding)
        return x
    
    def __repr__(self):
        return f"RDFTConv({self.in_channels}, {self.out_channels}, {self.kernel_size=})"