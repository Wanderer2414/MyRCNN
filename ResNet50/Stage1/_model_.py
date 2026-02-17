from torch.nn import Module, Conv2d, MaxPool2d
from torch import Tensor, device

class InitialLayer(Module):
    def __init__(self, in_channels, out_channels: int, device: device):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1, device=device)
        self.pool = MaxPool2d(3, stride=2, padding=1)
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.conv(x))
        return x