from torch.nn import Module, Conv2d, MaxPool2d
from torch import Tensor

class InitialLayer(Module):
    def __init__(self, in_channels, out_channels: int):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3)
        self.pool = MaxPool2d(3, stride=2)
    def forward(self, x: Tensor) -> Tensor:
        x = self.pool(self.conv(x))
        return x