from torch.nn import Module, Conv2d, Sequential
from torch import Tensor, device

class FirstBottleneckBlock(Module):
    def __init__(self, in_channels, out_channels: int, device: device) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, device=device),
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, device=device),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device)
        )
        self.res = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.res(x)

class BottleneckBlock(Module):
    def __init__(self, channels, median_channels: int, device: device) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=median_channels, kernel_size=1, device=device),
            Conv2d(in_channels=median_channels, out_channels=median_channels, kernel_size=3, padding=1, device=device),
            Conv2d(in_channels=median_channels, out_channels=channels, kernel_size=1, device=device),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x
class Model(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device) -> None:
        super().__init__()
        self.net = Sequential(
            FirstBottleneckBlock(in_channels=in_channels, out_channels=out_channels, device=device),
            BottleneckBlock(channels=out_channels, median_channels=out_channels, device=device),
            BottleneckBlock(channels=out_channels, median_channels=out_channels, device=device),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)