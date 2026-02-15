from torch.nn import Module, Conv2d, Sequential
from torch import Tensor

class FirstBottleNeckBlock(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        median_channels = in_channels//2
        self.net = Sequential(
            Conv2d(in_channels=in_channels, out_channels=median_channels, stride=2, kernel_size=1),
            Conv2d(in_channels=median_channels, out_channels=median_channels, stride=1, kernel_size=3),
            Conv2d(in_channels=median_channels, out_channels=out_channels, stride=1, kernel_size=1),
        )
        self.res = Conv2d(in_channels=in_channels, out_channels=out_channels, stride=1, kernel_size=1)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + self.res(x)
    
class BottleNeckBlock(Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        median_channels = channels//2
        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=median_channels, stride=1, kernel_size=1),
            Conv2d(in_channels=median_channels, out_channels=median_channels, stride=1, kernel_size=3),
            Conv2d(in_channels=median_channels, out_channels=channels, stride=1, kernel_size=1),
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x) + x

class Model(Module): 
    def __init__(self, in_channels: int, out_channels: int, num_layer:int) -> None:
        super().__init__()
        self.net = Sequential(
            FirstBottleNeckBlock(in_channels=in_channels, out_channels=out_channels),
            *[BottleNeckBlock(channels=out_channels) for i in range(num_layer-1)],
        )
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
        