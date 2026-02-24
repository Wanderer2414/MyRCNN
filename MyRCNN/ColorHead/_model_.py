from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d
from torch import Tensor, device

class ColorHead(Module):
    def __init__(self, device: device = device("cpu")) -> None:
        super().__init__()
        self.net = Sequential(
            AvgPool2d(kernel_size=5, stride=1, padding=2),
            Conv2d(in_channels=3, out_channels=64, kernel_size=1, device=device),
            AvgPool2d(kernel_size=5, stride=1, padding=2),
            ReLU(),
            Conv2d(in_channels=64, out_channels=1, kernel_size=1, device=device),
            AvgPool2d(kernel_size=5, stride=1, padding=2),
        )
    def forward(self, x:Tensor) -> Tensor:
        x = self.net(x)
        x = x/x.max()
        return x