from torch.nn import Module, Conv2d, ReLU, Sequential
from torch import Tensor, device

class ColorHead(Module):
    def __init__(self, channels: int = 3, device: device = device("cpu")) -> None:
        super().__init__()
        self.net1 = Sequential(
            Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, device=device),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device)
        )
        self.net2 = Sequential(
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device)
        )
    def forward(self, mask: Tensor, x:Tensor) -> Tensor:
        x = self.net1(x) + mask
        x[:] = x[:]/x[:].max()
        x = self.net2(x)
        return x