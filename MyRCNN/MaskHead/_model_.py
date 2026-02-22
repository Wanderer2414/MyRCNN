from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d
from torch import Tensor, device

class MaskHead(Module):
    def __init__(self, channels: int = 3, device: device = device("cpu")) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device)
        )
        self.avg = AvgPool2d(3, 1, 1)
    def forward(self, x:Tensor) -> Tensor:
        sq: Tensor = x.square()
        sum = self.avg(x)
        sum2 = self.avg(sq)*9
        sep:Tensor = 9*sq + sum2 - 2*x*sum
        sep = sep.sum(dim=1)
        sep = 1-sep/sep.max()
        return self.net(sep)