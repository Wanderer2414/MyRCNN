from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d
from torch import Tensor, device

class MaskHead(Module):
    def __init__(self, channels: int = 3, device: device = device("cpu")) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=6*channels, kernel_size=1, device=device),
            ReLU(),
            Conv2d(in_channels=6*channels, out_channels=channels, kernel_size=1, device=device)
        )
        self.avg = AvgPool2d(3, 1, 1)
    def forward(self, x:Tensor) -> Tensor:
        x = self.net(x)
        sq: Tensor = x.square()
        sum = self.avg(x)*9
        sum2 = self.avg(sq)*9
        sep:Tensor = 9*sq + sum2 - 2*x*sum
        sep = sep.sum(dim=1).unsqueeze(1)
        sep = 1-sep/sep.max()
        # sep = where(sep>0.5, ones_like(sep), zeros_like(sep))
        return sep