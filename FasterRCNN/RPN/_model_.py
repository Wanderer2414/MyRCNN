from torch.nn import Module, Conv2d
from torch import Tensor, arange, clamp
from . import ClassificationHead, RegressionHead

class Model(Module):
    def __init__(self, channels: int, device):
        super().__init__()
        median = channels//2
        self.prepare = Conv2d(in_channels=channels, out_channels=median, kernel_size=3, stride=1, padding=1, device=device)
        self.cls = ClassificationHead.ClassificationHead(median, device)
        self.reg = RegressionHead.RegressionHead(median, device)
        self.device = device
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.prepare(x)
        cls = self.cls(x)
        reg = self.reg(x)
        i = arange(reg.shape[1], device=self.device).view(1, reg.shape[1], 1, 1)
        j = arange(reg.shape[2], device=self.device).view(1, 1, reg.shape[2], 1)
        print(reg.shape)
        print(i.shape)
        print(j.shape)
        reg[:, :, :, :, 0] += i
        reg[:, :, :, :, 1] += j
        reg = clamp(reg, min = 0, max = x.shape[2])
        return cls, reg