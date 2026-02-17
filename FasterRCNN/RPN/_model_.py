from torch.nn import Module, Conv2d
from torch import Tensor, arange, clamp, cat, topk, zeros_like
from . import ClassificationHead, RegressionHead

class Model(Module):
    def __init__(self, channels: int, device):
        super().__init__()
        median = channels//2
        self.prepare = Conv2d(in_channels=channels, out_channels=median, kernel_size=3, stride=1, padding=1, device=device)
        self.cls = ClassificationHead.ClassificationHead(median, device)
        self.reg = RegressionHead.RegressionHead(median, device)
        self.device = device
    def forward(self, x: Tensor) -> Tensor:
        x = self.prepare(x)
        cls = self.cls(x)
        reg = self.reg(x)
        i = arange(reg.shape[1], device=self.device).view(1, reg.shape[1], 1, 1)
        j = arange(reg.shape[2], device=self.device).view(1, 1, reg.shape[2], 1)
        offset = zeros_like(reg)
        offset[:, :, :, :, 0] = i
        offset[:, :, :, :, 1] = j
        offset[:, :, :, :, 2] = reg[:, :, :, :, 0] + i
        offset[:, :, :, :, 3] = reg[:, :, :, :, 1] + j
        reg = reg + offset
        reg = clamp(reg, min = 0, max = x.shape[2])
        score = cat([cls, reg], dim=-1)
        B = score.shape[0]
        score = score.view(-1, 6)
        indices = score[:, 0].view(-1)
        indices = topk(indices, 3000)[1]
        score = score[indices].view(B, -1, 6)
        return score