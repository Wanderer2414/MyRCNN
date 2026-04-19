from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d
from torch import Tensor, device, tensor, float32 as tfloat
from torch.nn.functional import max_pool2d, conv2d

class MaskHead(Module):
    def __init__(self, channels: int = 3) -> None:
        super().__init__()
        self.Gx = tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=tfloat
        ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)

        self.Gy = tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=tfloat
        ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        self.Gx = self.Gx.to(device)
        self.Gy = self.Gy.to(device)
        return self
    def forward(self, x:Tensor) -> Tensor:
        Gx = conv2d(x, self.Gx, padding=1, stride=1, groups=3)
        Gy = conv2d(x, self.Gy, padding=1, stride=1, groups=3)
        x = (Gx * Gx + Gy * Gy).sqrt()
        x = x.sum(dim=1, keepdim=True)
        x = max_pool2d(x, 3, 1, 1)
        return x