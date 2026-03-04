from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d
from torch import Tensor, device, tensor, float32 as tfloat
from torch.nn.functional import max_pool2d, conv2d

class MaskHead(Module):
    def __init__(self, channels: int = 3, device: device = device("cpu")) -> None:
        super().__init__()
        self.Gx = tensor(
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]], dtype=tfloat, device=device
        ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)

        self.Gy = tensor(
            [[-1, -2, -1],
             [ 0,  0,  0],
             [ 1,  2,  1]], dtype=tfloat, device=device
        ).view(1, 1, 3, 3).expand(channels, 1, 3, 3)

        self.net = Sequential(
            Conv2d(in_channels=channels, out_channels=6*channels, kernel_size=1, device=device),
            ReLU(),
            Conv2d(in_channels=6*channels, out_channels=channels, kernel_size=1, device=device)
        )
        self.avg = AvgPool2d(3, 1, 1)
    def forward(self, x:Tensor) -> Tensor:
        Gx = conv2d(x, self.Gx, padding=1, stride=1, groups=3)
        Gy = conv2d(x, self.Gy, padding=1, stride=1, groups=3)
        x = (Gx * Gx + Gy * Gy).sqrt()
        x = x.sum(dim=1, keepdim=True)
        x = max_pool2d(x, 3, 1, 1)
        return x