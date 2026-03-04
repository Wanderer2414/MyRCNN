from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid

class ColorHead(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels*4, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=out_channels*4, out_channels=out_channels, kernel_size=1, device=device),
        )
        self.net = Sequential(
            Conv2d(in_channels=out_channels*2, out_channels=out_channels*4, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=out_channels*4, out_channels=out_channels, kernel_size=1, device=device),
        )
        self.ft = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, device=device)
        self.out_channels = out_channels
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = self.prepare(x)
        color = x
        size = 5
        while (min(color.shape[2], color.shape[3])>=size):
            color = avg_pool2d(color, kernel_size=size, stride=1, padding=size//2)
            score = self.net(cat([score, self.prepare(color)], dim=1))
            size *= 5
        score = self.ft(score)
        return score