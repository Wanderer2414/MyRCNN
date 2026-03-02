from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid

class ColorHead(Module):
    def __init__(self, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=4, out_channels=out_channels*2, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, device=device),
        )
        self.ft = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, device=device)
        self.out_channels = out_channels
    def forward(self, mask: Tensor, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = zeros((B, self.out_channels, H, W), device=x.device)
        mask = max_pool2d(1-mask, kernel_size=5, stride=1, padding=2)
        color = cat([mask, x], dim=1)
        color = self.net(color)
        size = 5
        while (min(color.shape[2], color.shape[3])>=size):
            color = avg_pool2d(color, kernel_size=size, stride=1, padding=size//2)
            score = score + color
            size *= 5
        score = self.ft(score)
        return score