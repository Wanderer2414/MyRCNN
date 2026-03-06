from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid

class ColorHead(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, device=device),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False, groups=out_channels, device=device),
            # Conv2d(in_channels=out_channels*4, out_channels=out_channels, kernel_size=1, device=device),
            LeakyReLU()
        )
        self.downgrade = Sequential(
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=3, padding=1, bias=False, device=device),
            Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False, groups=out_channels, device=device),
            LeakyReLU()
        )
        # self.net = Sequential(
        #     Conv2d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=1, device=device),
        #     Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False, groups=out_channels, device=device)
            # Conv2d(in_channels=out_channels*4, out_channels=out_channels, kernel_size=1, device=device),
            # LeakyReLU()
        # )
        self.percent = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, groups=out_channels, device=device)
        self.out_channels = out_channels
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = zeros(B, self.out_channels, H, W, device=x.device)
        downgrade = self.prepare(x)
        while (min(downgrade.shape[2], downgrade.shape[3])>=3):
            downgrade = self.downgrade(downgrade)
            score = self.percent(score) + interpolate(downgrade, size=(H, W), mode="nearest")
            # score = self.net(cat([score, t], dim=1))
            # score = score - score.min()
        # score = self.net(score)
        return score