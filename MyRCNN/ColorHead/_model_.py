from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, Parameter
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu

class ColorHead(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=4, kernel_size=1, bias=False, device=device),
            Conv2d(in_channels=4, out_channels=32, kernel_size=1, groups=4, device=device)
        )
        self.firstscore = Sequential(
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False, device=device),
            LeakyReLU()
        )
        self.downgrade1 = Sequential(
            Conv2d(in_channels=32, out_channels=160, kernel_size=3, stride=3, padding=1, groups=32, device=device, padding_mode="replicate"),
            LeakyReLU()
        )
        self.downgrade2 = Sequential(
            Conv2d(in_channels=160, out_channels=32, kernel_size=1, groups=32, device=device)
        )
        self.score1 = Sequential(
            Conv2d(in_channels=160, out_channels=320, kernel_size=1, groups=160, bias=False, device=device),
            LeakyReLU()
        )
        self.score2 = Sequential(
            Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1, bias=False, device=device)
        )
        self.weight = Parameter(ones(1, device=device))
        self.out_channels = out_channels
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = self.firstscore(x)
        downgrade: Tensor = self.prepare(x)
        while (min(downgrade.shape[2], downgrade.shape[3])>=3):
            downgrade = self.downgrade1(downgrade)
            sc:Tensor = self.score1(downgrade)
            sc = sc.view(B, 32, 10, sc.shape[-2], sc.shape[-1])
            even = sc[:, :, 0::2, :, :].max(dim=2).values
            odd = sc[:, :, 1::2, :, :].max(dim=2).values
            sc = cat([even, odd], dim=1)
            score = score*self.weight + interpolate(self.score2(sc), size=(H, W), mode="nearest")
            downgrade = self.downgrade2(downgrade)
            # score = self.net(cat([score, t], dim=1))
            # score = score - score.min()
        # score = self.net(score)
        return score