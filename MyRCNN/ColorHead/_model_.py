from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, Parameter
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu

class ColorHead(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.merge = tensor([1/256/256, 1/256, 1], device=device).view(1, 3, 1, 1)
        self.bias = Parameter(zeros(64, device=device)).view(1, 64, 1, 1)
        self.downgrade1 = Sequential(
            Conv2d(in_channels=64, out_channels=320, kernel_size=3, stride=3, padding=1, groups=64, device=device, padding_mode="replicate"),
            ReLU()
        )
        self.downgrade2 = Sequential(
            Conv2d(in_channels=320, out_channels=64, kernel_size=1, groups=64, device=device)
        )
        self.score1 = Sequential(
            Conv2d(in_channels=320, out_channels=640, kernel_size=1, groups=320, bias=False, device=device),
            ReLU()
        )
        self.score2 = Sequential(
            Conv2d(in_channels=128, out_channels=out_channels, kernel_size=1, bias=False, device=device)
        )
        self.out_channels = out_channels
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = zeros(B, self.out_channels, H, W, device=x.device)
        merge = (self.merge*x).sum(dim=1, keepdim=True)
        downgrade: Tensor = relu(merge.expand(B, 64, H, W)+self.bias.expand(B, 64, H, W))
        while (min(downgrade.shape[2], downgrade.shape[3])>=3):
            downgrade = self.downgrade1(downgrade)
            sc:Tensor = self.score1(downgrade)
            sc = sc.view(B, 64, 10, sc.shape[-2], sc.shape[-1])
            even = sc[:, :, 0::2, :, :].max(dim=2).values
            odd = sc[:, :, 1::2, :, :].max(dim=2).values
            sc = cat([even, odd], dim=1)
            score = score + interpolate(self.score2(sc), size=(H, W), mode="nearest")
            downgrade = self.downgrade2(downgrade)
            # score = self.net(cat([score, t], dim=1))
            # score = score - score.min()
        # score = self.net(score)
        return score