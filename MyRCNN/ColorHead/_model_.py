from torch.nn import Module, Conv2d, ReLU, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter, Sequential
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, conv2d
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu, pad, unfold
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, Interpolate, Splitter, Stack
from math import log, floor
def mode_pool2d(x, kernel_size=3, stride=1, padding=0) -> Tensor:
    if padding > 0:
        x = pad(x, (padding, padding, padding, padding), mode="constant")
    B, C, H, W = x.shape
    patches = unfold(x, kernel_size=kernel_size, stride=stride)
    patches = patches.view(B, C, kernel_size * kernel_size, -1)
    median = patches.mode(dim=2).values
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    return median.view(B, C, H_out, W_out)
        
class ColorHead(Module):
    def __init__(self, in_channels: int, half_out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1, device=device),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels, device=device),
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, device=device),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels, device=device), #  groups
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True)
        )
        self.downgrade = Sequential(
            # Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=3, stride=3, padding=1, bias=False, device=device)
            SharedConv(kernel_size=3, stride=3, padding=1, device=device)
        )
        self.ft = Sequential(
            BatchNorm2d(half_out_channels*2, device=device),
            MaxLeakyReLU(scale=0.7, threshold=0.1),
            BatchNorm2d(half_out_channels*2, device=device),
            EmphaseLocal(kernel_size=11, device=device)
        )
        self.interpolate = Sequential(
            Splitter(
                Sequential(
                    SharedConv(kernel_size=5, padding=2, device=device),
                    BatchNorm2d(half_out_channels, device=device),
                    LeakyReLU(inplace=True)
                ),
                Sequential(
                    SharedConv(kernel_size=5, padding=2, device=device),
                    BatchNorm2d(half_out_channels, device=device),
                    LeakyReLU(inplace=True)
                )
            ),
            Stack(1)
        )
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor) -> Tensor:
        x = (((x*255)/16).round()*16)/256
        x = mode_pool2d(x, kernel_size=11, stride=1, padding=5)
        B, C, H, W = x.shape
        downgrade: Tensor = self.prepare(x)
        score = zeros(B, self.half_out_channels*2, H, W, device=x.device)
        # previous = downgrade
        n = floor(log(min(downgrade.shape[2], downgrade.shape[3]), 3))
        for i in range(n):
            downgrade = self.downgrade(downgrade) # + avg_pool2d(downgrade, kernel_size=3, stride=3, padding=1)
            zoomout = interpolate(downgrade, size=(H, W), mode="bilinear")
            score = score + self.interpolate(zoomout)
        score = self.ft(score)
        score = self.ft(score)
        score = self.ft(score)
        return score/n