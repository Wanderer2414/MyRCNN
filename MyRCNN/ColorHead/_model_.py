from torch.nn import Module, Conv2d, ReLU, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter, Sequential
from torch import Tensor, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, conv2d
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu, pad, unfold
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, Interpolate, Splitter, Stack
from math import log, floor
class ColorDownsample(Module):
    def __init__(self, down: int):
        super().__init__()
        self.down = down
    def forward(self, x:Tensor) -> Tensor:
        x = (((x*255)/self.down).round()*self.down)/256
        return x
        
class ModePool2d(Module):
    def __init__(self, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
    def forward(self,x: Tensor) -> Tensor:
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode="constant")
        B, C, H, W = x.shape
        patches = unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        median = patches.mode(dim=2).values
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return median.view(B, C, H_out, W_out)
        
        
class ColorHead(Module):
    def __init__(self, batch: int, in_channels: int, half_out_channels: int) -> None:
        super().__init__()
        self.prepare = Sequential(
            ColorDownsample(16),
            ModePool2d(kernel_size=11, stride=1, padding=1),
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels),
            BatchNorm2d(num_features=half_out_channels),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels), #  groups
            BatchNorm2d(num_features=half_out_channels),
            LeakyReLU(inplace=True)
        )
        self.downgrade = Sequential(
            # Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=3, stride=3, padding=1, bias=False)
            SharedConv(half_out_channels, kernel_size=3, stride=3, padding=1)
        )
        self.ft = Sequential(
            BatchNorm2d(half_out_channels*2),
            MaxLeakyReLU(scale=0.7, threshold=0.1),
            BatchNorm2d(half_out_channels*2),
            EmphaseLocal(half_out_channels*2, kernel_size=11)
        )
        self.interpolate = Sequential(
            Splitter(
                Sequential(
                    SharedConv(half_out_channels, kernel_size=5, padding=2),
                    BatchNorm2d(half_out_channels),
                    LeakyReLU(inplace=True)
                ),
                Sequential(
                    SharedConv(half_out_channels, kernel_size=5, padding=2),
                    BatchNorm2d(half_out_channels),
                    LeakyReLU(inplace=True)
                )
            ),
            Stack(1)
        )
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)]).view(1, in_channels, 1, 1)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
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