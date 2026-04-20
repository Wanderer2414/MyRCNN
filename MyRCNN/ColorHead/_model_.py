from torch.nn import Module, Conv2d, ReLU, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter, Sequential
from torch import Tensor, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, conv2d, argmax,bincount, int64
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu, pad, unfold
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, Interpolate, Splitter, View, MinLeakyReLU
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
        if (self.training):
            median = patches.mode(dim=2).values
        else:
            # 2. Convert to integer bins [0 to 16]
            bins = (patches * 16.0).round().clamp(0.0, 16.0).to(int64)

            # 3. Build bin counts without expanding to a huge one-hot tensor
            N = bins.shape[-1]
            counts = zeros(B, C, 17, N, device=x.device, dtype=x.dtype)
            counts.scatter_add_(2, bins, ones(bins.shape, dtype=x.dtype, device=x.device))

            # 4. Extract mode values
            mode_bins = counts.argmax(dim=2)
            median = mode_bins.to(x.dtype) / 16.0

        # argmax(bincount(x))
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return median.view(B, C, H_out, W_out)
        
        
class ColorHead(Module):
    def __init__(self, batch: int, in_channels: int, half_out_channels: int) -> None:
        super().__init__()
        self.prepare = Sequential(
            ColorDownsample(16),
            ModePool2d(kernel_size=11, stride=1, padding=1),
            Conv2d(in_channels=in_channels, out_channels=half_out_channels*2, kernel_size=1),
            MaxLeakyReLU(threshold=0.1, scale=0.01),
            Conv2d(in_channels=half_out_channels*2, out_channels=half_out_channels*2, kernel_size=1, groups=half_out_channels), #  groups
            MinLeakyReLU(threshold=0.1, scale=0.01)
        )
        self.downgrade = Sequential(
            SharedConv(half_out_channels*2, kernel_size=3, stride=3, padding=1),
            LeakyReLU(inplace=True)
        )
        self.ft = Sequential(
            BatchNorm2d(half_out_channels*2, affine=True),
            SharedConv(channels=half_out_channels*2, kernel_size=1),
            EmphaseLocal(half_out_channels*2, kernel_size=11),
        )
        self.max = Sequential(
            BatchNorm2d(half_out_channels*2),
            MaxLeakyReLU(scale=0.7, threshold=0.1),
        )
        self.interpolate = Sequential(
            SharedConv(half_out_channels*2, kernel_size=5, padding=2),
            BatchNorm2d(half_out_channels*2),
            LeakyReLU(inplace=True)
        )
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)]).view(1, in_channels, 1, 1)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor):
        B, C, H, W = x.shape
        downgrade: Tensor = self.prepare(x)
        score: Tensor = zeros(B, self.half_out_channels*2, H, W, device=x.device)
        n = floor(log(min(downgrade.shape[2], downgrade.shape[3]), 3))
        for i in range(n):
            downgrade = self.downgrade(downgrade) # + avg_pool2d(downgrade, kernel_size=3, stride=3, padding=1)
            zoomout = interpolate(downgrade, size=(H, W), mode="bilinear")
            score = score + self.interpolate(zoomout)
        score = self.ft(score)
        score = self.ft(score)
        score = self.ft(score)
        # score = self.max(score)
        return score