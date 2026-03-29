from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, cuda
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu, pad, unfold
import gc
from math import log, floor, sqrt
def mode_pool2d(x, kernel_size=3, stride=1, padding=0) -> Tensor:
    if padding > 0:
        x = pad(x, (padding, padding, padding, padding), mode="reflect")
    B, C, H, W = x.shape
    patches = unfold(x, kernel_size=kernel_size, stride=stride)
    patches = patches.view(B, C, kernel_size * kernel_size, -1)
    median = patches.mode(dim=2).values
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    return median.view(B, C, H_out, W_out)

class SharedConv2d(Module):
    def __init__(self, channels: int, kernel_size: int=1, stride:int = 1, bias=False, dilation: int = 1, device: device = device("cpu")):
        super().__init__()
        self.kernel_size = kernel_size

        # Single shared kernel
        self.weight = Parameter(
            tensor([0.5], device=device).repeat(kernel_size*kernel_size*channels).reshape(channels, 1, kernel_size, kernel_size)
        )
        self.dilation = dilation
        if bias:
            self.bias = Parameter(zeros(1, device=device))
        else:
            self.bias = None
        self.stride = stride
    def forward(self, x):
        N, C, H, W = x.shape

        return conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.kernel_size // 2,
            groups=C
        )
    
class MaxReLU(Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0, slope: float = 0.01) -> None:
        super().__init__()
        self.slope = slope
        self.padding = padding
        self.stride = stride
        self.kernel_size = kernel_size
        self.mid = (self.kernel_size+1) * (self.kernel_size//2)
    def forward(self, x: Tensor) -> Tensor:
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        B, C, H, W = x.shape
        patches = unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        M = patches.max(dim=-2, keepdim=True).values
        error = patches.std(dim=-2, keepdim=True, unbiased=False)/10
        mid = patches[:, :, self.mid:self.mid+1, :]
        output = where(mid >= M-error, M, self.slope*mid)
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return output.reshape(B, C, H_out, W_out)
def maxReLU(x: Tensor, kernel_size:int, stride: int = 1, padding:int = 0, slope: float= 0.01) -> Tensor:
    print("Here")
    if padding > 0:
        x = pad(x, (padding, padding, padding, padding), mode="reflect")
    mid = (kernel_size+1)*(kernel_size//2)
    B, C, H, W = x.shape
    patches = unfold(x, kernel_size=kernel_size, stride=stride)
    patches = patches.view(B, C, kernel_size * kernel_size, -1)
    M = patches.max(dim=-2, keepdim=True).values
    error = patches.std(dim=-2, keepdim=True, unbiased=False)/10
    mid = patches[:, :, mid:mid+1, :]
    output = where(mid >= M-error, M, slope*mid)
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    print("Out")
    return output.reshape(B, C, H_out, W_out)
class Mix(Module):
    def __init__(self, device: device = device("cpu")):
        super().__init__()
        self.a = Parameter(tensor([0.5], device=device))
        self.b = Parameter(tensor([0.5], device=device))
    def forward(self, a: Tensor, b: Tensor):
        return a*self.a + b*self.b
class ColorHead(Module):
    def __init__(self, in_channels: int, half_out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1, device=device),
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, device=device), #  groups
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True)
        )
        self.downgrade = Sequential(
            SharedConv2d(kernel_size=3, channels=half_out_channels, bias=False, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU(inplace=True)
        )
        self.width = Sequential(
            SharedConv2d(kernel_size=1, channels=half_out_channels, bias=False, device=device),
            # Conv2d(in_channels=half_out_channels, out_channels=2*half_out_channels, kernel_size=1, groups=half_out_channels, bias=False, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU(inplace=True)
            # Conv2d(in_channels=2*half_out_channels, out_channels=2*half_out_channels, kernel_size=1, groups=2*half_out_channels, device=device),
        ) # 2, 2, H, W
        self.height = Sequential(
            SharedConv2d(kernel_size=1, channels=half_out_channels, bias=False, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU(inplace=True)
        )
        self.residual = Conv2d(in_channels=half_out_channels, out_channels=2*half_out_channels, kernel_size=1, groups=half_out_channels, bias=False, device=device)
        self.mix = Mix(device=device)
        self.norm = BatchNorm2d(half_out_channels*2, device=device)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor) -> Tensor:
        x = (((x*255)/16).round()*16)/256
        x = mode_pool2d(x, kernel_size=11, stride=1, padding=5)
        B, C, H, W = x.shape
        prepare = self.prepare(x)
        downgrade: Tensor = prepare
        intermedian = stack([prepare, prepare]).reshape(B, 32, H, W)
        n = floor(log(min(downgrade.shape[-2], downgrade.shape[-1]), 3))
        score = zeros(B, self.half_out_channels*2, H, W, device=x.device)
        # previous = downgrade
        for i in range(n):
            downgrade = self.downgrade(downgrade) # + avg_pool2d(downgrade, kernel_size=3, stride=3, padding=1)
            # zoomout = interpolate(downgrade, size=(H, W), mode="nearest")
            # mix = stack([previous, zoomout], dim=2).reshape(B, 32, H, W)
            # previous = zoomout
            # sc = self.score(zoomout)
            width = self.width(downgrade)
            height = self.height(downgrade)
            sc = stack([width, height], dim=1).reshape(B, 32, H, W)
            intermedian = sc + intermedian
            # kernel = 3*pow(3,i)
            # score = score + maxReLU(intermedian, kernel_size=kernel, stride=1, padding=kernel//2)
        # score = self.norm(score)
            # score = self.mix(score, sc)
        # score = score + self.residual(prepare)
        return intermedian