from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, conv2d
from torch.nn.functional import interpolate, avg_pool2d, max_pool2d, sigmoid, conv2d, relu, pad, unfold

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

class SharedConv(Module):
    def __init__(self, channels:int, kernel_size: int, stride: int = 1, padding: int = 0, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.5]]]], device=device).repeat(channels, 1, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, weight=self.kernel, stride=self.stride, padding=self.padding, groups=x.shape[1])
class Mix(Module):
    def __init__(self, device: device = device("cpu")) -> None:
        super().__init__()
        self.a = Parameter(tensor([0.5], device=device))
        self.b = Parameter(tensor([0.5], device=device))
    def forward(self, a: Tensor, b:Tensor) -> Tensor:
        return a*self.a + b*self.b
        
class ColorHead(Module):
    def __init__(self, in_channels: int, half_out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels*2, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=in_channels*2, out_channels=half_out_channels, kernel_size=1, device=device),
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels, device=device), #  groups
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU()
        )
        self.downgrade = Sequential(
            # Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=3, stride=3, padding=1, bias=False, device=device)
            SharedConv(channels=half_out_channels, kernel_size=3, stride=3, padding=1, device=device)
        )
        self.width = Sequential(
            SharedConv(channels=half_out_channels, kernel_size=1, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU()
        )
        self.height = Sequential(
            SharedConv(channels=half_out_channels, kernel_size=1, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU()
        )
        self.mix = Mix(device=device)
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor) -> Tensor:
        x = (((x*255)/16).round()*16)/256
        x = mode_pool2d(x, kernel_size=11, stride=1, padding=5)
        B, C, H, W = x.shape
        downgrade: Tensor = self.prepare(x)
        score = zeros(B, self.half_out_channels*2, H, W, device=x.device)
        # previous = downgrade
        while (min(downgrade.shape[2], downgrade.shape[3])>=3):
            downgrade = self.downgrade(downgrade) # + avg_pool2d(downgrade, kernel_size=3, stride=3, padding=1)
            zoomout = interpolate(downgrade, size=(H, W), mode="nearest")
            width = self.width(zoomout)
            height = self.height(zoomout)
            mix = stack([width, height], dim=1).reshape(B, self.half_out_channels*2, H, W)
            score = score + mix
        return score