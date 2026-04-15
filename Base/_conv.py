
from torch.nn import Module, Parameter
from torch import tensor, Tensor, device, float as tfloat, sigmoid, conv2d, where, amax

class SharedConv(Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.weight = Parameter(tensor(0.5, device=device).expand(in_channels, 1, kernel_size, kernel_size).clone())
        self.stride = stride
        self.padding = padding
        self.in_channels = in_channels
        self.bias = Parameter(tensor(0.0, dtype=tfloat, device=device).expand(in_channels)) if bias else None


    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.in_channels)

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.weight, bias=self.bias, stride=self.stride, padding=self.padding, groups=self.in_channels)
class EmphaseLocal(Module):
    def __init__(self, in_channels: int, kernel_size:int, device: device = device("cpu")) -> None:
        super().__init__()
        self.conv = SharedConv(in_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2, bias=True, device=device)
    def forward(self, x:Tensor) -> Tensor:
        score = sigmoid(self.conv(x))
        return x * score
class MaxLeakyReLU(Module):
    def __init__(self, threshold: float = 0.1, scale: float = 0.01):
        super().__init__()
        self.threshold = threshold
        self.scale = scale
    def forward(self, x:Tensor) -> Tensor:
        B, C, H, W = x.shape
        score = sigmoid(x)
        M = amax(score, dim=(-2, -1), keepdim=True) - self.threshold
        M = M.expand(B, C, H, W)
        alt = x*self.scale
        return where(score>=M, x, alt)
        

class MaxChannelReLU(Module):
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale
    def forward(self, x:Tensor) -> Tensor:
        return x.mean(dim=1, keepdim=True)*self.scale + (1-self.scale)*x.max(dim=1, keepdim=True).values
    