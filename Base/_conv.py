
from typing_extensions import Self

from torch.nn import Module, Parameter
from torch import tensor, Tensor, device, float as tfloat, sigmoid, conv2d, where, amax
class SharedConv(Module):
    def __init__(self, batch:int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.5]]]], device=device).repeat(1, 1, kernel_size, kernel_size))
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.batch = batch
        if (self.bias):
            self.bias = Parameter(tensor([0], dtype=tfloat, device=device))
        else:
            self.bias = 0
    def train(self, mode: bool = True):
        super().train(mode)
        if (not mode):
            self.kernel = self.kernel.expand(self.batch, 1, self.kernel_size, self.kernel_size)
        return self
    def forward(self, x: Tensor) -> Tensor:
        if (self.training):
            kernel = self.kernel.expand(x.shape[1], 1, self.kernel_size, self.kernel_size)
        else:
            kernel = self.kernel
        return conv2d(x, weight=kernel, stride=self.stride, padding=self.padding, groups=x.shape[1]) + self.bias
class EmphaseLocal(Module):
    def __init__(self, batch:int, kernel_size:int, device: device = device("cpu")) -> None:
        super().__init__()
        self.conv = SharedConv(batch, kernel_size, padding=kernel_size//2, bias=True, device=device)
    def forward(self, x:Tensor) -> Tensor:
        score = sigmoid(self.conv(x))
        return x*score
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
    