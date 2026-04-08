
from torch.nn import Module, Parameter
from torch import tensor, Tensor, device, float as tfloat, sigmoid, conv2d, where
class SharedConv(Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.5]]]], device=device).repeat(1, 1, kernel_size, kernel_size))
        self.stride = stride
        self.padding = padding
        if (bias):
            self.bias = Parameter(tensor([0], dtype=tfloat, device=device))
        else:
            self.bias = 0
        self.kernel_size = kernel_size
    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.expand(x.shape[1], 1, self.kernel_size, self.kernel_size)
        return conv2d(x, weight=kernel, stride=self.stride, padding=self.padding, groups=x.shape[1]) + self.bias
class EmphaseLocal(Module):
    def __init__(self, kernel_size:int, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.device = device
        self.conv = SharedConv(kernel_size, padding=kernel_size//2, bias=True, device=device)
        
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
        M = score.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values - self.threshold
        M = M.expand(B, C, H, W)
        return where(score>=M, x, self.scale*x)
        
