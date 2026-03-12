from torch.nn import Module
from torch import Tensor
from torch.nn.functional import pad, unfold

class MedianPool2d(Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x) -> Tensor:
        # x: (B, C, H, W)

        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')

        B, C, H, W = x.shape

        patches = unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride
        )

        patches = patches.view(
            B,
            C,
            self.kernel_size * self.kernel_size,
            -1
        )

        median = patches.median(dim=2)[0]

        H_out = (H + 2*self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2*self.padding - self.kernel_size) // self.stride + 1

        return median.view(B, C, H_out, W_out)

def median_pool2d(x, kernel_size=3, stride=1, padding=0) -> Tensor:
    if padding > 0:
        x = pad(x, (padding, padding, padding, padding), mode="reflect")
    B, C, H, W = x.shape
    patches = unfold(x, kernel_size=kernel_size, stride=stride)
    patches = patches.view(B, C, kernel_size * kernel_size, -1)
    median = patches.mode(dim=2).values
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    return median.view(B, C, H_out, W_out)