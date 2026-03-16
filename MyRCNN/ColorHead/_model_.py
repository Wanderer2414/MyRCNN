from torch.nn import Module, Conv2d, ReLU, Sequential, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones
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

class ColorHead(Module):
    def __init__(self, in_channels: int, out_channels: int, device: device = device("cpu")) -> None:
        super().__init__()
        self.prepare = Sequential(
            Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1, device=device),
            BatchNorm2d(num_features=16, device=device),
            LeakyReLU()            
        )
        self.downgrade = Sequential(
            Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=3, padding=1, bias=False, device=device)
        )
        self.score = Sequential(
            Conv2d(in_channels=32, out_channels=16, kernel_size=1, groups=16, device=device),
            Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1, bias=False, device=device)
        )
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
        self.out_channels = out_channels
    def forward(self, x:Tensor) -> Tensor:
        x = mode_pool2d(x, kernel_size=5, stride=1, padding=2)
        B, C, H, W = x.shape
        downgrade: Tensor = self.prepare(x)
        score = zeros(B, self.out_channels, H, W, device=x.device)
        previous = downgrade
        while (min(downgrade.shape[2], downgrade.shape[3])>=3):
            downgrade = self.downgrade(downgrade) # + avg_pool2d(downgrade, kernel_size=3, stride=3, padding=1)
            zoomout = interpolate(downgrade, size=(H, W), mode="nearest")
            mix = stack([previous, zoomout], dim=2).reshape(B, 32, H, W)
            previous = zoomout
            score = score + self.score(mix)
        return score