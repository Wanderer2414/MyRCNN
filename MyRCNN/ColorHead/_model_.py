import torch
from torch.nn import Module, Conv2d, ReLU, AvgPool2d, MaxPool2d, Sigmoid, Linear, LeakyReLU, BatchNorm2d, Parameter, Sequential
from torch import Tensor, device, cat, where, stack, arange, float as tfloat, zeros, tensor, ones, conv2d
from torch.nn.functional import interpolate, pad, unfold, avg_pool2d, max_pool2d, sigmoid, conv2d, relu
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, Interpolate, Parallel, Stack, Expand, Splitter, SumZip, Select, Loop, Mul
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

        # 2. Convert to integer bins [0 to 16]
        bins = (patches * 16.0).round().clamp(0.0, 16.0).to(torch.int64)

        # 3. Build bin counts without expanding to a huge one-hot tensor
        N = bins.shape[-1]
        counts = zeros(B, C, 17, N, device=x.device, dtype=x.dtype)
        counts.scatter_add_(2, bins, ones(bins.shape, dtype=x.dtype, device=x.device))

        # 4. Extract mode values
        mode_bins = counts.argmax(dim=2)
        mode_vals = mode_bins.to(x.dtype) / 16.0

        # 5. Reshape to output
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return mode_vals.view(B, C, H_out, W_out)
        
class CommonBlock(Module):
    def __init__(self, half_out_channels: int, device: device):
        super().__init__()
        self.module = Sequential(
            SharedConv(in_channels=half_out_channels,kernel_size=5, padding=2, device=device),
            BatchNorm2d(half_out_channels, device=device),
            LeakyReLU(inplace=True)
        )
    def forward(self, x:Tensor) -> Tensor:
        return self.module(x)
class ColorHead(Module):
    def __init__(self, in_channels: int, half_out_channels: int, size: int, device: device = device("cpu")) -> None:
        super().__init__()
        n = floor(log(size, 3))
        self.downgrade = Sequential(
            ColorDownsample(16),
            ModePool2d(kernel_size=11, stride=1, padding=1),
            Conv2d(in_channels=in_channels, out_channels=half_out_channels, kernel_size=1, device=device),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels, device=device),
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, device=device),
            Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=1, groups=half_out_channels, device=device), #  groups
            BatchNorm2d(num_features=half_out_channels, device=device),
            LeakyReLU(inplace=True),
            # Conv2d(in_channels=half_out_channels, out_channels=half_out_channels, kernel_size=3, stride=3, padding=1, bias=False, device=device)
            Expand(in_channels=half_out_channels, out_channels=half_out_channels*2, mode="zero"),
            Loop(
                n,
                Sequential(
                    Select(half_out_channels*2, ((half_out_channels, half_out_channels*2),),
                           Sequential(
                                SharedConv(in_channels=half_out_channels, kernel_size=3, stride=3, padding=1, device=device),
                                Parallel(
                                    Sequential(
                                        Interpolate(size=(size, size), mode="bilinear"),
                                        Parallel(
                                            CommonBlock(half_out_channels, device),
                                            CommonBlock(half_out_channels, device),
                                        ),
                                    ),
                                    Module(),
                                )
                            )),
                    Splitter((in_channels*2, in_channels), 
                                SumZip(split_size=in_channels),
                                Module()
                           )
                )
            ),
            Parallel(
                Module(),
                Sequential(
                    Loop(
                        3,
                        Sequential(
                            BatchNorm2d(half_out_channels*2, device=device),
                            MaxLeakyReLU(scale=0.7, threshold=0.1),
                            BatchNorm2d(half_out_channels*2, device=device),
                            EmphaseLocal(in_channels=half_out_channels*2, kernel_size=11, device=device),
                        )
                    ),
                    Mul(1/n)
                )
            )
        )
        # self.weight = tensor([pow(256, in_channels-i) for i in range(in_channels)], device=device).view(1, in_channels, 1, 1)
        self.half_out_channels = half_out_channels
    def forward(self, x:Tensor) -> tuple[Tensor, Tensor]:
        result = self.downgrade(x)
        mask = result[:, :self.half_out_channels*2, :, :]
        score = result[:, self.half_out_channels*2:, :, :]
        return mask, score
        