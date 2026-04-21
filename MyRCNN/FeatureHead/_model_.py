from typing import Any

from typing_extensions import Self

from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d, Sigmoid
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum, split, cdist, int64, floor, sort, tensor_split, amax, ones, amin
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, MaxChannelReLU, Splitter, Merger, View
class WidthConv(Module):
    def __init__(self, channels:int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.1]]]]).repeat(1, 1, 1, kernel_size))
        self.stride = stride
        self.padding = padding
        if (bias):
            self.bias = Parameter(tensor([0], dtype=tfloat))
        else:
            self.bias = 0
        self.channels = channels
        self.kernel_size = kernel_size
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if (not self.training):
            self.weight = self.kernel.expand(self.channels, 1, 1, self.kernel_size)
        return self
    def forward(self, x: Tensor) -> Tensor:
        if (self.training):
            kernel = self.kernel.expand(self.channels, 1, 1, self.kernel_size)
            result = (conv2d(x, weight=kernel, stride=(1, self.stride), padding=(0, self.padding), groups=self.channels) + self.bias)
        else:
            result = (conv2d(x, weight=self.weight, stride=(1, self.stride), padding=(0, self.padding), groups=self.channels) + self.bias)
        return sigmoid(result)*x.shape[-1]
    

class HeightConv(Module):
    def __init__(self, channels:int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.1]]]]).repeat(1, 1, kernel_size, 1))
        self.stride = stride
        self.padding = padding
        if (bias):
            self.bias = Parameter(tensor([0], dtype=tfloat))
        else:
            self.bias = 0
        self.channels = channels
        self.kernel_size = kernel_size
    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if (not mode):
            self.weight = self.kernel.expand(self.channels, 1, self.kernel_size, 1)
        return self
    def forward(self, x: Tensor) -> Tensor:
        if (self.training):
            kernel = self.kernel.expand(x.shape[1], 1, self.kernel_size, 1)
            result = (conv2d(x, weight=kernel, stride=(self.stride, 1), padding=(self.padding, 0), groups=x.shape[1]) + self.bias)
        else:
            result= (conv2d(x, weight=self.weight, stride=(self.stride, 1), padding=(self.padding, 0), groups=x.shape[1]) + self.bias)
        return sigmoid(result)*x.shape[-2]
class ChannelNormalize(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        M = x.detach().max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.expand(B, C, H, W)
        return x/M*x.detach().max()
class Distribute(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    def forward(self, x:Tensor) -> Tensor:
        score = x[:, :self.channels, :, :]
        result = cat([(x[:, self.channels*i:self.channels*(i+1), :, :] * score).sum(dim=(-2, -1), keepdim=True) for i in range(x.shape[1]//self.channels)], dim = 1)
        result = result[:, self.channels:, :, :]
        return result
class PercentWeight(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    def forward(self, x:Tensor) -> Tensor:
        return x/x.sum(dim=(-2, -1), keepdim=True)
class Bbx(Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
    def forward(self, x:Tensor) -> Tensor:
        wh = x[:, :self.channels, :, :]
        xy = x[:, self.channels:, :, :]
        return cat([xy - wh/2, xy + wh/2], dim=1)
class Filter(Module):
    def __init__(self, scale:float = 0.01):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        x = sigmoid(x)
        M = amax(x, dim=(-2, -1), keepdim=True) - 0.005
        return x*((x>M) + self.scale)
class BoundingBoxRegression(Module):
    def __init__(self, batch:int, half_color_channels: int):
        super().__init__()
        self.channels = half_color_channels*2
        self.bbx = Sequential(
            Splitter(
                Filter(),
                Sequential(
                    Splitter(
                        WidthConv(half_color_channels*2, kernel_size=11, stride=1, padding=5, bias=True),
                        HeightConv(half_color_channels*2, kernel_size=11, stride=1, padding=5, bias=True),
                    ),
                    SharedConv(channels=self.channels*2, kernel_size=1, stride=1, padding=0, bias=True)
                )
            ),
            Splitter(
                None,
                Merger((self.channels, self.channels, self.channels),
                       MaxChannelReLU(),
                       MaxChannelReLU(),
                       MaxChannelReLU()
                )
            )
        )
        self.distribute = Sequential(
            Merger((self.channels,), 
                    Sequential(
                        Splitter(
                            PercentWeight(self.channels),
                            None
                        ),   
                    )
                    
                    ),
            Distribute(self.channels),
            Merger((self.channels, self.channels*4),
                   None,
                   Bbx(self.channels*2)
                   )
        )
        self.batch = batch
    def forward(self, x: Tensor):
        swh: Tensor = self.bbx(x)
        B, C, H, W = x.shape
        
        if (self.training):
            return swh[:, self.channels*3:, :, :]
        else:
            idx = arange(B,device=x.device).view(B, 1, 1).expand(B, self.channels, 1)
            col = arange(W, device=x.device).view(1, 1, 1, W).expand(B, self.channels, H, W)
            row = arange(H, device=x.device).view(1, 1, H, 1).expand(B, self.channels, H, W)
            swhcr = cat([swh[:, :self.channels*3, :, :], col, row], dim=1)
            sx1y1x2y2 = self.distribute(swhcr)
            bbox = stack([sx1y1x2y2[:, self.channels*i: self.channels*(i+1), :, :] for i in range(5)], dim=1).permute(0,2,3,4,1).view(B, -1, 5)
            result = cat([idx, bbox[:, :, 1:], bbox[:, :, :1]], dim=-1).view(-1, 6)
            return swh[:, self.channels*3:, :, :], result
       
    
class FeatureHead(Module):
    def __init__(self, batch: int, mask_channels: int, half_color_channels: int, num_classes: int):
        super().__init__()
        self.bbx = BoundingBoxRegression(batch, half_color_channels=half_color_channels)
        # self.cls = Classification(boundary_channels=mask_channels, color_channels=half_color_channels*2, num_classes=num_classes)
        self.num_classes = num_classes
        
    def forward(self, color: Tensor) -> tuple[Tensor, Tensor]:
        
        if (self.training):
            score = self.bbx(color) # [B, SWH, H, W]
            return score
        else:
            score, bbx = self.bbx(color) # [B, SWH, H, W]
            return score, bbx