from typing_extensions import Self

from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d, Sigmoid
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum, split, cdist, int64, floor, sort, tensor_split, amax, ones, amin
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, MaxChannelReLU
class WidthConv(Module):
    def __init__(self, channels:int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
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
            return (conv2d(x, weight=kernel, stride=(1, self.stride), padding=(0, self.padding), groups=self.channels) + self.bias)*x.shape[-1]
        return (conv2d(x, weight=self.weight, stride=(1, self.stride), padding=(0, self.padding), groups=self.channels) + self.bias)*x.shape[-1]
    

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
            return (conv2d(x, weight=kernel, stride=(self.stride, 1), padding=(self.padding, 0), groups=x.shape[1]) + self.bias)*x.shape[-2]
        return (conv2d(x, weight=self.weight, stride=(self.stride, 1), padding=(self.padding, 0), groups=x.shape[1]) + self.bias)*x.shape[-2]
class ChannelNormalize(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        M = x.detach().max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.expand(B, C, H, W)
        return x/M*x.detach().max()
class BoundingBoxRegression(Module):
    def __init__(self, batch:int, half_color_channels: int):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, bias=False)
        )
        self.width = Sequential(
            WidthConv(half_color_channels*2, kernel_size=11, stride=1, padding=5),
            LeakyReLU(inplace=True)
        )
        self.height = Sequential(
            HeightConv(half_color_channels*2, kernel_size=11, stride=1, padding=5),
            LeakyReLU(inplace=True)
        )
        self.score = Sequential(
            # BatchNorm2d(num_features=half_color_channels*2, affine=False),
            Sigmoid()
        )
        self.max = MaxChannelReLU()
        self.channels = half_color_channels*2
        self.batch = batch
    # def to(self, *args, **kwargs):
    #     super().to(*args, **kwargs)
    #     super().to(*args, **kwargs)
    #     device = kwargs.get("device", None)
    #     if device is None and len(args) > 0:
    #         device = args[0]
    #     if (self.training):
    #         self.batch_idx = self.batch_idx.to(device)
    #     return self
    def forward(self, x: Tensor):
        wh: Tensor = self.bbx(x)
        B, C, H, W = x.shape
        w: Tensor = self.width(wh)
        # w = w - amin(w, dim=(-2, -1) , keepdim=True)
        h = self.height(wh)
        # h = h - amin(h, dim=(-2, -1) , keepdim=True)
        
        sx = self.score(x)
        M = amax(sx, dim=(-2, -1), keepdim=True).expand(B, C, H, W) - 0.01
        score: Tensor = x*(sx>M)
        swh = cat([self.max(score), self.max(w), self.max(h)], dim=1)
        if (self.training):
            return swh
        else:
            score = score/score.sum(dim=(-2,-1), keepdim=True)
            ws = (w*score).sum(dim=(-2,-1))
            hs = (h*score).sum(dim=(-2,-1))
            ps = (sx*score).sum(dim=(-2,-1))
            i = arange(B,device=x.device).view(B, 1).expand(B, self.channels)
            col = (arange(W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)*score).sum(dim=(-2, -1))
            row = (arange(H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)*score).sum(dim=(-2, -1))
            x1 =  col - ws/2
            y1 =  row - hs/2
            x2 = col + ws/2
            y2 = row + hs/2
            bbox = stack([i, x1, y1, x2 ,y2, ps], dim=-1).view(-1, 6)
            return  swh, bbox
       
    
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