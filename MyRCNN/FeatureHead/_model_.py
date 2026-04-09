from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d, Sigmoid
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import roi_align
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, MaxChannelReLU

class WidthConv(Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.1]]]], device=device).repeat(1, 1, 1, kernel_size))
        self.stride = stride
        self.padding = padding
        if (bias):
            self.bias = Parameter(tensor([0], dtype=tfloat, device=device))
        else:
            self.bias = 0
        self.kernel_size = kernel_size
    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.expand(x.shape[1], 1, 1, self.kernel_size)
        return (conv2d(x, weight=kernel, stride=(1, self.stride), padding=(0, self.padding), groups=x.shape[1]) + self.bias)*x.shape[-1]
    

class HeightConv(Module):
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.kernel = Parameter(tensor([[[[0.1]]]], device=device).repeat(1, 1, kernel_size, 1))
        self.stride = stride
        self.padding = padding
        if (bias):
            self.bias = Parameter(tensor([0], dtype=tfloat, device=device))
        else:
            self.bias = 0
        self.kernel_size = kernel_size
    def forward(self, x: Tensor) -> Tensor:
        kernel = self.kernel.expand(x.shape[1], 1, self.kernel_size, 1)
        return (conv2d(x, weight=kernel, stride=(self.stride, 1), padding=(self.padding, 0), groups=x.shape[1]) + self.bias)*x.shape[-2]
class ChannelNormalize(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        M = x.detach().max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.expand(B, C, H, W)
        return x/M
class BoundingBoxRegression(Module):
    def __init__(self, half_color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, bias=False, device=device)
        )
        self.score = Sequential(
            ChannelNormalize(),
            
            BatchNorm2d(half_color_channels*2, affine=False, device=device),
            AvgPool2d(kernel_size=11,stride=1, padding=5),
            MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            AvgPool2d(kernel_size=11,stride=1, padding=5),
            MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            AvgPool2d(kernel_size=11,stride=1, padding=5),
            MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            AvgPool2d(kernel_size=11,stride=1, padding=5),
            MaxLeakyReLU(scale=0.01, threshold=0),
            
            MaxChannelReLU(),
            Sigmoid()
        )
        self.width = Sequential(
            WidthConv(kernel_size=11, stride=1, padding=5, device=device),
            SharedConv(kernel_size=1, stride=1, bias=False, device=device),
            MaxChannelReLU()
        )
        self.height = Sequential(
            HeightConv(kernel_size=11, stride=1, padding=5, device=device),
            SharedConv(kernel_size=1, stride=1, bias=False, device=device),
            MaxChannelReLU()
        )
    def forward(self, x: Tensor):
        wh: Tensor = self.bbx(x)
        B, C, H, W = x.shape
        w = self.width(wh[:, 0::2, :, :])
        h = self.height(wh[:, 1::2, :, :])
        # w = (sigmoid(w)-0.5)*W
        # h = (sigmoid(h)-0.5)*H
        wh = cat([w,h], dim=1)
        score: Tensor = self.score(x)
        return cat([score, wh], dim=1)
       
        

class FeatureHead(Module):
    def __init__(self, mask_channels: int, half_color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = BoundingBoxRegression(half_color_channels=half_color_channels, device=device)
        # self.cls = Classification(boundary_channels=mask_channels, color_channels=half_color_channels*2, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> list[Tensor]:
        bbx: Tensor = self.bbx(color) # [B, SWH, H, W]
        B, C, H, W = color.shape
        score = bbx[:,0:1,:,:]
        y = arange(H, dtype=tfloat, device=mask.device).view(1, 1, H, 1).expand(B, 1, H, W).reshape(B, -1, 1)
        x = arange(W, dtype=tfloat, device=mask.device).view(1, 1, 1, W).expand(B, 1, H, W).reshape(B, -1, 1)
        bbx_flat = bbx.permute(0, 2, 3, 1).reshape(B, -1, 3)
        score_flat = bbx_flat[:,:,0:1]
        w = bbx_flat[:,:,1:2]
        h = bbx_flat[:,:,2:3]
        mask = (score_flat>0.8) & (w>5) & (h>5)

        cx = x = x[mask]
        cy = y = y[mask]
        w = w[mask]
        h = h[mask]
        
        s = score_flat[mask]
        x1 = (x-w).floor().long()
        x2 = (x+w).ceil().long()
        y1 = (y-h).floor().long()
        y2 = (y+h).ceil().long()
        
        # indices = (x1 < 0) | (x2 > W)
        # n_width = (minimum(W - cx[indices], cx[indices])/w[indices]).detach()
        # w[indices] = w[indices]*n_width
        
        # x1[indices] = (cx[indices] - w[indices]).floor().long()
        # x2[indices] = (cx[indices] + w[indices]).ceil().long()
        
        # indices = (y1 < 0) | (y2 > H)
        # n_height = (minimum(H - cy[indices], cy[indices])/h[indices]).detach()
        # h[indices] = h[indices]*n_height
        # y1[indices] = (cy[indices] - h[indices]).floor().long()
        # y2[indices] = (cy[indices] + h[indices]).ceil().long()
        
        
        out = stack([s,x1,y1,x2,y2],dim=-1).unsqueeze(0)
        return [score, out]