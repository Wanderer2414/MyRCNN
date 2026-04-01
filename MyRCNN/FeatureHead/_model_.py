from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import roi_align
class MaxChannelReLU(Module):
    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = scale
    def forward(self, x:Tensor) -> Tensor:
        return x.mean(dim=1, keepdim=True)*self.scale + (1-self.scale)*x.max(dim=1, keepdim=True).values
    
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
        self.device = device
        self.conv = SharedConv(kernel_size, padding=kernel_size//2, bias=True, device=device)
        
    def forward(self, x:Tensor) -> Tensor:
        score = sigmoid(self.conv(x))
        return x*score
class BoundingBoxRegression(Module):
    def __init__(self, half_color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, bias=False, device=device)
        )
        self.score = Sequential(
            BatchNorm2d(half_color_channels*2, device=device),
            MaxLeakyReLU(scale=0.01, threshold=0),
            Conv2d(in_channels=half_color_channels*2, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, device=device),
            MaxChannelReLU(),
            # BatchNorm2d(1, affine=False, device=device),
            # MaxLeakyReLU(threshold=0.1, scale=0.1),
            # EmphaseLocal(channels=1, kernel_size=11, device=device),
            # MaxLeakyReLU(threshold=0.1, scale=0.1)
            # BatchNorm2d(1, device=device)
            # Conv2d(in_channels=half_color_channels*2, out_channels=1, kernel_size=5, stride=1, padding=2, bias=False, device=device),
        )
        self.width = Conv2d(in_channels=half_color_channels, out_channels=half_color_channels, kernel_size=(1,11), stride=1, padding=(0,5),groups=half_color_channels,device=device)
        self.height = Conv2d(in_channels=half_color_channels, out_channels=half_color_channels, kernel_size=(11,1), stride=1, padding=(5, 0),groups=half_color_channels, device=device)
    def forward(self, x: Tensor):
        wh: Tensor = self.bbx(x)
        B, C, H, W = x.shape
        w = self.width(wh[:, 0::2, :, :]).max(dim=1, keepdim=True).values
        h = self.height(wh[:, 1::2, :, :]).max(dim=1, keepdim=True).values
        w = (sigmoid(w)-0.5)*W
        h = (sigmoid(h)-0.5)*H
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
        score_flat = sigmoid(bbx_flat[:,:,0:1])
        w = bbx_flat[:,:,1:2]
        h = bbx_flat[:,:,2:3]
        mask = (score_flat>0.8)

        cx = x = x[mask]
        cy = y = y[mask]
        w = w[mask]
        h = h[mask]
        
        s = score_flat[mask]
        x1 = (x-w).floor().long()
        x2 = (x+w).ceil().long()
        y1 = (y-h).floor().long()
        y2 = (y+h).ceil().long()
        
        indices = (x1 < 0) | (x2 > W)
        width = minimum(W - cx[indices], cx[indices])
        x1[indices] = (cx[indices] - width).floor().long()
        x2[indices] = (cx[indices] + width).ceil().long()
        
        indices = (y1 < 0) | (y2 > H)
        height = minimum(H - cy[indices], cy[indices])
        y1[indices] = (cy[indices] - height).floor().long()
        y2[indices] = (cy[indices] + height).ceil().long()
        
        
        out = stack([s,x1,y1,x2,y2],dim=-1).unsqueeze(0)
        return [score, out]