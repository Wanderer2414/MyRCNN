from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d, Sigmoid
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum, split, cdist, int64, floor, sort, tensor_split, amax
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import roi_align, nms
from Base import MaxLeakyReLU, SharedConv, EmphaseLocal, MaxChannelReLU
class WidthConv(Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.weight = Parameter(tensor(0.1, device=device).expand(in_channels, 1, 1, kernel_size).clone())
        self.stride = stride
        self.padding = padding
        self.groups = in_channels
        self.bias = Parameter(tensor(0.0, dtype=tfloat, device=device).expand(in_channels)) if bias else None
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        out = conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=(1, self.stride),
            padding=(0, self.padding),
            groups=self.groups
        )
        return out * x.shape[-1]


class HeightConv(Module):
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False, device: device = device("cpu")) -> None:
        super().__init__()
        self.weight = Parameter(tensor(0.1, device=device).expand(in_channels, 1, kernel_size, 1).clone())
        self.stride = stride
        self.padding = padding
        self.groups = in_channels
        self.bias = Parameter(tensor(0.0, dtype=tfloat, device=device).expand(in_channels)) if bias else None
        self.kernel_size = kernel_size

    def forward(self, x: Tensor) -> Tensor:
        out = conv2d(
            x,
            weight=self.weight,
            bias=self.bias,
            stride=(self.stride, 1),
            padding=(self.padding, 0),
            groups=self.groups
        )
        return out * x.shape[-2]
class ChannelNormalize(Module):
    def __init__(self):
        super().__init__()
    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        M = x.detach().max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values.expand(B, C, H, W)
        return x/M*x.detach().max()
class BoundingBoxRegression(Module):
    def __init__(self, half_color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, bias=False, device=device)
        )
        self.width = Sequential(
            WidthConv(in_channels=2*half_color_channels, kernel_size=11, padding=5, device=device),
            SharedConv(in_channels=2*half_color_channels, kernel_size=1, stride=1, bias=False, device=device),
        )
        self.height = Sequential(
            HeightConv(in_channels=2*half_color_channels, kernel_size=11, padding=5, device=device),
            SharedConv(in_channels=2*half_color_channels, kernel_size=1, stride=1, bias=False, device=device),
        )
        self.score = Sequential(
            # BatchNorm2d(num_features=half_color_channels*2, device=device, affine=False),
            Sigmoid()
        )
        self.max = MaxChannelReLU()
        self.channels = half_color_channels*2
    def forward(self, x: Tensor):
        wh: Tensor = self.bbx(x)
        B, C, H, W = x.shape
        w = self.width(wh)
        h = self.height(wh)
        # w = (sigmoid(w)-0.5)*W
        # h = (sigmoid(h)-0.5)*H
        sx = self.score(x)
        M = amax(sx, dim=(-2, -1), keepdim=True).expand(B, C, H, W) - 0.01
        score: Tensor = x*(sx>M)
        S = score.sum(dim=(-2,-1), keepdim=True)
        ws = (w*score/S).sum(dim=(-2,-1))
        hs = (h*score/S).sum(dim=(-2,-1))
        ps = amax(sx, dim=(-2, -1))
        col = arange(W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        row = arange(H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x1 = (col*score/S).sum(dim=(-2, -1)) - ws/2
        y1 = (row*score/S).sum(dim=(-2, -1)) - hs/2
        x2 = x1 + ws
        y2 = y1 + hs
        score = self.max(score)
        w = self.max(w)
        h = self.max(h)
        i = arange(B, device=x.device).view(B, 1).expand(B, self.channels)
        bbox = stack([i, x1, y1, x2 ,y2, ps], dim=-1).view(-1, 6)
        return cat([score, w, h], dim=1), bbox
       
    
class FeatureHead(Module):
    def __init__(self, mask_channels: int, half_color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = BoundingBoxRegression(half_color_channels=half_color_channels, device=device)
        # self.cls = Classification(boundary_channels=mask_channels, color_channels=half_color_channels*2, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> tuple[Tensor, Tensor]:
        score, bbx = self.bbx(color) # [B, SWH, H, W]
        return score, bbx
        # B, C, H, W = color.shape
        # score = bbx[:,0:1,:,:]
        # y = arange(H, dtype=tfloat, device=mask.device).view(1, 1, H, 1).expand(B, 1, H, W).reshape(B, -1, 1)
        # x = arange(W, dtype=tfloat, device=mask.device).view(1, 1, 1, W).expand(B, 1, H, W).reshape(B, -1, 1)
        # bbx_flat = bbx.permute(0, 2, 3, 1).reshape(B, -1, 3)
        # score_flat = bbx_flat[:,:,0:1]
        # w = bbx_flat[:,:,1:2]
        # h = bbx_flat[:,:,2:3]
        # mask = (score_flat>0.8) & (w>5) & (h>5)

        # cx = x = x[mask]
        # cy = y = y[mask]
        # w = w[mask]
        # h = h[mask]
        
        # s = score_flat[mask]
        # x1 = (x-w).floor()
        # x2 = (x+w).ceil()
        # y1 = (y-h).floor()
        # y2 = (y+h).ceil()
        
        # out = stack([x1,y1,x2,y2],dim=-1)
        # box = nms(out, s, 0.5)
        # out = cat([s.unsqueeze(-1), out], dim=-1)
        # distance = stack([cx, cy], dim=-1)
        # out = getnear(out, distance, 3)
        # N = out.shape[0]
        # current = 0
        # result = zeros(0, 5)
        # for i in range(1, N):
        #     distance = ((cx[i] - cx[current]).square() + (cy[i] - cy[current]).square()).sqrt()
        #     if (distance > 10):
        #         out[current] /= (i-current)
        #         result = cat([result, out[current].unsqueeze(0)], dim=0)
        #         current = i
        #     else: out[current] = out[current] + out[i]
        # out[current] /= N-current
        # result = cat([result, out[current].unsqueeze(0)], dim=0).unsqueeze(0)
        
        return [score, out.unsqueeze(0)]