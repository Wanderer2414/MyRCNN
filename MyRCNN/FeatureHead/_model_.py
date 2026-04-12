from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d, Sigmoid
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum, split, cdist, int64, floor, sort, tensor_split
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import roi_align, nms
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
            
            # BatchNorm2d(half_color_channels*2, affine=False, device=device),
            # AvgPool2d(kernel_size=11,stride=1, padding=5),
            # MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            # BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            # AvgPool2d(kernel_size=11,stride=1, padding=5),
            # MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            # BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            # AvgPool2d(kernel_size=11,stride=1, padding=5),
            # MaxLeakyReLU(scale=0.1, threshold=0.01),
            
            BatchNorm2d(num_features=half_color_channels*2, affine=False, device=device),
            AvgPool2d(kernel_size=11,stride=1, padding=5),
            MaxLeakyReLU(scale=0.01, threshold=0),
            
            MaxChannelReLU()
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
       
        
def getnear(origin:Tensor, points: Tensor, threshold: float) -> Tensor:

    device = points.device

    # Step 1: assign grid cells
    cell_size = threshold
    grid = floor(points / cell_size).to(int64)

    # Step 2: hash grid coordinates → 1D key
    keys = grid[:, 0] * 1000000 + grid[:, 1]

    # Step 3: sort by grid cell
    sorted_keys, indices = sort(keys)
    points_sorted = points[indices]
    grid_sorted = grid[indices]

    # Step 4: find boundaries of same cell
    diff = sorted_keys[1:] != sorted_keys[:-1]
    split_idx = where(diff)[0] + 1

    groups = tensor_split(origin, split_idx.tolist())
    groups = cat([i.mean(dim=0,keepdim=True) for i in groups], dim=0)
    
    return groups
class FeatureHead(Module):
    def __init__(self, mask_channels: int, half_color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = BoundingBoxRegression(half_color_channels=half_color_channels, device=device)
        # self.cls = Classification(boundary_channels=mask_channels, color_channels=half_color_channels*2, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> Tensor:
        bbx: Tensor = self.bbx(color) # [B, SWH, H, W]
        return bbx
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