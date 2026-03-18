from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import nms
def var_pool2d(x, kernel_size=3, stride=1, padding=0) -> Tensor:
    print(x.shape)
    if padding > 0:
        x = pad(x, (padding, padding, padding, padding), mode="reflect")
    B, C, H, W = x.shape
    patches = unfold(x, kernel_size=kernel_size, stride=stride)
    patches = patches.view(B, C, kernel_size * kernel_size, -1)
    M, I = patches.max(dim=2, keepdim=True)
    patches = patches[:, :, 0:1, :]*2 - M
    H_out = (H - kernel_size) // stride + 1
    W_out = (W - kernel_size) // stride + 1
    return patches.view(B, C, H_out, W_out)
class VarPool2d(Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, x: Tensor) -> Tensor:
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        B, C, H, W = x.shape
        patches = unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        M, I = patches.max(dim=2, keepdim=True)
        patches = patches[:, :, 0:1, :]*2 - M
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return patches.view(B, C, H_out, W_out)
    
class SumPool2d(Module):
    def __init__(self, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    def forward(self, x: Tensor) -> Tensor:
        if self.padding > 0:
            x = pad(x, (self.padding, self.padding, self.padding, self.padding), mode="reflect")
        B, C, H, W = x.shape
        patches = unfold(x, kernel_size=self.kernel_size, stride=self.stride)
        patches = patches.view(B, C, self.kernel_size * self.kernel_size, -1)
        patches = patches.sum(dim=2)
        H_out = (H - self.kernel_size) // self.stride + 1
        W_out = (W - self.kernel_size) // self.stride + 1
        return patches.view(B, C, H_out, W_out)
class BoundingBoxRegression(Module):
    def __init__(self, half_color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=5, stride=1, padding=2, groups=2*half_color_channels, bias=False, device=device)
        )
        self.score = Sequential(
            Conv2d(in_channels=half_color_channels*2, out_channels=1, kernel_size=1, bias=False, device=device),
            AvgPool2d(kernel_size=5, stride=1, padding=2)
        )
    def forward(self, x: Tensor):
        color = avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        wh: Tensor = self.bbx(color)
        w = wh[:, 0::2, :, :].max(dim=1, keepdim=True).values
        h = wh[:, 1::2, :, :].max(dim=1, keepdim=True).values
        wh = cat([w,h], dim=1)
        score: Tensor = self.score(x)
        score = avg_pool2d(score, kernel_size=3, stride=1, padding=1)
        return cat([score, wh], dim=1)
        
class Classification(Module):
    # 900 x 900 input
    def __init__(self, boundary_channels: int, color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.boundary_score = Sequential(
            Conv2d(in_channels=boundary_channels, out_channels=32, kernel_size=3, stride=3, device=device), # 300x300
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 300x300
            Conv2d(in_channels=8, out_channels=32, kernel_size=3, stride=3, device=device), # 100x100
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 100x100
            Conv2d(in_channels=8, out_channels=32, kernel_size=4, stride=4, device=device), # 25x25
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 25x25
            Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=5, device=device), # 5x5
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=1, device=device), # 5x5
        )
        self.color_score = Sequential(
            Conv2d(in_channels=color_channels, out_channels=color_channels*2, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=color_channels*2, out_channels=4, kernel_size=1, device=device),
            Conv2d(in_channels=4, out_channels=8, kernel_size=9, stride=9, device=device), # 900x900
            LeakyReLU(),
            Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 100x100
            Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=4, device=device), # 25x25
            LeakyReLU(),
            Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 25x25
            Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=5, device=device), # 5x5
            LeakyReLU(),
            Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 5x5
            
        )
        self.cls = Sequential(
            Conv2d(in_channels=20, out_channels=30, kernel_size=1, device=device), # 5x5
            LeakyReLU(),
            Conv2d(in_channels=30, out_channels=num_classes, kernel_size=5, device=device), # 5x5
        )
        
    def forward(self, mask: Tensor, color: Tensor):
        mask = self.boundary_score(mask)
        color = self.color_score(color)
        combine = cat([mask, color], dim=1)
        cls = self.cls(combine)
        return cls
        

class FeatureHead(Module):
    def __init__(self, mask_channels: int, half_color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = BoundingBoxRegression(half_color_channels=half_color_channels, device=device)
        # self.cls = Classification(boundary_channels=mask_channels, color_channels=self.cls_channels, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> list[Tensor]:
        bbx: Tensor = self.bbx(color) # [B, SWH, H, W]
        B, C, H, W = color.shape
        score = bbx[:, 0:1, :, :]
        y = arange(H, dtype=tfloat, device=mask.device).view(1, 1, H, 1).expand(B, 1, H, W).reshape(B, -1, 1)
        x = arange(W, dtype=tfloat, device=mask.device).view(1, 1, 1, W).expand(B, 1, H, W).reshape(B, -1, 1)
        bbx_flat = bbx.permute(0, 2, 3, 1).reshape(B, -1, 3)
        score_flat = sigmoid(bbx_flat[:,:,0:1])
        w = bbx_flat[:,:,0:1]
        h = bbx_flat[:,:,1:2]

        topk_idx = topk(score_flat,10,dim=1).indices

        mask = zeros_like(score_flat,dtype=tbool)
        mask = mask.scatter(1,topk_idx,True)
        mask = mask & (score_flat>0.6)
        mask = mask | (score_flat>0.9)

        x = x[mask]
        y = y[mask]
        w = w[mask].exp()*10
        h = h[mask].exp()*10
        s = score_flat[mask]
        x1 = (x-w).floor().long()
        x2 = (x+w).ceil().long()
        y1 = (y-h).floor().long()
        y2 = (y+h).ceil().long()
        out = stack([s,x1,y1,x2,y2],dim=-1).unsqueeze(0)
        
        # boxes= out[:, 1:] # [N, x1, y1, x2, y2]
        # confidence = out[:, 0]
        # nms_idx = nms(boxes, confidence, 0.2)
        # out = out[nms_idx].view(B, -1, 5)
        return [score, out]
        # B, C, H, W = bbx.shape
        # score = bbx[:, 0:1, :, :]
        # score_flat = sigmoid(score).reshape(B, -1, 1)
        # indices= topk(score_flat, 10, dim=1).indices
        # indices = zeros_like(score_flat, dtype=tbool, device=score.device).scatter(1, indices, True) | (score_flat > 0.8)
        # w  = bbx[:, 1:2, :, :].reshape(B, -1, 1)
        # h = bbx[:, 2:3, :, :].reshape(B, -1, 1)
        # indices = indices & (w>5) & (h>5)
        # w: Tensor = w[indices]
        # h: Tensor = h[indices]
        # boxes = stack([x1, y1, x2, y2], dim=-1)
        # color_boxes = zeros(size=(0, self.cls_channels, 900, 900), dtype=tfloat, device=mask.device)
        # boundary_boxes = zeros(size=(0, mask.shape[1], 900, 900), dtype=tfloat, device=mask.device)
        # for box in boxes:
        #     x1, y1, x2, y2 = box
        #     mask_box = interpolate(mask[:, :, y1:y2, x1:x2], size=(900, 900), mode="nearest")
        #     color_box = interpolate(color[:, self.bbx_channels:, y1:y2, x1:x2], size=(900, 900), mode="nearest")
        #     color_boxes = cat([color_boxes, color_box], dim=0)
        #     boundary_boxes = cat([boundary_boxes, mask_box], dim=0)
        # cls: Tensor = zeros(size=(B, H * W, 100), dtype=tfloat, device=mask.device)
        # indices = indices.repeat(1, 1, 100)
        # pred:Tensor = self.cls(boundary_boxes, color_boxes)
        # cls[indices] =pred.flatten()
        # cls = cls.reshape(B, self.num_classes, H, W)
        # score = cat([score, bbx[:, 1:3, :, :], cls], dim = 1)
        # return score