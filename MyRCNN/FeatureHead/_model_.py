from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid

class BoundingBoxRegression(Module):
    def __init__(self, color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=color_channels, out_channels=color_channels*2, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=color_channels*2, out_channels=2, kernel_size=1, device=device)
        )
        self.score = Conv2d(in_channels=color_channels, out_channels=1, kernel_size=3, padding=1, stride=1, device=device)
        self.laplace = tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=tfloat, device=device).view(1, 1, 3, 3)
        self.ft =  Sequential(
            Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=6, out_channels=2, kernel_size=1, device=device)
        )
        self.color_channels = color_channels
    def forward(self, x: Tensor):
        color = conv2d(x, self.laplace.expand(self.color_channels, 1, 3, 3), stride=1, padding=1, groups=self.color_channels)
        wh = self.bbx(color)
        score = sigmoid(conv2d(self.score(x), self.laplace, stride=1, padding=1))
        wh = self.ft(cat([score, wh], dim=1))
        return cat([score, wh], dim=1)
        
class Classification(Module):
    # 900 x 900 input
    def __init__(self, mask_channels: int, color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.boundary_score = Sequential(
            Conv2d(in_channels=mask_channels, out_channels=32, kernel_size=3, stride=3, device=device), # 300x300
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
    def __init__(self, mask_channels: int, color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = BoundingBoxRegression(color_channels=color_channels, device=device)
        self.cls = Classification(mask_channels=mask_channels, color_channels=color_channels, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> Tensor:
        bbx: Tensor = self.bbx(color) # [B, SWH, H, W]
        B, C, H, W = bbx.shape
        score = bbx[:, 0:1, :, :]
        score_flat = score.reshape(B, -1, 1)
        indices= topk(score_flat, 10, dim=1).indices
        indices = zeros_like(score_flat, dtype=tbool, device=score.device).scatter(1, indices, True) | (score_flat > 0.8)
        B, C, H, W = color.shape
        w  = bbx[:, 1:2, :, :].reshape(B, -1, 1)
        h = bbx[:, 2:3, :, :].reshape(B, -1, 1)
        y = arange(H, dtype=tfloat, device=mask.device).view(1, 1, H, 1).expand(B, 1, H, W).reshape(B, -1, 1)
        x = arange(W, dtype=tfloat, device=mask.device).view(1, 1, 1, W).expand(B, 1, H, W).reshape(B, -1, 1)
        indices = indices & (w>5) & (h>5)
        w: Tensor = w[indices]
        h: Tensor = h[indices]
        x: Tensor = x[indices]
        y: Tensor = y[indices]
        x1 = (x-w).floor().long()
        x2 = (x+w).ceil().long()
        y1 = (y-w).floor().long()
        y2 = (y+w).ceil().long()
        boxes = stack([x1, y1, x2, y2], dim=-1)
        color_boxes = zeros(size=(0, color.shape[1], 900, 900), dtype=tfloat, device=mask.device)
        mask_boxes = zeros(size=(0, mask.shape[1], 900, 900), dtype=tfloat, device=mask.device)
        for box in boxes:
            x1, y1, x2, y2 = box
            mask_box = interpolate(mask[:, :, y1:y2, x1:x2], size=(900, 900), mode="nearest")
            color_box = interpolate(color[:, :, y1:y2, x1:x2], size=(900, 900), mode="nearest")
            color_boxes = cat([color_boxes, color_box], dim=0)
            mask_boxes = cat([mask_boxes, mask_box], dim=0)
        cls: Tensor = zeros(size=(B, H * W, 100), dtype=tfloat, device=mask.device)
        indices = indices.repeat(1, 1, 100)
        pred:Tensor = self.cls(mask_boxes, color_boxes)
        cls[indices] =pred.flatten()
        cls = cls.reshape(B, self.num_classes, H, W)
        score = cat([score, cls], dim = 1)
        return score