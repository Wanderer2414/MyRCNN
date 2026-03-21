from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU, AvgPool2d, Parameter, BatchNorm2d
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros, tensor, conv2d, topk, float as tfloat, arange, stack, bool as tbool, meshgrid, minimum, maximum
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate, sigmoid, pad, unfold, relu
from torchvision.ops import roi_align
class BoundingBoxRegression(Module):
    def __init__(self, half_color_channels: int, device: device = device("cpu")):
        super().__init__()
        self.bbx = Sequential(
            Conv2d(in_channels=2*half_color_channels, out_channels=half_color_channels*2, kernel_size=1, groups=2*half_color_channels, bias=False, device=device)
        )
        self.score = Sequential(
            Conv2d(in_channels=half_color_channels*2, out_channels=1, kernel_size=1, bias=False, device=device),
            
        )
        self.width = Conv2d(in_channels=half_color_channels, out_channels=half_color_channels, kernel_size=(1,5), stride=1, padding=(0,2), bias=False, groups=half_color_channels,device=device)
        self.height = Conv2d(in_channels=half_color_channels, out_channels=half_color_channels, kernel_size=(5,1), stride=1, padding=(2, 0), bias=False, groups=half_color_channels, device=device)
    def forward(self, x: Tensor):
        wh: Tensor = self.bbx(x)
        w = self.width(wh[:, 0::2, :, :]).max(dim=1, keepdim=True).values
        h = self.height(wh[:, 1::2, :, :]).max(dim=1, keepdim=True).values
        wh = cat([w,h], dim=1)
        score: Tensor = self.score(x)
        return cat([score, wh], dim=1)
       
def nms(boxes: Tensor, iou_threshold)->Tensor:
    N = boxes.shape[0]
    rows, cols = meshgrid(arange(N, device=boxes.device), arange(N, device=boxes.device), indexing='ij')
    
    boxes1 = boxes.unsqueeze(1).expand(N, N, 4)
    boxes2 = boxes.unsqueeze(0).expand(N, N, 4)
    x1 = max(boxes1[:, :, 0], boxes2[:, :, 0])
    x2 = min(boxes1[:, :, 2], boxes2[:, :, 2])
    y1 = max(boxes1[:, :, 1], boxes2[:, :, 1])
    y2 = min(boxes1[:, :, 3], boxes2[:, :, 3])
    s = ((boxes[:, 2]-boxes[:, 0])*(boxes[:, 3] - boxes[:, 1]))
    s1 = s.unsqueeze(1).expand(N, N)
    s2 = s.unsqueeze(0).expand(N, N)
    intersect = ((x2-x1)*(y2-y1))
    IoU = intersect/(s1+s2 - intersect)
    IoU = IoU * (rows>cols)
    cond = IoU > iou_threshold
    indices = (cond.any(dim=1).logical_not())
    
    return boxes[indices] 
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
        self.cls = Classification(boundary_channels=mask_channels, color_channels=half_color_channels*2, num_classes=num_classes, device=device)
        self.num_classes = num_classes
        
    def forward(self, mask: Tensor, color: Tensor) -> list[Tensor]:
        bbx: Tensor = self.bbx(color) # [B, SWH, H, W]
        B, C, H, W = color.shape
        score = bbx[:,0:1,:,:]
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
        mask = mask | (score_flat>0.8)

        cx = x = x[mask]
        cy = y = y[mask]
        w = w[mask].exp()*10
        h = h[mask].exp()*10
        
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
        # boxes= out[:, 1:] # [N, x1, y1, x2, y2]
        # boxes = nms(boxes, 0.6)
        # N = boxes.shape[0]
        # boxes = cat([zeros(N, 1), boxes], dim=-1)
        # output_size = (400, 400)
        # mask_crop: Tensor = roi_align(mask, boxes, output_size)
        # color_crop: Tensor = roi_align(color, boxes, output_size)
        # cls = self.cls(mask_crop, color_crop)
        # out = cat([out, cls], dim=-1)
        return [score, out]