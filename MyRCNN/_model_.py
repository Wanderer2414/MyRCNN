from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like, load, stack, sigmoid, floor, int64, where, sort, tensor_split, ones, no_grad, softmax
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import complete_box_iou_loss, roi_align
from torch.optim import Adam
from typing import Callable
from time import time
from torch.utils.data import DataLoader
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
from utils import non_max_suppression, mean_average_precision
import os

dev = "cpu"
row = arange(400, dtype=tfloat, device=dev).view(1,1,400,1).expand(1,1,400,400)
col = arange(400, dtype=tfloat, device=dev).view(1,1,1,400).expand(1,1,400,400)
center_x = center_y = 200
distance = ((col-center_x).square() + (row-center_y).square()).sqrt()
target = distance.min(dim=1, keepdim=True).values
target = 1-target/target.max()
    
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
class MyRCNN(Module):
    def __init__(self, channels: int, device: device = device("cpu"))->None:
        super().__init__()
        self.mask = MaskHead.MaskHead(device=device)
        self.color = ColorHead.ColorHead(in_channels=3, half_out_channels=16, device=device)
        self.feat = FeatureHead.FeatureHead(half_color_channels=16, mask_channels=1, num_classes=100, device=device)
    def forward(self, x:Tensor) -> tuple[Tensor, Tensor,Tensor, Tensor, Tensor]:
        boundary: Tensor = self.mask(x)
        mask, color = self.color(x)
        score, bbx = self.feat(boundary, color)
        return boundary, mask, color, score, bbx
def MyBBLoss(scores: Tensor, labels: Tensor) -> Tensor:
    C_gt = 0
    score_box = roi_align(scores[:, 0:1, :, :], labels[:, 0:5], (400, 400))  # type: ignore[assignment]
    boxes = labels[:, 1:5]
    wh = roi_align(scores[:, 1:3, :, :], labels[:, 0:5], (400, 400)) # type: ignore[assignment]
    C_gt = score_box.shape[0]
    
    score =  binary_cross_entropy_with_logits(score_box, target.expand(C_gt, 1, 400, 400))

    
    FIoULoss = FIoU(score_box, wh)
    return score + FIoULoss
    return score
def FIoU(score: Tensor, wh: Tensor, eps:float = 1e-7) -> Tensor:
    """Summary

    Args:
        boxes (Tensor): [N, 4] -> [x1,y1,x2,y2]
        boxes_gt (Tensor): [M, 4] -> [x1, y1, x2, y2]

    Returns:
        Tensor: _description_
    """
    H, W = score.shape[-2:]
    w = wh[:, 0:1, :, :]
    h = wh[:, 1:2, :, :]
    
    indices = (sigmoid(score) > 0.8) * score
    N = score.shape[0]
    count = indices.view(N, -1).sum(dim=-1, keepdim=True).view(N, 1, 1, 1)
    indices = indices/(count+eps)
    x = arange(W, dtype=tfloat, device=score.device).view(1, 1, 1, W).expand(1, 1, H, W) / W
    y = arange(H, dtype=tfloat, device=score.device).view(1, 1, H, 1).expand(1, 1, H, W) / H
    
    pred_w = (w*indices).sum(dim=(-2, -1))
    pred_h = (h*indices).sum(dim=(-2, -1))
    pred_x = (x*indices).sum(dim=(-2, -1))
    pred_y = (y*indices).sum(dim=(-2, -1))
    pred = cat([pred_x, pred_y, pred_w, pred_h], dim=1)
    N = pred.shape[0]
    target= ones(N, 4, device=score.device) * 0.5
    loss = binary_cross_entropy_with_logits(pred, target)
    return loss

def Overlapse(boxes: Tensor, boxes_gt: Tensor) -> Tensor:
    """Summary

    Args:
        boxes (Tensor): [N, 4] -> [x1,y1,x2,y2]
        boxes_gt (Tensor): [N, 4] -> [x1, y1, x2, y2]

    Returns:
        Tensor: _description_
    """
    x1 = max(boxes[:, :, :, 0:1], boxes_gt[:, :, :, 0:1])
    x2 = min(boxes[:, :, :, 2:3], boxes_gt[:, :, :, 2:3])
    y1 = max(boxes[:, :, :, 1:2], boxes_gt[:, :, :, 1:2])
    y2 = min(boxes[:, :, :, 3:4], boxes_gt[:, :, :, 3:4])
    return ((x2 - x1)*(y2-y1)).abs()
    
def ClsLoss(cls: Tensor, label: Tensor) -> Tensor:
    """_summary_
    Args:
        boxes (Tensor): [N, num_classes] []
        label (Tensor): [1, N, 5] [x1, y1, x2, y2, cls]

    Returns:
        Tensor: 
    """
    N = cls.shape[0]
    cls_label = label[:, -1].long()
    loss = cross_entropy(cls, cls_label, reduction="mean")
    return loss
batch_size = 4
class Model:
    def __init__(self, train_data:DataLoader, test_data: DataLoader, num_classes, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.cls = Classfication.Classification(mask_channels=32, num_classes=num_classes, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        self.num_classes = num_classes
    def inference(self, x:Tensor) -> Tensor:
        x = x.to(device=self.device)
        boundary, mask, color, score, bbx = self.model(x)
        cls:Tensor = self.cls(boundary, score[:, 0:1, :, :], x, bbx[:, :-1])
        cls_score:Tensor = softmax(cls, dim=-1)
        cls_score = cls * bbx[:, -1:]
        cls_score = cls_score.unsqueeze(-1)
        N = cls.shape[0]
        cls_range = arange(self.num_classes, device=self.device).view(1, self.num_classes, 1).expand(N, self.num_classes, 1)
        batch = bbx[:, 0:1].view(N, 1, 1).expand(N, self.num_classes, 1)
        bbx = bbx[:, 1:5].view(N, 1, 4).expand(N, self.num_classes, 4)
        result = cat([batch, cls_range, cls_score, bbx], dim=-1).view(N*self.num_classes, 7)
        # ls = non_max_suppression(result.tolist(), 0.8, 0.3)
        # return tensor(ls, device=self.device)
        return result
        
    def train(self):
        start = time()
        if (os.path.exists("bbx.pth")):
            self.model.load_state_dict(load("bbx.pth", map_location=self.device))
            print("Load model!")
        else:
            epoches = 2
            data_size = len(self.train_data)
            for epoch in range(epoches):
                for i, (tens, labels) in enumerate(self.train_data):
                    boundary, mask, color, score, bbx = self.model(tens.to(self.device))
                    lss = MyBBLoss(score, labels.to(self.device))
                    self.opt.zero_grad()
                    lss.backward()
                    self.opt.step()
                    # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                    # if ((i+1) % (size//5) == 0):
                    #     print(f"Saved: {(i+1)} / {size//5} progress")
                    #     save(self.model.state_dict(), "bbx.pth")
                    show_progress_counter(i+1, data_size, start, f"Epoch {epoch}/{epoches}; Loss {lss}", epoch, epoches)
                    if (i%100 == 0):
                        save(self.model.state_dict(), "bbx.pth")
                save(self.model.state_dict(), "bbx.pth")
        if (os.path.exists("cls.pth")):
            self.cls.load_state_dict(load("cls.pth", map_location=self.device))
            print("Load model!")
        else:
            start = time()
            epoches = 2
            data_size = len(self.train_data)
            for epoch in range(epoches):
                for i, (tens, labels) in enumerate(self.train_data):
                    tens = tens.to(self.device)
                    labels = labels.to(self.device)
                    boundary, mask, color, score, bbx = self.model(tens)
                    boxes = labels[:, 0:5]
                    cls = self.cls(boundary, score[:, 0:1, :, :], tens, boxes)
                    lss = ClsLoss(cls, labels)
                    self.opt2.zero_grad()
                    lss.backward()
                    self.opt2.step()
                    show_progress_counter(i+1, data_size, start, f"Epoch {epoch}/{epoches}; Loss {lss}", epoch, epoches)
                    if (i%100 == 0):
                        save(self.cls.state_dict(), "cls.pth")
                save(self.cls.state_dict(), "cls.pth")
    def Evaluate(self):
        ap = 0.0
        start = time()
        data_size = len(self.test_data)
        for i,(tens, labels) in enumerate(self.test_data):
            pred = self.inference(tens)
            labels = labels.to(self.device)
            former = labels[:, :1]
            latter = labels[:, 1:-1]
            cls = labels[:, -1:]
            labels = cat([former, cls, ones(labels.shape[0], 1, device=self.device), latter], dim=-1)
            ap += mean_average_precision(pred.tolist(), labels.tolist(), num_classes=self.num_classes)
            show_progress_counter(i+1, data_size, start, f"AP: {ap/(i+1)}", 0, 1)
        return ap/data_size