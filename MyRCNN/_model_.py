from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like, load, stack, sigmoid, floor, int64, where, sort, tensor_split, ones, no_grad
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import complete_box_iou_loss, roi_align
from torch.optim import Adam
from dataset import Dataset
from typing import Callable
from time import time
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
from utils import non_max_suppression, mean_average_precision
import os


row = arange(400, dtype=tfloat, device="cuda").view(1,1,400,1).expand(1,1,400,400)
col = arange(400, dtype=tfloat, device="cuda").view(1,1,1,400).expand(1,1,400,400)
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
    def forward(self, x:Tensor) -> tuple[Tensor,Tensor, Tensor, Tensor]:
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(x)
        score, bbx = self.feat(mask, color)
        return mask, color, score, bbx
def MyBBLoss(scores: list[Tensor], labels: list[Tensor]) -> Tensor:
    C_gt = 0
    
    score_box = zeros(0, 1, 400, 400, device=scores[0].device)
    whs = zeros(0, 2, 400, 400, device=scores[0].device)
    for score, label in zip(scores, labels):
        score_box = cat([score_box, roi_align(score[:, 0:1, :, :], [label[:, 0:4]], (400, 400))], dim=0)
        
        boxes = label[:, 0:4]
        wh = roi_align(score[:, 1:3, :, :], [label[:, 0:4]], (400, 400))
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        N = label.shape[0]
        wh_gt = stack([w,h], dim=0).view(N, 2, 1, 1)
        wh /= wh_gt/400
        whs = cat([whs, wh], dim=0)
    C_gt += score_box.shape[0]
    
    score =  binary_cross_entropy_with_logits(score_box, target.expand(C_gt, 1, 400, 400))

    
    FIoULoss = FIoU(score_box, whs)
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
    pred_x1 = -w/2 + col
    pred_x2 = w/2 + col
    pred_y1 = -h/2 + row
    pred_y2 = h/2 + row
    
    pred_x1 = (pred_x1*indices).sum(dim=(-2, -1))
    pred_x2 = (pred_x2*indices).sum(dim=(-2, -1))
    pred_y1 = (pred_y1*indices).sum(dim=(-2, -1))
    pred_y2 = (pred_y2*indices).sum(dim=(-2, -1))
    
    pred_w = (w*indices/W).sum(dim=(-2, -1))
    pred_h = (h*indices/H).sum(dim=(-2, -1))
    return ((1-pred_w).square() + (1-pred_h).square()).mean()

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
        label (Tensor): [1, 1, 5] [x1, y1, x2, y2, cls]

    Returns:
        Tensor: 
    """
    N = cls.shape[0]
    cls_label = label[0, 0, -1].long()
    cls_target = cls_label.expand(N)
    loss = cross_entropy(cls, cls_target, reduction="mean")
    return loss
batch_size = 4
class Model:
    def __init__(self, dt:Dataset, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.cls = Classfication.Classification(boundary_channels=1, color_channels=32, num_classes=dt.getClassSize(), device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        self.data = dt
        self.device = device
        self.num_classes = dt.getClassSize()
    def inference(self, x:Tensor) -> Tensor:
        mask, color, score, bbx = self.model(x.to(device=self.device))
        bbx = bbx.squeeze(0)
        cls:Tensor = self.cls(mask, color, bbx[:, :-1])
        cls = cls * bbx[:, -1:]
        N = cls.shape[0]
        cls_range = arange(self.num_classes, device=self.device).view(self.num_classes, 1, 1).expand(self.num_classes, N, 1)
        cls = cls.permute(1, 0).unsqueeze(-1)
        bbx = bbx[:, :-1].unsqueeze(0).expand(self.num_classes, N, 4)
        result = cat([cls_range, cls, bbx], dim=-1).view(N*self.num_classes, 6)
        ls = non_max_suppression(result.tolist(), 0.8, 0.5)
        return tensor(ls, device=self.device)
        
    def train(self):
        size = self.data.getTrainSize()
        start = time()
        if (os.path.exists("bbx.pth")):
            self.model.load_state_dict(load("bbx.pth", map_location=self.device))
            print("Load model!")
        else:
            epoches = 5
            for epoch in range(epoches):
                for i in range(0, self.data.getTrainSize(), batch_size):
                    tens = []
                    label = []
                    out = []
                    for j in range(batch_size):
                        if (i+j>=self.data.getTrainSize()):
                            break
                        ten = self.data.getTrainTensor(i+j).to(self.device)
                        tens.append(ten)
                        label.append(x.getTrainLabel(i+j).to(self.device))
                        out.append(self.model(ten)[-2])
                    # boxes = label[:, :, 1:].squeeze(0)
                    # cls = self.cls(mask, color, boxes)
                    lss = MyBBLoss(out, label)
                    self.opt.zero_grad()
                    lss.backward()
                    self.opt.step()
                    # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                    # if ((i+1) % (size//5) == 0):
                    #     print(f"Saved: {(i+1)} / {size//5} progress")
                    #     save(self.model.state_dict(), "bbx.pth")
                    show_progress_counter(i+1, self.data.getTrainSize(), start, f"Epoch {epoch/epoches}; Loss {lss}", epoch, epoches)
                    if (i%100 == 0):
                        save(self.model.state_dict(), "bbx.pth")
                save(self.model.state_dict(), "bbx.pth")
        if (os.path.exists("cls.pth")):
            self.cls.load_state_dict(load("cls.pth", map_location=self.device))
            print("Load model!")
        else:
            start = time()
            epoches = 5
            for epoch in range(epoches):
                for i in range(self.data.getTrainSize()):
                    tens:Tensor = self.data.getTrainTensor(i).to(self.device)
                    label:Tensor =  self.data.getTrainLabel(i).unsqueeze(0).to(self.device)
                    if (label.shape[1] == 0):
                      continue
                    mask, color, score, bbx = self.model(tens)
                    boxes = label[:, :, 1:].squeeze(0)
                    cls = self.cls(mask, color, boxes)
                    lss = ClsLoss(cls, label)
                    self.opt2.zero_grad()
                    lss.backward()
                    self.opt2.step()
                    # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                    # if ((i+1) % (size//5) == 0):
                    #     print(f"Saved: {(i+1)} / {size//5} progress")
                    #     save(self.cls.state_dict(), "cls.pth")
                    show_progress_counter(i+1, self.data.getTrainSize(), start, f"Epoch {epoch}/{epoches}; Loss {lss}", epoch, epoches)
                    if (i%100 == 0):
                        save(self.cls.state_dict(), "cls.pth")
                save(self.cls.state_dict(), "cls.pth")
    def Evaluate(self):
        ap = 0.0
        start = time()
        for i in range(self.data.getTestSize()):
            pred = self.inference(self.data.getTestTensor(i))
            pred = cat([ones(pred.shape[0], 1, device=self.device)*i, pred], dim=1)
            label = self.data.getTestLabel(i).to(device=self.device)
            one = ones(label.shape[0], 1, device=self.device)
            label = cat([one*i, label[:, -1:], one, label[:, :-1]], dim=1)
            ap += mean_average_precision(pred.tolist(), label.tolist(), num_classes=self.data.getClassSize())
            show_progress_counter(i+1, self.data.getTestSize(), start, f"AP: {ap/(i+1)}", 0, 1)
        return ap/self.data.getTestSize()