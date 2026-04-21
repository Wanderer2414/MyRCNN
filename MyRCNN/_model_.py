from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like, load, stack, sigmoid, floor, int64, where, sort, tensor_split, ones, no_grad, softmax
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import distance_box_iou_loss, roi_align
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from typing import Callable
from time import time
from torch.utils.data import DataLoader
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
from utils import non_max_suppression, mean_average_precision
from dataset2 import YOLODataset, collect_fn
from typing import Iterator
import config
import os

dev = config.DEVICE
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
    _modules: dict[str, Module] #type: ignore
    def __init__(self, batch: int, channels: int, num_classes: int)->None:
        super().__init__()
        self.mask = MaskHead.MaskHead()
        self.color = ColorHead.ColorHead(batch, in_channels=3, half_out_channels=16)
        self.feat = FeatureHead.FeatureHead(batch, half_color_channels=16, mask_channels=1, num_classes=num_classes)
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    def __len__(self):
        return len(self._modules)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        self.device = device
        for module in self:
            module.to(*args, **kwargs)
        return self
    
    def forward(self, x:Tensor):
        color = self.color(x)
        if (self.training):
            return self.feat(color)
        else: 
            boundary: Tensor = self.mask(x)
            score, bbx = self.feat(color)
            return boundary, score, bbx
def MyBBLoss(scores: Tensor, labels: Tensor) -> Tensor:
    C_gt = 0
    score_box = roi_align(scores[:, 0:1, :, :], labels[:, 0:5], (400, 400))  # type: ignore[assignment]
    wh = roi_align(scores[:, 1:3, :, :], labels[:, 0:5], (400, 400)) # type: ignore[assignment]
    C_gt = score_box.shape[0]
    
    score =  binary_cross_entropy_with_logits(score_box, target.expand(C_gt, 1, 400, 400))

    x1_gt = labels[:, 1]
    y1_gt = labels[:, 2]
    x2_gt = labels[:, 3]
    y2_gt = labels[:, 4]
    w_gt = (x2_gt - x1_gt).view(wh.shape[0], 1, 1, 1)
    h_gt = (y2_gt - y1_gt).view(wh.shape[0], 1, 1, 1)
    w = wh[:, 0:1, :, :]/(w_gt+1e-7)
    h = wh[:, 1:2, :, :]/(h_gt+1e-7)
    wh = cat([w,h],dim=1)
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
    w = wh[:, 0:1, :, :]*400
    h = wh[:, 1:2, :, :]*400
    
    indices = (sigmoid(score) > 0.8) * score
    N = score.shape[0]
    indices = indices/(indices.sum(dim=(-2, -1), keepdim=True)+eps)
    x = arange(W, dtype=tfloat, device=score.device).view(1, 1, 1, W).expand(1, 1, H, W)
    y = arange(H, dtype=tfloat, device=score.device).view(1, 1, H, 1).expand(1, 1, H, W)
    
    pred_w = (w*indices).sum(dim=(-2, -1))
    pred_h = (h*indices).sum(dim=(-2, -1))
    pred_x = (x*indices).sum(dim=(-2, -1))
    pred_y = (y*indices).sum(dim=(-2, -1))
    pred_x1 = pred_x - pred_w/2
    pred_x2 = pred_x + pred_w/2
    pred_y1 = pred_y - pred_h/2
    pred_y2 = pred_y + pred_h/2
    boxes = cat([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    target = tensor([[0, 0, 400, 400]], device=score.device).expand(boxes.shape[0], 4)
    # print(boxes)
    # print(target)
    loss = distance_box_iou_loss(boxes,target, reduction="mean")
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
class Model(Module):
    _modules: dict[str, Module] # type:ignore
    def __init__(self, bbx_epoches: int =1, cls_epoches: int = 1):
        super().__init__()
        self.model = MyRCNN(config.BATCH_SIZE,channels=3, num_classes=config.NUM_CLASSES)
        self.cls = Classfication.Classification(mask_channels=32, num_classes=config.NUM_CLASSES)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        clip_grad_norm_(self.model.parameters(), max_norm=1.0) 
        clip_grad_norm_(self.cls.parameters(), max_norm=1.0) 
        self.current_bbx_epoches = 0
        self.current_cls_epoches = 0
        self.bbx_epoches = bbx_epoches
        self.cls_epoches = cls_epoches
        self.num_classes = config.NUM_CLASSES
    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())
    def __len__(self):
        return len(self._modules)
    
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        self.device = device
        for module in self:
            module.to(*args, **kwargs)
        return self
    def inference(self, num_classes:int, x:Tensor) -> Tensor:
        boundary, score, bbx = self.model(x)
        cls:Tensor = self.cls(boundary, score[:, 0:1, :, :], x, bbx[:, :-1])
        cls_score:Tensor = softmax(cls, dim=-1)
        cls_score = cls * bbx[:, -1:]
        cls_score = cls_score.unsqueeze(-1)
        N = cls.shape[0]
        cls_range = arange(num_classes, device=x.device).view(1, num_classes, 1).expand(N, num_classes, 1)
        batch = bbx[:, 0:1].view(N, 1, 1).expand(N, self.num_classes, 1)
        bbx = bbx[:, 1:5].view(N, 1, 4).expand(N, self.num_classes, 4)
        result = cat([batch, cls_range, cls_score, bbx], dim=-1).view(N*num_classes, 7)
        return result
    
    def train(self, mode: bool = True):
        super().train(mode)
        if (mode):
            data = YOLODataset(config.TRAIN_DIR, config.IMG_DIR, config.LABEL_DIR, config.ANCHORS, transform=config.train_transforms)
            self.loader = DataLoader(data, batch_size=config.BATCH_SIZE, collate_fn=collect_fn, num_workers=config.NUM_WORKERS);
            
            start = time()
            self.model.train()
            data_size = len(self.loader)
            total_epoches = self.current_bbx_epoches + self.bbx_epoches
            for epoch in range(self.bbx_epoches):
                for i, (tens, labels) in enumerate(self.loader):
                    score = self.model(tens.to(self.device))
                    lss = MyBBLoss(score, labels.to(self.device))
                    self.opt.zero_grad()
                    lss.backward()
                    self.opt.step()
                    show_progress_counter(i+1, data_size, start, f"Epoch {self.current_bbx_epoches}/{total_epoches}; Loss {lss}", epoch, self.bbx_epoches)
                    if (i%100 == 0):
                        save(self.model.state_dict(), "bbx.pth")
                self.current_bbx_epoches += 1
                save(self.state_dict(), "model.pth")
            self.model.eval()
            start = time()
            data_size = len(self.loader)
            total_epoches = self.current_cls_epoches + self.cls_epoches
            for epoch in range(self.cls_epoches):
                for i, (tens, labels) in enumerate(self.loader):
                    tens = tens.to(self.device)
                    labels = labels.to(self.device)
                    boundary, score, bbx = self.model(tens)
                    boxes = labels[:, 0:5]
                    cls = self.cls(boundary, score[:, 0:1, :, :], tens, boxes)
                    lss = ClsLoss(cls, labels)
                    self.opt2.zero_grad()
                    lss.backward()
                    self.opt2.step()
                    show_progress_counter(i+1, data_size, start, f"Epoch {self.current_cls_epoches}/{total_epoches}; Loss {lss}", epoch, self.cls_epoches)
                    if (i%100 == 0):
                        save(self.cls.state_dict(), "cls.pth")
                self.current_cls_epoches+=1
                save(self.state_dict(), "model.pth")
        else:
            data = YOLODataset(config.TEST_DIR, config.IMG_DIR, config.LABEL_DIR, config.ANCHORS, transform=config.test_transforms)
            self.loader = DataLoader(data, batch_size=1, num_workers=1, collate_fn=collect_fn);
    
        #     self.model.train()
        #     self.cls.train()
        #     ap = 0.0
        #     start = time()
        #     data_size = len(self.loader)
        #     for i,(tens, labels) in enumerate(self.loader):
        #         pred = self.inference(config.NUM_CLASSES, tens)
        #         labels = labels.to(self.device)
        #         former = labels[:, :1]
        #         latter = labels[:, 1:-1]
        #         cls = labels[:, -1:]
        #         labels = cat([former, cls, ones(labels.shape[0], 1, device=self.device), latter], dim=-1)
        #         ap += mean_average_precision(pred.tolist(), labels.tolist(), num_classes=config.NUM_CLASSES)
        #         show_progress_counter(i+1, data_size, start, f"mAP: {ap/(i+1)}", 0, 1)
        #     print(f"mAP: {ap/data_size}")
        return self
    def forward(self, x:Tensor) -> Tensor:
        boundary, score, bbx = self.model(x)
        cls:Tensor = self.cls(boundary, score[:, 0:1, :, :], x, bbx[:, :-1])
        # cls_score:Tensor = softmax(cls, dim=-1)
        # cls_score = cls * bbx[:, -1:]
        # cls_score = cls_score.unsqueeze(-1)
        # N = cls.shape[0]
        # cls_range = arange(self.num_classes, device=x.device).view(1, self.num_classes, 1).expand(N, self.num_classes, 1)
        # batch = bbx[:, 0:1].view(N, 1, 1).expand(N, self.num_classes, 1)
        # bbx = bbx[:, 1:5].view(N, 1, 4).expand(N, self.num_classes, 4)
        # result = cat([batch, cls_range, cls_score, bbx], dim=-1).view(N*self.num_classes, 7)
        return cls
        