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
        module = super().to(*args, **kwargs)
        device = kwargs.get("device", None)
        if device is None and len(args) > 0:
            device = args[0]
        self.device = device
        for module in self:
            module.to(*args, **kwargs)
        return module
    def forward(self, x:Tensor):
        boundary: Tensor = self.mask(x)
        color = self.color(x)
        # if (self.training):
        score = self.feat(boundary, color)
        return boundary, score
        # else: 
            # score, bbx = self.feat(boundary, color)
            # return boundary, score, bbx
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
class Model(Module):
    _modules: dict[str, Module] # type:ignore
    def __init__(self, dataset: str):
        super().__init__()
        if (dataset == "Pascal"):
            num_classes = len(config.PASCAL_CLASSES)
            self.model = MyRCNN(config.BATCH_SIZE,channels=3, num_classes=num_classes)
            self.cls = Classfication.Classification(mask_channels=32, num_classes=num_classes)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        self.current_epoches = 0
        self.dataset = dataset
        self.device = device("cpu")
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
            if (self.dataset == "Pascal"):
                train_data = YOLODataset(config.TRAIN_DIR, config.IMG_DIR, config.LABEL_DIR, config.ANCHORS, transform=config.train_transforms)
                train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, collate_fn=collect_fn, num_workers=config.NUM_WORKERS);
                start = time()
                data_size = len(train_loader)
                for i, (tens, labels) in enumerate(train_loader):
                    boundary, score = self.model(tens.to(self.device))
                    lss = MyBBLoss(score, labels.to(self.device))
                    self.opt.zero_grad()
                    lss.backward()
                    self.opt.step()
                    show_progress_counter(i+1, data_size, start, f"Epoch {self.current_epoches}/{self.current_epoches+1}; Loss {lss}")
                    if (i%100 == 0):
                        save(self.model.state_dict(), "bbx.pth")
                save(self.model.state_dict(), "bbx.pth")
                start = time()
                data_size = len(train_loader)
                for i, (tens, labels) in enumerate(train_loader):
                    tens = tens.to(self.device)
                    labels = labels.to(self.device)
                    boundary, score = self.model(tens)
                    boxes = labels[:, 0:5]
                    cls = self.cls(boundary, score[:, 0:1, :, :], tens, boxes)
                    lss = ClsLoss(cls, labels)
                    self.opt2.zero_grad()
                    lss.backward()
                    self.opt2.step()
                    show_progress_counter(i+1, data_size, start, f"Epoch {self.current_epoches}/{self.current_epoches+1}; Loss {lss}")
                    if (i%100 == 0):
                        save(self.cls.state_dict(), "cls.pth")
                save(self.cls.state_dict(), "cls.pth")
        else:
            if (self.dataset == "Pascal"):
                test_data = YOLODataset(config.TEST_DIR, config.IMG_DIR, config.LABEL_DIR, config.ANCHORS, transform=config.train_transforms)
                test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE, num_workers=config.NUM_WORKERS, collate_fn=collect_fn);
                ap = 0.0
                start = time()
                data_size = len(test_loader)
                for i,(tens, labels) in enumerate(test_loader):
                    pred = self.inference(len(config.PASCAL_CLASSES), tens)
                    labels = labels.to(self.device)
                    former = labels[:, :1]
                    latter = labels[:, 1:-1]
                    cls = labels[:, -1:]
                    labels = cat([former, cls, ones(labels.shape[0], 1, device=self.device), latter], dim=-1)
                    ap += mean_average_precision(pred.tolist(), labels.tolist(), num_classes=len(config.PASCAL_CLASSES))
                    show_progress_counter(i+1, data_size, start, f"mAP: {ap/(i+1)}")
                print(f"mAP: {ap/data_size}")
        return self