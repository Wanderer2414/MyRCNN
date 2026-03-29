from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like, load
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import complete_box_iou_loss
from torch.optim import Adam
from dataset import Dataset
from typing import Callable
from time import time
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
import os
class MyRCNN(Module):
    def __init__(self, channels: int, device: device = device("cpu"))->None:
        super().__init__()
        self.mask = MaskHead.MaskHead(device=device)
        self.color = ColorHead.ColorHead(in_channels=3, half_out_channels=16, device=device)
        self.feat = FeatureHead.FeatureHead(half_color_channels=16, mask_channels=1, num_classes=100, device=device)
    def forward(self, x:Tensor) -> tuple[Tensor,Tensor,Tensor]:
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(x)
        feature: Tensor = self.feat(mask, color)
        return mask, color, feature
def MyBBLoss(scores: list[Tensor], label: Tensor) -> Tensor:
    label = label.squeeze().squeeze()
    X1, Y1, X2, Y2 = label[0:4]
    X1 = X1.floor().long()
    X2 = X2.ceil().long()
    Y1 = Y1.floor().long()
    Y2 = Y2.ceil().long()
    B, C, H, W = scores[0].shape
    score = scores[0][:, 0:1, :, :]
    # w = scores[:, 1:2, :, :]
    # h = scores[:, 2:3, :, :]
    row = arange(H, device=label.device, dtype=tfloat).view(1,1,H,1).expand(1,1,H,W)
    col = arange(W, device=label.device, dtype=tfloat).view(1,1,1,W).expand(1,1,H,W)
    # x1 = (col-w).floor()
    # x2 = (col+w).ceil()
    # y1 = (row-h).floor()
    # y2 = (row+h).ceil()
    pred_box = score[:, :, Y1:Y2, X1:X2]
    center = tensor([(X1+X2)/2, (Y1+Y2)/2], device=label.device).view(2, 1, 1, 1, 1).expand(2, 1,1,H,W)
    distance = ((row-center[1]).square() + (col-center[0]).square()).sqrt()
    target = distance[:, :, Y1:Y2, X1:X2]
    target = 1-target/target.max()
    score =  binary_cross_entropy_with_logits(pred_box, target)
    
    # boxes = scores[1][:, :, 1:].squeeze(0)
    # cx = (boxes[:, 0:1] + boxes[:, 2:3])/2
    # cy = (boxes[:, 1:2] + boxes[:, 3:4])/2
    # X1k = (2*X1 + X2)/3
    # X2k = (X1 + 2*X2)/3
    # Y1k = (2*Y1 + Y2)/3
    # Y2k = (Y1 + 2*Y2)/3
    
    # indices = (cx>X1k) & (cx<X2k) & (cy>Y1k) & (cy<Y2k)
    # indices = indices.repeat(1, 4)
    # boxes = boxes[indices].reshape(-1, 4)
    # N = boxes.shape[0]
    # if (N>0):
    #     boxes_gt = label[0:4].unsqueeze(0).repeat(N, 1)
    #     FIoULoss = FIoU(boxes, boxes_gt).mean()
    #     return score + FIoULoss
    return score
def FIoU(boxes: Tensor, boxes_gt: Tensor, eps:float = 1e-7) -> Tensor:
    """Summary

    Args:
        boxes (Tensor): [N, 4] -> [x1,y1,x2,y2]
        boxes_gt (Tensor): [N, 4] -> [x1, y1, x2, y2]

    Returns:
        Tensor: _description_
    """
    pred_x1 = boxes[:, 0:1]
    pred_x2 = boxes[:, 2:3]
    pred_y1 = boxes[:, 1:2]
    pred_y2 = boxes[:, 3:4]
    pred_cX = (pred_x1 + pred_x2)/2
    pred_cY = (pred_y1 + pred_y2)/2
    pred_w = pred_x2 - pred_x1
    pred_h = pred_y2 - pred_y1
    pred_s = pred_w * pred_h
    gt_x1 = boxes_gt[:, 0:1]
    gt_x2 = boxes_gt[:, 2:3]
    gt_y1 = boxes_gt[:, 1:2]
    gt_y2 = boxes_gt[:, 3:4]
    gt_cX = (gt_x1 + gt_x2)/2
    gt_cY = (gt_y1 + gt_y2)/2
    gt_w = gt_x2 - gt_x1
    gt_h = gt_y2 - gt_y1
    gt_s = gt_w * gt_h
    # c2 = 4*(((pred_cX - gt_cX)/gt_w).square() + ((pred_cY - gt_cY)/gt_h).square())
    pred_r = pred_w/(pred_h+eps)
    gt_r = gt_w/(gt_h + eps)
    sq = (pred_s/(gt_s + eps) - 1).square()
    rq = (pred_r - gt_r).square()
    return sq + rq

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
class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.cls = Classfication.Classification(boundary_channels=1, color_channels=32, num_classes=100, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        self.device = device
    def train(self, x: Dataset):
        size = x.getTrainSize()
        start = time()
        if (os.path.exists("bbx.pth") and False):
            self.model.load_state_dict(load("bbx.pth", map_location=self.device))
            print("Load model!")
        else:
            for epoch in range(50):
                sloss = 0
                for i in range(30):
                    tens:Tensor = x.getTrainTensor(i).to(self.device)
                    label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
                    if (label.shape[1] == 0):
                      continue
                    mask, color, out = self.model(tens)
                    boxes = label[:, :, 1:].squeeze(0)
                    cls = self.cls(mask, color, boxes)
                    lss = MyBBLoss(out,label)
                    self.opt.zero_grad()
                    lss.backward()
                    self.opt.step()
                    # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                    sloss += lss
                    if ((i+1) % (size//5) == 0):
                        print(f"Saved: {(i+1)} / {size//5} progress")
                        save(self.model.state_dict(), "bbx.pth")
                show_progress_counter(epoch+1, 50, start, f"{sloss/10}")
                save(self.model.state_dict(), "bbx.pth")
            
        # start = time()
        # for epoch in range(50):
        #     sloss = 0
        #     for i in range(30):
        #         tens:Tensor = x.getTrainTensor(i).to(self.device)
        #         label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
        #         if (label.shape[1] == 0):
        #           continue
        #         mask, color, out = self.model(tens)
        #         boxes = label[:, :, 1:].squeeze(0)
        #         cls = self.cls(mask, color, boxes)
        #         lss = ClsLoss(cls, label)
        #         self.opt2.zero_grad()
        #         lss.backward()
        #         self.opt2.step()
        #         show_progress_counter(i+1, size, start, f"Loss: {lss}")
        #         sloss += lss
        #         if ((i+1) % (size//5) == 0):
        #             print(f"Saved: {(i+1)} / {size//5} progress")
        #             save(self.cls.state_dict(), "cls.pth")
        #     show_progress_counter(epoch+1, 50, start, f"{sloss/10}")
        #     save(self.cls.state_dict(), "cls.pth")