from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like, load, stack, sigmoid
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import complete_box_iou_loss, roi_align
from torch.optim import Adam
from dataset import Dataset
from typing import Callable
from time import time
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
import os


row = arange(400, dtype=tfloat, device="cuda").view(1,1,400,1).expand(1,1,400,400)
col = arange(400, dtype=tfloat, device="cuda").view(1,1,1,400).expand(1,1,400,400)
center_x = center_y = 200
distance = ((col-center_x).square() + (row-center_y).square()).sqrt()
target = distance.min(dim=1, keepdim=True).values
target = 1-target/target.max()

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
def MyBBLoss(scores: list[Tensor], labels: list[Tensor]) -> Tensor:
    C_gt = 0
    
    score_box = zeros(0, 1, 400, 400, device="cuda")
    whs = zeros(0, 2, 400, 400, device="cuda")
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
    
    indices = sigmoid(score) > 0.8
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
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.cls = Classfication.Classification(boundary_channels=1, color_channels=32, num_classes=100, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.opt2 = Adam(self.cls.parameters(), lr=1e-4)
        self.device = device
    def train(self, x: Dataset):
        size = x.getTrainSize()
        start = time()
        if (os.path.exists("bbx.pth")):
            self.model.load_state_dict(load("bbx.pth", map_location=self.device))
            print("Load model!")
        else:
            for i in range(0, x.getTrainSize(), batch_size):
                tens = []
                label = []
                out = []
                for j in range(batch_size):
                    if (i+j>=x.getTrainSize()):
                        break
                    ten = x.getTrainTensor(i+j).to(self.device)
                    tens.append(ten)
                    label.append(x.getTrainLabel(i+j).to(self.device))
                    out.append(self.model(ten)[-1])
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
                show_progress_counter(i+1, x.getTrainSize(), start, f"{lss}")
                if (i%100 == 0):
                    save(self.model.state_dict(), "bbx.pth")
            save(self.model.state_dict(), "bbx.pth")
            
        # start = time()
        # for i in range(x.getTrainSize()):
        #     tens:Tensor = x.getTrainTensor(i).to(self.device)
        #     labelA:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
        #     if (labelA.shape[1] == 0):
        #       continue
        #     maskA, colorA, outA = self.model(tens)
        #     boxes = labelA[:, :, 1:].squeeze(0)
        #     cls = self.cls(maskA, colorA, boxes)
        #     lss = ClsLoss(cls, labelA)
        #     self.opt2.zero_grad()
        #     lss.backward()
        #     self.opt2.step()
        #     # show_progress_counter(i+1, size, start, f"Loss: {lss}")
        #     # if ((i+1) % (size//5) == 0):
        #     #     print(f"Saved: {(i+1)} / {size//5} progress")
        #     #     save(self.cls.state_dict(), "cls.pth")
        #     show_progress_counter(i+1, x.getTrainSize(), start, f"{lss}")
        #     save(self.cls.state_dict(), "cls.pth")