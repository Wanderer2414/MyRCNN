from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, arange, zeros, bool as tbool, exp, long as tlong, cat, maximum,float as tfloat, zeros_like
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import complete_box_iou_loss
from torch.optim import Adam
from dataset import Dataset
from typing import Callable
from time import time
from display import show_progress_counter
from . import ColorHead, MaskHead, FeatureHead, Classfication
class MyRCNN(Module):
    def __init__(self, channels: int, device: device = device("cpu"))->None:
        super().__init__()
        self.mask = MaskHead.MaskHead(device=device)
        self.color = ColorHead.ColorHead(in_channels=3, out_channels=8, device=device)
        self.feat = FeatureHead.FeatureHead(color_channels=8, mask_channels=1, num_classes=100, device=device)
    def forward(self, x:Tensor) -> Tensor:
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(x)
        feature: Tensor = self.feat(mask, color)
        return feature
def MyBBLoss(scores: Tensor, label: Tensor) -> Tensor:
    label = label.squeeze().squeeze()
    X1, Y1, X2, Y2 = label[0:4]
    X1 = X1.floor().long()
    X2 = X2.ceil().long()
    Y1 = Y1.floor().long()
    Y2 = Y2.ceil().long()
    B, C, H, W = scores.shape
    score = scores[:, 0:1, :, :]
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
    h, w = target.shape[2:4]
    h //= 2
    w //= 2
    # print(f"Target: {target[0, 0, h, w]}, Logits: {pred_box[0, 0, h, w]} ({h}, {w}, {target.shape}")
    score =  binary_cross_entropy_with_logits(pred_box, target)
    return score
def MyLoss(scores: Tensor, label: Tensor) -> Tensor:
    # Score, width, height, class x 100
    label = label.squeeze().squeeze()
    B, C, H, W = scores.shape
    X1, Y1, X2, Y2 = label[0:4]
    X1 = X1.floor().long()
    X2 = X2.ceil().long()
    Y1 = Y1.floor().long()
    Y2 = Y2.ceil().long()
    center = tensor([(X1+X2)/2, (Y1+Y2)/2], device=label.device).view(2, 1, 1, 1, 1).expand(2, 1,1,H,W)
    row = arange(H, device=label.device, dtype=tfloat).view(1,1,H,1).expand(1,1,H,W)
    col = arange(W, device=label.device, dtype=tfloat).view(1,1,1,W).expand(1,1,H,W)
    distance = ((row-center[1]).square() + (col-center[0]).square()).sqrt()
    score = distance.clone()
    score = (1-score/score.max())
    region_pred = scores[:, 0, Y1:Y2, X1:X2]
    target = score[:, 0, Y1:Y2, X1:X2]
    score = binary_cross_entropy_with_logits(region_pred, target)
    target = label[0:4].view(-1, 4)
    w = scores[:, 1, :, :].unsqueeze(1)
    h = scores[:, 2, :, :].unsqueeze(1)
    x1 = col.clone() - w
    y1 = col.clone() - h
    x2 = col.clone() + w
    y2 = col.clone() + h
    position = cat([x1, y1, x2, y2], dim=-1)
    w = w[:, :, Y1:Y2, X1:X2]
    h = h[:, :, Y1:Y2, X1:X2]
    position = position.reshape(-1, 4)
    target = target.expand(position.shape)
    iou_score = complete_box_iou_loss(position, target)
    score = score + iou_score.mean()
    cls_target = zeros(B, H, W, device=label.device, dtype=tlong)
    cls_target[:, Y1:Y2, X1:X2] = label[-1].long()
    cls = scores[:, 3:, :, :]
    score = score + cross_entropy(cls, cls_target)    
    return score

class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-2)
        self.device = device
    def train(self, x: Dataset, loss: Callable[[Tensor, Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for epoch in range(100):
            sloss = 0
            for i in range(10):
                tens:Tensor = x.getTrainTensor(i).to(self.device)
                label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
                if (label.shape[1] == 0):
                  continue
                out: Tensor = self.model(tens)
                lss = loss(out,label)
                self.opt.zero_grad()
                lss.backward()
                self.opt.step()
                # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                sloss += lss
                if ((i+1) % (size//5) == 0):
                    print(f"Saved: {(i+1)} / {size//5} progress")
                    save(self.model.state_dict(), "model.pth")
            show_progress_counter(epoch+1, 100, start, f"{sloss/10}")
            save(self.model.state_dict(), "model.pth")