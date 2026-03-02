from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, softmax, arange, stack, log, ones_like, where, zeros_like, zeros, bool as tbool, exp, long as tlong, cat
from torch.nn.functional import cross_entropy, binary_cross_entropy_with_logits
from torchvision.ops import generalized_box_iou
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
        self.color = ColorHead.ColorHead(out_channels=16, device=device)
        self.feat = FeatureHead.FeatureHead(out_channels=16, device=device)
        self.cls = Classfication.Classification(channels=16, num_classes=100, device=device)
    def forward(self, x:Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(mask, x)
        feature: Tensor = self.feat(mask, color)
        return self.cls(color, feature)
    
def MyLoss(scores: Tensor, label: Tensor) -> Tensor:
    # Score, width, height, class x 100
    label = label.squeeze().squeeze()
    B, C, H, W = scores.shape
    X1, Y1, X2, Y2 = label[0:4]
    X1 = X1.floor().long()
    X2 = X2.ceil().long()
    Y1 = Y1.floor().long()
    Y2 = Y2.ceil().long()
    center = tensor([X1+X2, Y1+Y2], device=label.device).view(2, 1, 1, 1, 1).expand(2, 1,1,H,W)
    row = arange(H, device=label.device).view(1,1,H,1).expand(1,1,H,W)
    col = arange(W, device=label.device).view(1,1,1,W).expand(1,1,H,W)
    distance = ((row-center[1]).square() + (col-center[0]).square()).sqrt()
    indices = ((row >= Y1) & (row <= Y2)) & ((col >= X1) & (col <= X2))
    score = distance.clone()
    score = (1-score/score.max())
    region_pred = scores[:, 0, Y1:Y2, X1:X2]
    target = score[:, 0, Y1:Y2, X1:X2]
    score = binary_cross_entropy_with_logits(region_pred, target)
    target = label[0:4].view(-1, 4)
    position = cat([col, row, col, row], dim=1)
    w = scores[:, 1, :, :].unsqueeze(1)
    h = scores[:, 2, :, :].unsqueeze(1)
    position[:, 0] = position[:, 0] - w
    position[:, 1] = position[:, 1] - h
    position[:, 2] = position[:, 2] + w
    position[:, 3] = position[:, 3] + h
    position = position[:, :, Y1:Y2, X1:X2]
    position = position.reshape(-1, 4)
    score = score + generalized_box_iou(position, target)
    cls_target = zeros(B, H, W, device=label.device, dtype=tlong)
    cls_target[:, Y1:Y2, X1:X2] = label[-1].long()
    cls = scores[:, 3:, :, :]
    score = cross_entropy(cls, cls_target)
    return score

class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.device = device
    def train(self, x: Dataset, loss: Callable[[Tensor, Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for epoach in range(100):
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
            show_progress_counter(epoach+1, 100, start, f"{sloss/10}")
            save(self.model.state_dict(), "model.pth")