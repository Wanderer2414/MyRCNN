from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear, NLLLoss
from torch import Tensor, device, save, tensor, softmax, arange, stack, log, ones_like, where, zeros_like, zeros, bool as tbool, exp, long as tlong
from torch.nn.functional import cross_entropy
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
        self.color = ColorHead.ColorHead(device=device)
        self.feat = FeatureHead.FeatureHead(out_channels=64, device=device)
        self.colorWeight = Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device)
        self.ft = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device)
        )
        self.mix = Linear(in_features=64, out_features=64, device=device)
        self.cls = Classfication.Classification(channels=64, num_classes=100, device=device)
    def forward(self, x:Tensor) -> list[Tensor]:
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(x)
        feature: Tensor = self.feat(mask, color, x)
        color = color.expand(-1, 64, -1, -1).contiguous()
        combine: Tensor = feature + self.colorWeight(color)
        M: Tensor = combine[:].max()
        combine = combine / M
        combine = self.ft(combine)
        combine = combine.permute(0, 2, 3, 1)
        combine = self.mix(combine)
        combine = combine.permute(0, 3, 1, 2)
        scores: list[Tensor] = self.cls(combine)
        return scores
    
def MyLoss(scores: list[Tensor], label: Tensor) -> Tensor:
    x1, y1, x2, y2, cls = label[0][0]
    cls = cls.long()
    out: Tensor = tensor([0], device=label.device)
    for mat in scores:
        x1 /= 5
        y1 /= 5
        x2 /= 5
        y2 /= 5
        X1 = x1.floor().long()
        Y1 = y1.floor().long()
        X2 = x2.ceil().long()
        Y2 = y2.ceil().long()
        B, H, W = mat.shape[0:3]
        # rol = arange(H, device=label.device).view(1, H, 1).expand(1, H, W)
        # col = arange(W, device=label.device).view(1, 1, W).expand(1, H, W)
        # indices = (rol>y) & (rol < y+h) & (col > x) & col < (x + h)
        logits = 1-mat.permute(0, 3, 1, 2)
        logits[:, :, Y1:Y2, X1:X2] = 1-logits[:, :, Y1:Y2, X1:X2]
        target = zeros((B, H, W), dtype=tlong, device=mat.device)
        target[:, :, :] = cls
        score = cross_entropy(logits, target)
        out = out + score
    return out

class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-5)
        self.device = device
    def train(self, x: Dataset, loss: Callable[[list[Tensor], Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for epoach in range(50):
            sloss = 0
            for i in range(10):
                tens:Tensor = x.getTrainTensor(i).to(self.device)
                label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
                if (label.shape[1] == 0):
                  continue
                out: list[Tensor] = self.model(tens)
                lss = loss(out,label)
                self.opt.zero_grad()
                lss.backward()
                self.opt.step()
                # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                sloss += lss
                if ((i+1) % (size//5) == 0):
                    print(f"Saved: {(i+1)} / {size//5} progress")
                    save(self.model.state_dict(), "model.pth")
            show_progress_counter(epoach+1, 50, start, f"{sloss/10}")
            save(self.model.state_dict(), "model.pth")