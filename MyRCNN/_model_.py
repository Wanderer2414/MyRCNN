from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear
from torch import Tensor, device, save, tensor, softmax, arange, stack, log, ones_like, where, zeros_like, zeros, bool
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
        self.color = ColorHead.ColorHead()
        self.feat = FeatureHead.FeatureHead(out_channels=64)
        self.colorWeight = Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device)
        self.downward = Sequential(
            MaxPool2d(kernel_size=3, stride=1, padding=1),
            Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, device=device)
        )
        self.ft = Sequential(
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device),
            ReLU(),
            Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device)
        )
        self.mix = Linear(in_features=64, out_features=64, device=device)
        self.cls = Classfication.Classification(channels=64, num_classes=100, device=device)
    def forward(self, x:Tensor) -> list[Tensor]:
        x = self.downward(x)
        mask: Tensor = self.mask(x)
        color: Tensor = self.color(mask, x)
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
    x, y, w, h, cls = label[0][0].detach().numpy()
    cls = int(cls)
    mask = zeros(100, device=label.device, dtype=bool)
    mask[cls] = 1
    out: Tensor = tensor([0], device=label.device)
    for mat in scores:
        B, H, W = mat.shape[0:3]
        rol = arange(H).view(1, H, 1).expand(1, H, W)
        col = arange(W).view(1, 1, W).expand(1, H, W)
        indices = (rol>y) & (rol < y+h) & (col > x) & col < (x + h)
        score = where(mask, -log(mat[indices]), -log(ones_like(mat[indices])-mat[indices])).sum()
        out = out + score
        x /= 5
        y /= 5
        w /= 5
        h /= 5
    return out

class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = MyRCNN(channels=3, device=device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.device = device
    def train(self, x: Dataset, loss: Callable[[list[Tensor], Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for i in range(size):
            tens:Tensor = x.getTrainTensor(i).to(self.device)
            label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
            if (label.shape[1] == 0):
              continue
            out: list[Tensor] = self.model(tens)
            lss = loss(out,label)
            self.opt.zero_grad()
            lss.backward()
            self.opt.step()
            show_progress_counter(i+1, size, start, f"Loss: {lss}")
        show_progress_counter(size, size, start, "Done")
        save(self.model.state_dict(), "model.pth")