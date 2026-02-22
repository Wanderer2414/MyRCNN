from torch.nn import Module, Conv2d, Sequential, ReLU, MaxPool2d, Linear
from torch import Tensor, device
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
        combine[:] = combine[:]/combine[:].max()
        combine = self.ft(combine)
        combine = combine.permute(0, 2, 3, 1)
        combine = self.mix(combine)
        combine = combine.permute(0, 3, 1, 2)
        scores: list[Tensor] = self.cls(combine)
        return scores