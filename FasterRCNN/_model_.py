import ResNet50
from torch.nn import Module, Sequential
from torch import Tensor
from . import RPN
class FasterRCNN(Module):
    def __init__(self, device):
        super().__init__()
        self.feature_extraction = Sequential(
            ResNet50.Stage1.InitialLayer(in_channels=3, out_channels=64),
            ResNet50.Stage2.Model(in_channels=64, out_channels=256),
            ResNet50.Stage3.Model(in_channels=256, out_channels=512, num_layer=4),
            ResNet50.Stage4.Model(in_channels=512, out_channels=1024, num_layer=6),
        )
        self.rpn = RPN.Model(channels=1024, device=device)
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.feature_extraction(x)
        cls, reg = tuple[Tensor,Tensor](self.rpn(x))
        return cls, reg