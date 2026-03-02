from torch.nn import Module, Sequential
from torch import Tensor, device
from . import Stage1, Stage2, Stage3, Stage3 as Stage4, Stage3 as Stage5, FinalStage

class ResNet(Module):
    def __init__(self, device: device = device("cpu")):
        super().__init__()
        self.net = Sequential(
            Stage1.InitialLayer(in_channels=3, out_channels=64, device=device),
            Stage2.Model(in_channels=64, out_channels=256, device=device),
            Stage3.Model(in_channels=256, out_channels=512, num_layer=4, device=device),
            Stage4.Model(in_channels=512, out_channels=1024, num_layer=6, device=device),
            Stage5.Model(in_channels=1024, out_channels=2048, num_layer=3, device=device),
            FinalStage.Model(channels=2048, num_classes=1000)
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)