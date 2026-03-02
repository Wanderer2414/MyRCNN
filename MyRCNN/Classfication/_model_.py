from torch.nn import Module, Conv2d, Sequential, LeakyReLU
from torch import Tensor,device, cat
        
class Classification(Module):
    def __init__(self, channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.score = Sequential(
            Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=3, stride=1, padding=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=4*channels, out_channels=1, kernel_size=1, stride=1, device=device),
        )
        self.size = Sequential(
            Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=1, stride=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=4*channels, out_channels=2, kernel_size=1, stride=1, device=device)
        )
        self.cls = Sequential(
            Conv2d(in_channels=2*channels, out_channels=num_classes*2, kernel_size=1, stride=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=num_classes*2, out_channels=num_classes, kernel_size=1, stride=1, device=device)
        )
    def forward(self, color: Tensor, feature: Tensor) -> Tensor:
        x = cat([color, feature], dim=1)
        score: Tensor = self.score(x)
        size: Tensor = self.size(x)
        size = (size.exp()+1).exp()
        cls: Tensor = self.cls(x)
        return cat([score, size, cls], dim=1)