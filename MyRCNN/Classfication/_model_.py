from torch.nn import Module, Conv2d, Sequential, Linear, Softmax
from torch import Tensor,device

class Classification(Module):
    def __init__(self, channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.downward = Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=5, padding=2, device=device)
        self.cls1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=5, padding=2, device=device)
        self.cls2 = Sequential(
            Linear(in_features=channels, out_features=num_classes, device=device),
            Softmax(dim=-1)
        )
        
    def forward(self, x: Tensor) -> list[Tensor]:
        out: list[Tensor] = []
        while (min(x.shape[-2], x.shape[-1])>5):
            score: Tensor = self.cls1(x)
            score = score.permute(0, 2, 3, 1)
            score = self.cls2(score)
            out.append(score)
            x = self.downward(x)
        return out