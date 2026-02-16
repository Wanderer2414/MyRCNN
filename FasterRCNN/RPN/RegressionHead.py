from torch.nn import Module, Conv2d
from torch import Tensor, softmax

class RegressionHead(Module):
    def __init__(self, channels: int, device):
        super().__init__()
        self.conv = Conv2d(in_channels=channels, out_channels=36, kernel_size=1, stride=1, device=device)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(x.shape[0], x.shape[1], x.shape[2], 9, 4)
        x = softmax(x, dim=4)
        return x