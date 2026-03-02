from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, LeakyReLU
from torch import Tensor, where, zeros_like, ones_like, device, cat, zeros
from torch.nn.functional import max_pool2d, avg_pool2d, interpolate

class FeatureHead(Module):
    def __init__(self, out_channels: int, device: device = device("cpu")):
        super().__init__()
        self.net = Sequential(
            Conv2d(in_channels=1, out_channels=2*out_channels, kernel_size=5, stride=1, padding=2, device=device),
            LeakyReLU(),
            Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=1, device=device),
        )
        self.down = Sequential(
            Conv2d(in_channels=out_channels, out_channels=2*out_channels, kernel_size=5, stride=5, padding=2, device=device),
            LeakyReLU(),
            Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=1, device=device),
        )
        self.ft = Sequential(
            Conv2d(in_channels=out_channels*2, out_channels=4*out_channels, kernel_size=1, device=device),
            LeakyReLU(),
            Conv2d(in_channels=4*out_channels, out_channels=out_channels, kernel_size=1, device=device)
        )
        self.out_channels = out_channels
        
    def forward(self, mask: Tensor, color: Tensor) -> Tensor:
        B, C, H, W = mask.shape
        mask = self.net(max_pool2d(1-mask, 3, 1, 1))
        score = zeros((B, self.out_channels, H, W),device=mask.device)
        size = 5 
        while (min(mask.shape[-2], mask.shape[-1])>=size):
            mask = self.down(mask)
            score = score + interpolate(mask, size=(H, W), mode="nearest")
            size *= 5
        score = cat([score, color], dim=1)
        score = self.ft(score)
        return score