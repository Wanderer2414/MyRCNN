from torch.nn import Module, Conv2d, Sequential, ReLU,MaxPool2d, AvgPool2d, Linear
from torch import Tensor, where, zeros_like, ones_like, device

class ExtractFeature(Module):
    def __init__(self, channels: int, device: device = device("cpu")):
        super().__init__()
        self.conv = Conv2d(in_channels=1, out_channels=channels, kernel_size=5, stride=1, padding=2, device=device)
        self.avg = AvgPool2d(3,1,1)
        self.net1 = Sequential(
            Conv2d(in_channels=1,out_channels=channels,kernel_size=5,stride=1, padding=2, device=device),
            Conv2d(in_channels=channels,out_channels=channels,kernel_size=1,stride=1, device=device),
            ReLU(),
        )
        self.linr = Linear(in_features=channels, out_features=channels, device=device)
        self.net2 = Conv2d(in_channels=channels,out_channels=channels,kernel_size=5,stride=1, padding=2, device=device)
        
    def forward(self, x: Tensor):
        x = self.conv(x)
        squ = x * x
        sum = 9*self.avg(x)
        sum2 = 9*self.avg(squ)
        x = 9 * squ + sum2 - 2*sum*x
        x = 1 - x/x.max()
        alpha = x.mean()
        x = x.max(dim=1, keepdim=True).values
        x = where(x < alpha, zeros_like(x), ones_like(x))
        x = self.net1(x)
        B, C, W, H = x.shape
        x = x.view(B, -1, 64)
        x = self.linr(x)
        x = x.view(B, C, W, H)
        x = self.net2(x)
        return x
        

class FeatureHead(Module):
    def __init__(self, out_channels: int, device: device = device("cpu")):
        super().__init__()
        self.pre1 = Conv2d(in_channels=1, out_channels=1, kernel_size=1, device=device)
        self.pre2 = Conv2d(in_channels=1, out_channels=1, kernel_size=1, device=device)
        self.net = Sequential(
            Conv2d(in_channels=3, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device)
        )
        self.snet = Sequential(
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            ReLU(),
            Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, device=device),
            MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        self.ext = ExtractFeature(channels=out_channels,device=device)
        
        
    def forward(self, mask: Tensor, color: Tensor, x:Tensor) -> Tensor:
        feature: Tensor = self.pre1(color) + self.pre2(1-mask) + self.net(x)
        feature = self.snet(feature/feature.max())
        feature = feature/feature.max()
        alpha = feature.mean()
        feature = where(feature < alpha, zeros_like(feature), ones_like(feature))
        x = self.ext(feature)
        return x