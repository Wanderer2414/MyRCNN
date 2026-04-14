from torch.nn import Module, Conv2d, LeakyReLU,Sequential, BatchNorm2d, MaxPool2d
from torch import device, Tensor, cat
from torchvision.ops import roi_align
class Classification(Module):
    # 300 x 300 input
    def __init__(self, boundary_channels: int, color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.prepare_boundary = Sequential(
            Conv2d(in_channels=boundary_channels, out_channels=boundary_channels, kernel_size=5, stride=1, padding=2, groups=boundary_channels, device=device),
            MaxPool2d(kernel_size=5, stride=1, padding=2),
            BatchNorm2d(boundary_channels, device=device),
            LeakyReLU(),
            Conv2d(in_channels=boundary_channels, out_channels=32, kernel_size=1, bias=False, device=device),
        )
        self.prepare_color = Conv2d(in_channels=color_channels, out_channels=8, kernel_size=1, bias=False, device=device)
        self.channels = boundary_channels + color_channels
        self.cls = Sequential(
            Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 400x400
            Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 200x200
            Conv2d(in_channels=20, out_channels=40, kernel_size=1, bias=False, device=device), # 200x200
            BatchNorm2d(40, device=device),
            LeakyReLU(),
            Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device),
            Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 100x100
            Conv2d(in_channels=20, out_channels=40, kernel_size=1, bias=False, device=device),
            BatchNorm2d(40, device=device),
            LeakyReLU(),
            Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 100x100
            Conv2d(in_channels=20, out_channels=20, kernel_size=4, stride=4, bias=False, groups=20, device=device), # 25x25
            Conv2d(in_channels=20, out_channels=40, kernel_size=1, bias=False, device=device),
            BatchNorm2d(40, device=device),
            LeakyReLU(),
            Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 20x20
            Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=5, bias=False, groups=20, device=device), # 5x5
            Conv2d(in_channels=20, out_channels=40, kernel_size=1, bias=False, device=device),
            BatchNorm2d(40, device=device),
            LeakyReLU(),
            Conv2d(in_channels=40, out_channels=40, kernel_size=1, bias=False, device=device), # 5x5
            Conv2d(in_channels=40, out_channels=num_classes, kernel_size=5, device=device), 
        )
        
    def forward(self, mask: Tensor, color: Tensor, boxes: Tensor) -> Tensor:
        """_summary_
        Args:
            mask (Tensor): [B, C, H, W]
            color (Tensor): [B, C, H, W]
            boxes (Tensor): [N, 4] [[x1, y1, x2, y2]]

        Returns:
            [N, nums] [score_1, ..., score_n]
        """
        output_size = (400, 400)
        maskes: Tensor = roi_align(mask, [boxes], output_size)
        colors: Tensor = roi_align(color, [boxes], output_size)
        
        mix = cat([self.prepare_boundary(maskes), self.prepare_color(colors)], dim=1)
        cls: Tensor = self.cls(mix)
        return cls.squeeze(-1).squeeze(-1)