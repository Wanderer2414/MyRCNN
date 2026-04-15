from torch.nn import Module, Conv2d, LeakyReLU,Sequential, BatchNorm2d, MaxPool2d, AvgPool2d
from torch import device, Tensor, cat
from torchvision.ops import roi_align
from Base import Merger, Repeat
class Classification(Module):
    # 300 x 300 input
    def __init__(self, mask_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.cls = Sequential(
            Merger((40, 3), 
                   Sequential(
                        Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 400x400
                        Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 200x200
                        Conv2d(in_channels=20, out_channels=37, kernel_size=1, bias=False, device=device), # 200x200
                        BatchNorm2d(37, device=device),
                        LeakyReLU(inplace=True),
                    ),
                   Sequential(
                        AvgPool2d(kernel_size=2, stride=2),
                        Repeat(2)
                    )),
            Merger((40, 3), 
                   Sequential(
                        Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device),
                        Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 100x100
                        Conv2d(in_channels=20, out_channels=37, kernel_size=1, bias=False, device=device),
                        BatchNorm2d(37, device=device),
                        LeakyReLU(inplace=True),
                    ),
                   Sequential(
                        AvgPool2d(kernel_size=2, stride=2),
                        Repeat(2)
                    )),
            Merger((40, 3), 
                   Sequential(
                        Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 100x100
                        Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 50x50
                        Conv2d(in_channels=20, out_channels=37, kernel_size=1, bias=False, device=device),
                        BatchNorm2d(37, device=device),
                        LeakyReLU(inplace=True),
                    ),
                   Sequential(
                        AvgPool2d(kernel_size=2, stride=2),
                        Repeat(2)
                    )),
            Merger((40, 3), 
                   Sequential(
                        Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device), # 100x100
                        Conv2d(in_channels=20, out_channels=20, kernel_size=2, stride=2, bias=False, groups=20, device=device), # 25x25
                        Conv2d(in_channels=20, out_channels=37, kernel_size=1, bias=False, device=device),
                        BatchNorm2d(37, device=device),
                        LeakyReLU(inplace=True),
                    ),
                    AvgPool2d(kernel_size=2, stride=2),
                    ),
            Conv2d(in_channels=40, out_channels=20, kernel_size=1, bias=False, device=device),
            Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=5, bias=False, groups=20, device=device), # 5x5
            Conv2d(in_channels=20, out_channels=40, kernel_size=1, bias=False, device=device),
            LeakyReLU(),
            Conv2d(in_channels=40, out_channels=num_classes, kernel_size=5, device=device), 
        )
        self.channels = mask_channels
        
    def forward(self, boundary: Tensor, mask:Tensor,  color: Tensor, boxes: Tensor) -> Tensor:
        """_summary_
        Args:
            boundary (Tensor): [B, C, H, W]
            mask (Tensor): [B, C', H, W]
            color (Tensor): [B, C, H, W]
            boxes (Tensor): [N, 4] [[x1, y1, x2, y2]]

        Returns:
            [N, nums] [score_1, ..., score_n]
        """
        output_size = (400, 400)
        boundaries = boundary*mask
        B, C, H, W = color.shape
        color = color*mask
        boundaries: Tensor = roi_align(boundary, boxes, output_size) # type: ignore[assignment]
        colors: Tensor = roi_align(color, boxes, output_size) # type: ignore[assignment]
        mix = cat([boundaries.repeat(1, 40, 1, 1), colors], dim=1)
        cls: Tensor = self.cls(mix)
        # cls = cat([cls, boxes])
        return cls.squeeze(-1).squeeze(-1)