from torch.nn import Module, Conv2d, LeakyReLU,Sequential
from torch import device, Tensor, cat
from torchvision.ops import roi_align
class Classification(Module):
    # 300 x 300 input
    def __init__(self, boundary_channels: int, color_channels: int, num_classes: int, device: device = device("cpu")):
        super().__init__()
        self.boundary_score = Sequential(
            Conv2d(in_channels=boundary_channels, out_channels=32, kernel_size=3, stride=3, device=device), # 100x100
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 100x100
            Conv2d(in_channels=8, out_channels=32, kernel_size=2, stride=2, device=device), # 50x50
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 50x50
            Conv2d(in_channels=8, out_channels=32, kernel_size=5, stride=5, device=device), # 10x10
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=8, kernel_size=1, device=device), # 10x10
            Conv2d(in_channels=8, out_channels=32, kernel_size=2, stride=2, device=device), # 5x5
            LeakyReLU(),
            Conv2d(in_channels=32, out_channels=16, kernel_size=1, device=device), # 5x5
            Conv2d(in_channels=16, out_channels=num_classes, kernel_size=5, device=device)
        )
        # self.color_score = Sequential(
        #     Conv2d(in_channels=color_channels, out_channels=color_channels*2, kernel_size=1, device=device),
        #     LeakyReLU(),
        #     Conv2d(in_channels=color_channels*2, out_channels=4, kernel_size=1, device=device),
        #     Conv2d(in_channels=4, out_channels=8, kernel_size=9, stride=9, device=device), # 900x900
        #     LeakyReLU(),
        #     Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 100x100
        #     Conv2d(in_channels=4, out_channels=8, kernel_size=4, stride=4, device=device), # 25x25
        #     LeakyReLU(),
        #     Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 25x25
        #     Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=5, device=device), # 5x5
        #     LeakyReLU(),
        #     Conv2d(in_channels=8, out_channels=4, kernel_size=1, device=device), # 5x5 
        # )
        # self.cls = Sequential(
        #     Conv2d(in_channels=20, out_channels=30, kernel_size=1, device=device), # 5x5
        #     LeakyReLU(),
        #     Conv2d(in_channels=30, out_channels=num_classes, kernel_size=5, device=device), # 5x5
        # )
        
    def forward(self, mask: Tensor, color: Tensor, boxes: Tensor) -> Tensor:
        """_summary_
        Args:
            mask (Tensor): [B, C, H, W]
            color (Tensor): [B, C, H, W]
            boxes (Tensor): [N, 4] [[x1, y1, x2, y2]]

        Returns:
            [N, nums] [score_1, ..., score_n]
        """
        output_size = (300, 300)
        maskes: Tensor = roi_align(mask, [boxes], output_size)
        colors: Tensor = roi_align(color, [boxes], output_size)
        
        cls = self.boundary_score(maskes)
        # colors = self.color_score(colors)
        # combine = cat([mask, color], dim=1)
        # cls = self.cls(combine)
        return cls.squeeze(-2).squeeze(-1)