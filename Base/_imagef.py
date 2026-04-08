from torch.nn import Module
from torch.nn.functional import interpolate
from torch import Tensor
from typing import Sequence
class Interpolate(Module):
    def __init__(self, 
                 size: int|Sequence[int] | None = None, 
                 scale_factor: float | Sequence[float] | None = None,
                 mode: str = "nearest",
                 align_corners: bool | None = None,
                 recompute_scale_factor: bool | None = None,
                 antialias: bool = False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
    def forward(self, x:Tensor) -> Tensor:
        return interpolate(x, 
                           self.size, 
                           self.scale_factor, 
                           self.mode, 
                           self.align_corners, 
                           self.recompute_scale_factor, 
                           self.antialias)
class UnSize(Module):
    def __init__(self, module: Module, mode: str = "nearest", keepold: bool = False):
        super().__init__()
        self.module = module
        self.mode = mode
        self.keepold = keepold
    def forward(self, x):
        old = x
        B, C, H, W = x.shape
        x = self.module(x)
        if (self.keepold):
            return old, interpolate(x, size=(H, W), mode = self.mode)
        return interpolate(x, size=(H, W), mode = self.mode)