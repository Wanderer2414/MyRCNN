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