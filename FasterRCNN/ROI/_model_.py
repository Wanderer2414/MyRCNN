from torch.nn import Module, Linear
from torch import Tensor, ceil, floor, tensor, softmax, cat, arange
from numpy import int8
from torchvision.ops import roi_align
class Model(Module):
    def __init__(self, channels: int, num_class:int, device):
        super().__init__()
        self.linear = Linear(in_features=channels, out_features=num_class, device=device)
    def forward(self, x: Tensor, score: Tensor) -> Tensor:
        boxes = score[:, :, 2:6]
        boxes[:, :, 2:4] += boxes[:, :, 0:2]
        boxes[:, :, 0:2] = floor(boxes[:, :, 0:2])
        boxes[:, :, 2:4] = ceil(boxes[:, :, 2:4])
    

        B, C, H, W = x.shape
        device = x.device

        batch_indices = arange(B, device=device).view(-1, 1)
        batch_indices = batch_indices.expand(-1, boxes.size(1))  # (B, N)

        boxes = boxes.reshape(-1, 4)              # (B*N, 4)
        batch_indices = batch_indices.reshape(-1) # (B*N)

        rois = cat([batch_indices.unsqueeze(1).float(), boxes], dim=1)

        pooled = roi_align(x, rois, output_size=1)  # (B*N, C, 1, 1)

        proj = pooled.squeeze(-1).squeeze(-1)  # (B*N, C)

        linr = self.linear(proj)              # (B*N, num_classes)
        prd = softmax(linr, dim=-1)

        out = prd.view(B, -1, prd.size(-1))   # (B, N, num_
        return out