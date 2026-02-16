from torch.nn import Linear, Module
from torch import Tensor, softmax

class Model(Module):
    def __init__(self, channels: int, num_classes: int):
        super().__init__()
        self.linear = Linear(in_features=channels, out_features=num_classes)
    def forward(self, x: Tensor) -> Tensor:
        print(x.shape)
        x = x.mean(dim=(-2,-1))
        x = softmax(input=self.linear(x), dim=1)
        return x
        