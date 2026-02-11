from torch.nn import Module, Conv2d, ReLU, MaxPool2d, Linear
from torch import Tensor, flatten

class Model(Module):
    def __init__(self, width: int, height: int) -> None:
        super().__init__()

        self.cnn = Conv2d(in_channels=3, out_channels=16,
                          kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2)
        flattened_size = 16 * (height // 2) * (width // 2)

        self.linear = Linear(flattened_size, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.cnn(x)
        x = self.relu(x)
        x = self.pool(x)

        x = flatten(x, start_dim=1) 
        x = self.linear(x)

        return x
