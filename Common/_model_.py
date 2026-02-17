from dataset import Dataset
from torch import Tensor
from typing import Callable
class Model:
    def __init__(self, imgs: list[str]):
        self.images = imgs
    def train(self, x: Dataset, loss:Callable[[Tensor, Tensor], Tensor]):
        pass
    def inference(self, x: Tensor):
        pass