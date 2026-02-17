from torch import Tensor, tensor
class Dataset:
    def __init__(self):
        pass
    def getTrainSize(self) -> int:
        return 0
    def getTestSize(self) -> int:
        return 0
    def getTrainTensor(self, index: int) -> Tensor:
        return tensor([[]])
    def getTestTensor(self, index: int) -> Tensor:
        return tensor([[]])
    def getClassSize(self) -> int:
        return 0
    def getClass(self, index: int) -> str:
        return ""
    def getTrainLabel(self, index: int)-> Tensor:
        return tensor([])
    def getTestLabel(self, index: int) -> Tensor:
        return tensor([])