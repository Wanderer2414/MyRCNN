from torch import Tensor, tensor
class Dataset:
    def __init__(self):
        pass
    def getTrainSize(self) -> int:
        return 0
    def getTestSize(self) -> int:
        return 0
    def getTrainImage(self, index: int) -> Tensor:
        return tensor([[]])
    def getTestImage(self, index: int) -> Tensor:
        return tensor([[]])
    def getClassSize(self) -> int:
        return 0
    def getClass(self, index: int) -> str:
        return ""
    def getImgTrainInfo(self, index: int)-> Tensor:
        return tensor([])
    def getImgTestInfo(self, index: int) -> Tensor:
        return tensor([])