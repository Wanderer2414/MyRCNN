from torch import Tensor, from_numpy, tensor, cat, stack
from ._dataset_ import Dataset
from cv2 import imread
from ._funcs_ import ImgRead
from .dataset2 import YOLODataset
from . import config
class Coco(Dataset):
    def __init__(self, path:str):
        self.path = path
        
        self.train = YOLODataset("../COCO/train.csv", "../COCO/images/", "../COCO/labels/", config.ANCHORS)
        self.test = YOLODataset("../COCO/test.csv", "../COCO/images/", "../COCO/labels/", config.ANCHORS)
    def getTrainSize(self) -> int:
        return len(self.train)
    def getTestSize(self) -> int:
        return len(self.test)
    def getTestTensor(self, index: int) -> Tensor:
        return tensor(self.test[index][0]).permute(2, 0, 1).unsqueeze(0)/255
    def getTrainTensor(self, index: int) -> Tensor:
        return tensor(self.train[index][0]).permute(2, 0, 1).unsqueeze(0)/255
    def getTrainLabel(self, index:int) -> Tensor:
        ar = tensor(self.train[index][0])
        boxes = tensor(self.train[index][1])
        H, W = ar.shape[:2]
        w = boxes[:, 2]*W
        h = boxes[:, 3]*H
        x = boxes[:, 0]*W - w/2
        y = boxes[:, 1]*H - h/2
        cls = boxes[:, -1]
        return stack([x,y,x+w,y+h, cls], dim=0).permute(1, 0)
    def getTestLabel(self, index:int) -> Tensor:
        ar = tensor(self.test[index][0])
        boxes = tensor(self.test[index][1])
        H, W = ar.shape[:2]
        w = boxes[:, 2]*W
        h = boxes[:, 3]*H
        x = boxes[:, 0]*W - w/2
        y = boxes[:, 1]*H - h/2
        cls = boxes[:, -1]
        return stack([x,y,x+w,y+h, cls], dim=0).permute(1, 0)
    def getClassSize(self) -> int:
        return len(config.COCO_LABELS)
    def getClass(self, index: int) -> str:
        return config.COCO_LABELS[index]