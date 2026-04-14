from torch import Tensor, from_numpy, tensor, cat, stack
from cv2 import imread
from ._funcs_ import ImgRead
from dataset2 import YOLODataset
from torch.utils.data import Dataset
import config
class Coco(Dataset[Tensor]):
    def __init__(self, path:str, train_path_csv: str):
        self.path = path
        self.train = YOLODataset(train_path_csv, path + "/images/", path + "/labels/", config.ANCHORS)
    def __len__(self) -> int:
        return len(self.train)
    def __getitem__(self, index) -> Tensor:
        return tensor(self.train[index][0]).permute(2, 0, 1).unsqueeze(0)/255
    def getTrainTensor(self, index: int) -> tuple[Tensor, Tensor]:
        ar = tensor(self.train[index][0])
        boxes = tensor(self.train[index][1])
        H, W = ar.shape[:2]
        w = boxes[:, 2]*W
        h = boxes[:, 3]*H
        x = boxes[:, 0]*W - w/2
        y = boxes[:, 1]*H - h/2
        cls = boxes[:, -1]
        label =  stack([x,y,x+w,y+h, cls], dim=0).permute(1, 0)
        return tensor(self.train[index][0]).permute(2, 0, 1).unsqueeze(0)/255, label
