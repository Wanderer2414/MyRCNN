from torch import Tensor, from_numpy, tensor, cat
from ._dataset_ import Dataset
from pycocotools.coco import COCO as _COCO
from cv2 import imread
from ._funcs_ import ImgRead
class Coco(Dataset):
    def __init__(self, path:str):
        self.trainDir = path+"/train2017/"
        self.testDir = path+"/test2017/"
        self.train = _COCO(path+"/annotations/instances_train2017.json")
        self.test = _COCO(path+"/annotations/image_info_test2017.json")
        self.class_id = self.train.getCatIds()
    def getTrainSize(self) -> int:
        return len(self.train.getImgIds())
    def getTestSize(self) -> int:
        return len(self.test.getImgIds())
    def getTestTensor(self, index: int) -> Tensor:
        id = self.test.getImgIds()[index]
        path = self.testDir + self.test.loadImgs(id)[0]["file_name"]
        x = from_numpy(ImgRead(path)).float()/255
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        return x
    def getTrainTensor(self, index: int) -> Tensor:
        id = self.train.getImgIds()[index]
        path = self.trainDir + self.train.loadImgs(id)[0]["file_name"]
        x = from_numpy(ImgRead(path)).float()/255
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        return x
    def getTrainLabel(self, index:int) -> Tensor:
        id = self.train.getImgIds()[index]
        ann_ids = self.train.getAnnIds(imgIds=id)
        if (len(ann_ids)==0):
          return tensor([])
        ann_id = ann_ids[0]
        anns = self.train.loadAnns(ann_id)
        bbox = tensor([ann["bbox"] for ann in anns]).view(-1, 4)
        bbox[:, 2:4] += bbox[:, 0:2]
        catg = tensor([ann["category_id"] for ann in anns]).view(-1, 1)
        return cat([bbox, catg], dim=-1)
    def getClassSize(self) -> int:
        return len(self.class_id)
    def getClass(self, index: int) -> str:
        return self.train.loadCats(index)[0]["name"]