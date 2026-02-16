from torch import Tensor, from_numpy, tensor
from ._dataset_ import Dataset
from pycocotools.coco import COCO
from cv2 import imread
from ._funcs_ import ImgRead
class Coco(Dataset):
    def __init__(self, path:str):
        self.trainDir = path+"/train2017/"
        self.testDir = path+"/test2017/"
        self.train = COCO(path+"/annotations/instances_train2017.json")
        self.test = COCO(path+"/annotations/image_info_test2017.json")
        self.class_id = self.train.getCatIds()
    def getTrainSize(self) -> int:
        return len(self.train.getImgIds())
    def getTestSize(self) -> int:
        return len(self.test.getImgIds())
    def getTestImage(self, index: int) -> Tensor:
        id = self.test.getImgIds()[index]
        path = self.testDir + self.test.loadImgs(id)[0]["file_name"]
        x = from_numpy(ImgRead(path)).float()/255
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        return x
    def getTrainImage(self, index: int) -> Tensor:
        id = self.train.getImgIds()[index]
        path = self.trainDir + self.train.loadImgs(id)[0]["file_name"]
        x = from_numpy(ImgRead(path)).float()/255
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)
        return x
    def getImgTrainInfo(self, index:int) -> Tensor:
        id = self.train.getImgIds()[index]
        ann_id = self.train.getAnnIds(imgIds=id)[0]
        anns = self.train.loadAnns(ann_id)
        x = tensor([(ann["bbox"] + [ann["category_id"]]) for ann in anns])
        return x
    def getClassSize(self) -> int:
        return len(self.class_id)
    def getClass(self, index: int) -> str:
        return self.train.loadCats(index)[0]["name"]