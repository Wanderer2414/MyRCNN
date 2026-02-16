from torch import Tensor, from_numpy
from cv2.typing import MatLike
from cv2 import imread, imwrite, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from numpy import uint8
def ImgToTensor(img: MatLike) -> Tensor:
    x = from_numpy(img).float()/255
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    return x
def TensorToImg(tensor: Tensor) -> MatLike:
    tensor = tensor*255
    img = tensor[0].permute(1,2,0).numpy().astype(uint8)
    return img
def ImgRead(src: str)->MatLike:
    img = imread(src)
    if (img is None):
        raise
    img = cvtColor(img, COLOR_BGR2RGB)
    return img
def ImgWrite(src: str, img: MatLike) -> None:
    img = cvtColor(img, COLOR_RGB2BGR)
    imwrite(src, img)