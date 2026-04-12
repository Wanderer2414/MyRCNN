import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from utils import seed_everything
CHOSEN_DATASET = "COCO" # or "PASCAL_VOC"
DATASET = "../COCO" if CHOSEN_DATASET == "COCO" else "PASCAL_VOC"
# DATASET = "/kaggle/input/datasets/the0bserver/mscoco/COCO" if CHOSEN_DATASET == "COCO" else "/kaggle/input/datasets/the0bserver/pascal/PASCAL_VOC"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# seed_everything()  # If you want deterministic behavior
NUM_WORKERS = 4
BATCH_SIZE = 1
IMAGE_SIZE = 320
NUM_CLASSES = 80 if CHOSEN_DATASET == "COCO" else 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
NUM_EPOCHS = 100
CONF_THRESHOLD = 0.05
MAP_IOU_THRESH = 0.5
NMS_IOU_THRESH = 0.45
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16]
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_FILE = "checkpoint.pth.tar"
IMG_DIR = DATASET + "/images/"
LABEL_DIR = DATASET + "/labels/"
TRAIN_DIR = DATASET + "/train.csv"
TEST_DIR = DATASET + "/test.csv"
CONFIG_PATH = "MOLOv2v3_coco.cfg" if CHOSEN_DATASET == "COCO" else "MOLOv2v3_pascal.cfg"


# # COCO ANCHORS 
ANCHORS = [
    # Scale 1 (from mask = 3,4,5): Detects large objects on the coarse grid
    [(115/320, 74/320), (119/320, 199/320), (243/320, 238/320)], 
    
    # Scale 2 (from mask = 0,1,2): Detects small objects on the fine grid
    [(12/320, 18/320), (37/320, 49/320), (52/320, 132/320)]     
] if CHOSEN_DATASET == "COCO" else [
    # Scale 1 (from mask = 3,4,5): Detects large objects on the coarse grid
    [(189/320, 126/320), (137/320, 236/320), (265/320, 259/320)],     

    # Scale 2 (from mask = 0,1,2): Detects small objects on the fine grid
    [(26/320, 48/320), (67/320, 84/320), (72/320, 175/320)]
]

# PASCAL VOC ANCHORS 
# ANCHORS = [
#     # Scale 1 (from mask = 3,4,5): Detects large objects on the coarse grid
#     [(189/320, 126/320), (137/320, 236/320), (265/320, 259/320)],     

#     # Scale 2 (from mask = 0,1,2): Detects small objects on the fine grid
#     [(26/320, 48/320), (67/320, 84/320), (72/320, 175/320)]
# ]
NUM_ANCHORS = len(ANCHORS[0])  # Number of anchors per scale
NUM_SCALE = 2
scale = 1.1

train_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)),
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv2.BORDER_CONSTANT,
            value=0, # Added missing value parameter
        ),
        A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=0.4),
        A.OneOf(
            [
                A.ShiftScaleRotate(
                    rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0 # Added missing value parameter
                ),
                A.Affine(shear=15, p=0.5, mode=cv2.BORDER_CONSTANT, cval=0), # Updated from deprecated IAAAffine
            ],
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Blur(p=0.1),
        A.CLAHE(p=0.1),
        A.Posterize(p=0.1),
        A.ToGray(p=0.1),
        A.ChannelShuffle(p=0.05),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],clip=True),
)

test_transforms = A.Compose(
    [
        A.LongestMaxSize(max_size=IMAGE_SIZE),
        A.PadIfNeeded(
            min_height=IMAGE_SIZE, 
            min_width=IMAGE_SIZE, 
            border_mode=cv2.BORDER_CONSTANT,
            value=0 # Added missing value parameter
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
        ToTensorV2(),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],clip=True),
)

PASCAL_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

COCO_LABELS = ['person',
 'bicycle',
 'car',
 'motorcycle',
 'airplane',
 'bus',
 'train',
 'truck',
 'boat',
 'traffic light',
 'fire hydrant',
 'stop sign',
 'parking meter',
 'bench',
 'bird',
 'cat',
 'dog',
 'horse',
 'sheep',
 'cow',
 'elephant',
 'bear',
 'zebra',
 'giraffe',
 'backpack',
 'umbrella',
 'handbag',
 'tie',
 'suitcase',
 'frisbee',
 'skis',
 'snowboard',
 'sports ball',
 'kite',
 'baseball bat',
 'baseball glove',
 'skateboard',
 'surfboard',
 'tennis racket',
 'bottle',
 'wine glass',
 'cup',
 'fork',
 'knife',
 'spoon',
 'bowl',
 'banana',
 'apple',
 'sandwich',
 'orange',
 'broccoli',
 'carrot',
 'hot dog',
 'pizza',
 'donut',
 'cake',
 'chair',
 'couch',
 'potted plant',
 'bed',
 'dining table',
 'toilet',
 'tv',
 'laptop',
 'mouse',
 'remote',
 'keyboard',
 'cell phone',
 'microwave',
 'oven',
 'toaster',
 'sink',
 'refrigerator',
 'book',
 'clock',
 'vase',
 'scissors',
 'teddy bear',
 'hair drier',
 'toothbrush'
]