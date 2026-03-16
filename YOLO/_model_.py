import torch
import torch.nn as nn
from torch import Tensor,device, save
from torch.optim import Adam
from torch.nn.functional import interpolate
from dataset import Dataset
from time import time
from display import show_progress_counter
    
B = 2
C = 100
S = 7
class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.depth = B * 5 + C

        layers = [
            # Probe(0, forward=lambda x: print('#' * 5 + ' Start ' + '#' * 5)),
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),                   # Conv 1
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv1', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),                           # Conv 2
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv2', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 128, kernel_size=1),                                     # Conv 3
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 256, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv3', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(4):                                                          # Conv 4
            layers += [
                nn.Conv2d(512, 256, kernel_size=1),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(512, 512, kernel_size=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv4', forward=probe_dist),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ]

        for i in range(2):                                                          # Conv 5
            layers += [
                nn.Conv2d(1024, 512, kernel_size=1),
                nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        layers += [
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('conv5', forward=probe_dist),
        ]

        for _ in range(2):                                                          # Conv 6
            layers += [
                nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
                nn.LeakyReLU(negative_slope=0.1)
            ]
        # layers.append(Probe('conv6', forward=probe_dist))

        layers += [
            nn.Flatten(),
            nn.Linear(S * S * 1024, 4096),                            # Linear 1
            nn.Dropout(),
            nn.LeakyReLU(negative_slope=0.1),
            # Probe('linear1', forward=probe_dist),
            nn.Linear(4096, S * S * self.depth),                      # Linear 2
            # Probe('linear2', forward=probe_dist),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return torch.reshape(
            self.model.forward(x),
            (x.size(dim=0), S, S, self.depth)
        )

class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self, pred: Tensor, target: Tensor):
        # reshape predictions
        pred = pred.view(-1, self.S, self.S, self.C + self.B*5)

        # class probabilities
        pred_cls = pred[..., :self.C]
        target_cls = target[..., :self.C]

        # bbox predictions
        pred_boxes = pred[..., self.C:]
        target_boxes = target[..., self.C:]

        obj_mask = target[..., self.C].unsqueeze(-1)
        noobj_mask = 1 - obj_mask
        
        # ---- CLASS LOSS ----
        class_loss = self.mse(
            obj_mask * pred_cls,
            obj_mask * target_cls
        )

        # ---- BOX LOSS ----
        box_loss = 0
        obj_loss = 0
        noobj_loss = 0

        for b in range(self.B):

            pred_box = pred_boxes[..., b*5:(b+1)*5]
            target_box = target_boxes[..., b*5:(b+1)*5]

            # coordinates
            box_loss += self.mse(
                obj_mask * pred_box[..., :2],
                obj_mask * target_box[..., :2]
            )

            # width height (sqrt)
            box_loss += self.mse(
                obj_mask * torch.sqrt(torch.abs(pred_box[..., 2:4] + 1e-6)),
                obj_mask * torch.sqrt(target_box[..., 2:4])
            )

            # object confidence
            obj_loss += self.mse(
                obj_mask * pred_box[..., 4],
                obj_mask * target_box[..., 4]
            )

            # no object confidence
            noobj_loss += self.mse(
                noobj_mask * pred_box[..., 4],
                noobj_mask * target_box[..., 4]
            )

        total_loss = (
            self.lambda_coord * box_loss
            + obj_loss
            + self.lambda_noobj * noobj_loss
            + class_loss
        )

        return total_loss

def create_yolo_target(box: Tensor, img_w, img_h):
    box = box.squeeze()
    target = torch.zeros((S, S, C + B*5), dtype=torch.float, device=box.device)

    x1,y1,x2,y2,cls = box

    x = ((x1+x2)/2)/img_w
    y = ((y1+y2)/2)/img_h
    w = (x2-x1)/img_w
    h = (y2-y1)/img_h

    i = int(x*S)
    j = int(y*S)

    target[i,j,cls.long()] = 1
    target[i,j,C:C+5] = torch.tensor([x,y,w,h,1])

    return target
class Model:
    def __init__(self, device: device = device("cpu")):
        self.model = YOLOv1().to(device)
        self.opt = Adam(self.model.parameters(), lr=1e-4)
        self.device = device
    def train(self, x: Dataset):
        loss = YOLOLoss()
        size = x.getTrainSize()
        start = time()
        for epoch in range(100):
            sloss = 0
            for i in range(10):
                tens:Tensor = x.getTrainTensor(i).to(self.device)
                h,w = tens.shape[-2:]
                label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
                if (label.shape[1] == 0):
                  continue
                tens = interpolate(tens, size=(448, 448), mode="nearest")
                label =create_yolo_target(label, w, h)
                out: list[Tensor] = self.model(tens)
                lss = loss(out,label)
                self.opt.zero_grad()
                lss.backward()
                self.opt.step()
                # show_progress_counter(i+1, size, start, f"Loss: {lss}")
                sloss += lss
                if ((i+1) % (size//5) == 0):
                    print(f"Saved: {(i+1)} / {size//5} progress")
                    save(self.model.state_dict(), "model.pth")
            show_progress_counter(epoch+1, 100, start, f"{sloss/10}")
            save(self.model.state_dict(), "model.pth")