import ResNet50
from torch.nn import Module, Sequential
from torch.nn.functional import smooth_l1_loss, cross_entropy
from torch import Tensor, device, save, cat, tensor, zeros, long
from torch.optim.adam import Adam

from dataset._dataset_ import Dataset
from . import RPN, ROI
from Common import Model as CModel
from typing import Callable
from display import show_progress_counter
from time import time
from torchvision.ops import nms, box_iou
def faster_rcnn_loss(pred:Tensor, gt:Tensor, iou_thresh=0.5)->Tensor:
    B, N, _ = pred.shape
    obj_logits = pred[:, :, 0:2]
    box_pred = pred[:, :, 2:6]
    class_logits = pred[:, :, 6:]

    total_loss:Tensor = tensor([0])

    for b in range(B):

        pred_boxes = box_pred[b]
        gt_boxes = gt[b][:, :4]
        gt_classes = gt[b][:, 4].long()

        valid_gt_mask = gt_classes >= 0
        gt_boxes = gt_boxes[valid_gt_mask]
        gt_classes = gt_classes[valid_gt_mask]

        if gt_boxes.numel() == 0:
            continue

        ious = box_iou(pred_boxes, gt_boxes)
        max_iou, matched_idx = ious.max(dim=1)

        labels = zeros(N, dtype=long, device=pred.device)
        pos_mask = max_iou >= iou_thresh
        labels[pos_mask] = 1

        # Objectness
        obj_loss = cross_entropy(obj_logits[b], labels)

        loss = obj_loss

        if pos_mask.sum() > 0:

            matched_gt_boxes = gt_boxes[matched_idx[pos_mask]]
            matched_gt_classes = gt_classes[matched_idx[pos_mask]]

            box_loss = smooth_l1_loss(
                pred_boxes[pos_mask],
                matched_gt_boxes
            )

            cls_loss = cross_entropy(
                class_logits[b][pos_mask],
                matched_gt_classes
            )

            loss = loss + box_loss + cls_loss

        total_loss = total_loss + loss

    return total_loss

class FasterRCNN(Module):
    def __init__(self, device):
        super().__init__()
        self.feature_extraction = Sequential(
            ResNet50.Stage1.InitialLayer(in_channels=3, out_channels=64),
            ResNet50.Stage2.Model(in_channels=64, out_channels=256),
            ResNet50.Stage3.Model(in_channels=256, out_channels=512, num_layer=4),
            ResNet50.Stage4.Model(in_channels=512, out_channels=1024, num_layer=6),
        )
        self.rpn = RPN.Model(channels=1024, device=device)
        self.roi = ROI.Model(channels=1024, num_class=100, device= device)

    def fast_nms(self, preds: Tensor, iou_threshold=0.5, score_threshold=0.5) -> Tensor:
        out: Tensor = tensor([])
        for b in preds:
            scores = b[:, 0]
            boxes = b[:, 2:6]

            keep = scores > score_threshold
            boxes = boxes[keep]
            scores = scores[keep]

            keep_idx = nms(boxes, scores, iou_threshold)
            out = cat([out,(b[keep][keep_idx]).unsqueeze(0)])
        return out

    def forward(self, x: Tensor):
        x = self.feature_extraction(x)
        score = self.rpn(x)
        score = self.fast_nms(score)
        score *= 48
        pred = self.roi(x, score)
        return cat([score, pred],dim=2)
    
class Model(CModel):
    def __init__(self, device: device):
        self.model = FasterRCNN(device)
        self.opt = Adam(self.model.parameters(), lr=0.01)
    def train(self, x: Dataset, loss: Callable[[Tensor, Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for i in range(size):
            tens = x.getTrainTensor(i)
            out = self.model(tens)
            lss = loss(out, x.getTrainLabel(i).unsqueeze(0))
            self.opt.zero_grad()
            lss.backward()
            self.opt.step()
            show_progress_counter(i+1, size, start, f"Loss: {lss}")
        show_progress_counter(size, size, start, "Done")
        save(self.model.state_dict(), "model.pth")