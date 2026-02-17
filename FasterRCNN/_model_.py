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
def faster_rcnn_loss(pred: Tensor, gt: Tensor, iou_thresh=0.5) -> Tensor:
    B, N, _ = pred.shape
    obj_logits = pred[:, :, 0:2]  # [B, N, 2] - object/background logits
    box_pred = pred[:, :, 2:6]    # [B, N, 4] - box deltas
    class_logits = pred[:, :, 6:] # [B, N, 100] - class scores
    
    total_obj_loss = 0
    total_box_loss = 0
    total_cls_loss = 0
    num_batches = 0
    for b in range(B):
        pred_boxes = box_pred[b]
        gt_boxes = gt[b][:, :4]
        gt_classes = gt[b][:, 4].long()
        
        valid_gt_mask = gt_classes >= 0
        gt_boxes = gt_boxes[valid_gt_mask]
        gt_classes = gt_classes[valid_gt_mask]
        
        if gt_boxes.numel() == 0:
            # If no GT boxes, all predictions should be background
            labels = zeros(N, dtype=long, device=pred.device)
            obj_loss = cross_entropy(obj_logits[b], labels)
            total_obj_loss += obj_loss
            num_batches += 1
            continue
        
        # Compute IoU between predictions and GT
        ious = box_iou(pred_boxes, gt_boxes)
        max_iou, matched_idx = ious.max(dim=1)
        
        # Objectness labels: 1 for positive, 0 for background
        labels = zeros(N, dtype=long, device=pred.device)
        pos_mask = max_iou >= iou_thresh
        
        # Handle ambiguous samples (0.3 < IoU < 0.5) - ignore them in loss
        ignore_mask = (max_iou >= 0.3) & (max_iou < iou_thresh)
        
        labels[pos_mask] = 1
        labels[ignore_mask] = -1  # Ignore in loss
        
        # Objectness loss (only for non-ignored samples)
        obj_loss = cross_entropy(
            obj_logits[b][~ignore_mask], 
            labels[~ignore_mask],
            reduction='sum'
        )
        
        # Box regression loss (only for positive samples)
        box_loss = 0
        cls_loss = 0
        if pos_mask.sum() > 0:
            matched_gt_boxes = gt_boxes[matched_idx[pos_mask]]
            matched_gt_classes = gt_classes[matched_idx[pos_mask]]
            
            box_loss = smooth_l1_loss(
                pred_boxes[pos_mask],
                matched_gt_boxes,
                reduction='sum'
            )
            
            cls_loss = cross_entropy(
                class_logits[b][pos_mask],
                matched_gt_classes,
                reduction='sum'
            )
        
        total_obj_loss += obj_loss
        total_box_loss += box_loss
        total_cls_loss += cls_loss
        num_batches += 1
    
    # Normalize losses
    if num_batches > 0:
        total_obj_loss /= num_batches
        total_box_loss /= num_batches
        total_cls_loss /= num_batches
    
    total_loss = total_obj_loss + total_box_loss + total_cls_loss
    
    return total_loss

class FasterRCNN(Module):
    def __init__(self, device: device):
        super().__init__()
        self.feature_extraction = Sequential(
            ResNet50.Stage1.InitialLayer(in_channels=3, out_channels=64, device=device),
            ResNet50.Stage2.Model(in_channels=64, out_channels=256, device=device),
            ResNet50.Stage3.Model(in_channels=256, out_channels=512, num_layer=4, device=device),
            ResNet50.Stage4.Model(in_channels=512, out_channels=1024, num_layer=6, device=device),
        )
        self.rpn = RPN.Model(channels=1024, device=device)
        self.roi = ROI.Model(channels=1024, num_class=100, device= device)

    def fast_nms(self, preds: Tensor, iou_threshold=0.5, score_threshold=0.5) -> Tensor:
        out: Tensor = tensor([], device=preds.device)
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
        # score = self.fast_nms(score)
        score[2:6] *= 48
        pred = self.roi(x, score)
        return cat([score, pred],dim=-1)
    
class Model(CModel):
    def __init__(self, device: device):
        self.model = FasterRCNN(device)
        self.opt = Adam(self.model.parameters(), lr=0.01)
        self.device = device
    def train(self, x: Dataset, loss: Callable[[Tensor, Tensor], Tensor]):
        size = x.getTrainSize()
        start = time()
        for i in range(size):
            tens:Tensor = x.getTrainTensor(i).to(self.device)
            label:Tensor =  x.getTrainLabel(i).unsqueeze(0).to(self.device)
            if (label.shape[1] == 0):
              continue
            out = self.model(tens)
            lss = loss(out,label)
            self.opt.zero_grad()
            lss.backward()
            self.opt.step()
            show_progress_counter(i+1, size, start, f"Proposal: {out.shape[1]} Loss: {lss}")
        show_progress_counter(size, size, start, "Done")
        save(self.model.state_dict(), "model.pth")