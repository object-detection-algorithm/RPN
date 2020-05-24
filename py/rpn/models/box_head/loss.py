import torch.nn as nn
import torch.nn.functional as F
import torch

from rpn.utils import box_utils


class MultiBoxLoss(nn.Module):

    def __init__(self, lam=10.0):
        """Implement RPN MultiBox Loss.

        Basically, MultiBox loss combines classification loss and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.lam = lam

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            predicted_locations (batch_size, num_anchors, 4): predicted locations.
            labels (batch_size, num_anchors): real labels of all the anchors.
            gt_locations (batch_size, num_anchors, 4): real boxes corresponding all the anchors.
        """
        num_classes = confidence.size(2)
        # 用于计算分类损失的正负样本
        pos_mask = labels == 1
        neg_mask = labels == 0
        mask = pos_mask | neg_mask

        confidence = confidence[mask, :].view(-1, num_classes)
        classification_loss = F.cross_entropy(confidence, labels[mask].view(-1), reduction='sum')
        num_pos_cls = confidence.size(0)

        # 用于计算回归损失的正样本
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos_reg = gt_locations.size(0)

        # 分类损失和回归损失之前还有一个超参数lambda
        return smooth_l1_loss / num_pos_reg * self.lam, classification_loss / num_pos_cls
