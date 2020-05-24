# -*- coding: utf-8 -*-

"""
@date: 2020/5/22 下午10:41
@file: rpn_box_head.py
@author: zj
@description: 
"""

from torch import nn
import torch.nn.functional as F

from rpn.models import registry
from rpn.models.anchors import AnchorBox
from rpn.models.box_head.rpn_box_predictor import build_box_predictor
from rpn.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss


@registry.BOX_HEADS.register('RPNBoxHead')
class RPNBoxHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.predictor = build_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(cfg.MODEL.LAMBDA)
        self.post_processor = PostProcessor(cfg)

    def forward(self, features, image_h, image_w, targets=None):
        cls_logits, bbox_pred = self.predictor(features)
        if self.training:
            return self._forward_train(cls_logits, bbox_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred, features, image_h, image_w)

    def _forward_train(self, cls_logits, bbox_pred, targets):
        gt_boxes, gt_labels = targets['boxes'], targets['labels']
        reg_loss, cls_loss = self.loss_evaluator(cls_logits, bbox_pred, gt_labels, gt_boxes)
        loss_dict = dict(
            reg_loss=reg_loss,
            cls_loss=cls_loss,
        )
        detections = (cls_logits, bbox_pred)
        return detections, loss_dict

    def _forward_test(self, cls_logits, bbox_pred, features, image_h, image_w):
        feature_h = features.shape[2]
        feature_w = features.shape[3]
        self.anchors = AnchorBox(self.cfg)(feature_h, feature_w).to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.anchors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes)
        detections = self.post_processor(detections, image_h, image_w)
        return detections, {}
