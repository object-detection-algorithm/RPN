# -*- coding: utf-8 -*-

"""
@date: 2020/5/22 下午10:32
@file: rpn_detector.py
@author: zj
@description: 
"""

from torch import nn

from rpn.models.backbone import build_backbone
from rpn.models.box_head import build_box_head


class RPNDetector(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
