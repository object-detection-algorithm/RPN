# -*- coding: utf-8 -*-

"""
@date: 2020/5/23 上午11:55
@file: rpn_box_predictor.py
@author: zj
@description: 
"""

import torch
from torch import nn

from rpn.models import registry


@registry.BOX_PREDICTORS.register('RPNBoxPredictor')
class RPNBoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.conv = nn.Conv2d(cfg.MODEL.BOX_HEAD.FEATURE_MAP, cfg.MODEL.BOX_HEAD.CONV_OUTPUT, kernel_size=3, stride=1,
                              padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.cls_header = self.cls_block(cfg.MODEL.BOX_HEAD.CONV_OUTPUT, cfg.MODEL.ANCHORS.NUM, cfg.MODEL.NUM_CLASSES)
        self.reg_header = self.reg_block(cfg.MODEL.BOX_HEAD.CONV_OUTPUT, cfg.MODEL.ANCHORS.NUM)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        # features: (N, dims, feature_h, feature_w)
        x = self.conv(features)
        x = self.relu(x)

        # cls_logits: (N, feature_h, feature_w, num_per_location * num_classes)
        # bbox_pred: (N, feature_h, feature_w, num_per_location * 4)
        cls_logits = self.cls_header(x).permute(0, 2, 3, 1).contiguous()
        bbox_pred = self.reg_header(x).permute(0, 2, 3, 1).contiguous()

        # N
        batch_size = features.shape[0]
        # (N, feature_h, feature_w, num_per_location * num_classes) ->
        # (N, feature_h * feature_w * num_per_location, num_classes)
        cls_logits = cls_logits.view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        bbox_pred = bbox_pred.view(batch_size, -1, 4)

        return cls_logits, bbox_pred

    def cls_block(self, out_channels, boxes_per_location, num_classes):
        return nn.Conv2d(out_channels, boxes_per_location * num_classes, kernel_size=1, stride=1)

    def reg_block(self, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1, stride=1)


def build_box_predictor(cfg):
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
