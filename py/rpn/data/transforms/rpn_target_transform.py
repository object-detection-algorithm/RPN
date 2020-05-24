# -*- coding: utf-8 -*-

"""
@date: 2020/5/23 上午10:21
@file: rpn_target_transform.py
@author: zj
@description: 
"""

import torch
import numpy as np
from rpn.utils import box_utils
from rpn.models.anchors import AnchorBox


class RPNTargetTransform:

    def __init__(self, cfg):
        self.anchor_model = AnchorBox(cfg)
        self.center_variance = cfg.MODEL.CENTER_VARIANCE
        self.size_variance = cfg.MODEL.SIZE_VARIANCE
        # 正负样本划分阈值
        self.pos_threshold = cfg.MODEL.POS_THRESHOLD
        self.neg_threshold = cfg.MODEL.NEG_THRESHOLD
        # 用于训练的正负样本
        self.num_pos = int(cfg.MODEL.N_CLS * cfg.MODEL.POS_NEG_RATIO)
        self.num_neg = int(cfg.MODEL.N_CLS - self.num_pos)

    def __call__(self, image, gt_boxes, gt_labels):
        h, w = image.shape[:2]
        center_form_anchors = self.anchor_model(h, w)
        corner_form_anchors = box_utils.center_form_to_corner_form(center_form_anchors)

        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        # boxes: [num_priors, 4] 每个先验框对应的标注框坐标
        # labels: [num_priors] 每个先验框对应标签
        boxes, labels = box_utils.assign_anchors(gt_boxes, gt_labels, corner_form_anchors,
                                                 self.pos_threshold, self.neg_threshold, self.num_pos, self.num_neg)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, center_form_anchors,
                                                         self.center_variance, self.size_variance)

        return locations, labels
