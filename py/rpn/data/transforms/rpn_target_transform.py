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


class RPNTargetTransform:

    def __init__(self, center_form_anchors, center_variance, size_variance,
                 pos_threshold, neg_threshold, num_cls, pos_neg_ratio):
        self.center_form_anchors = center_form_anchors
        self.corner_form_anchors = box_utils.center_form_to_corner_form(center_form_anchors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        # 正负样本划分阈值
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        # 用于训练的正负样本
        self.num_pos = int(num_cls * pos_neg_ratio)
        self.num_neg = int(num_cls - self.num_pos)

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        # boxes: [num_priors, 4] 每个先验框对应的标注框坐标
        # labels: [num_priors] 每个先验框对应标签
        boxes, labels = box_utils.assign_anchors(gt_boxes, gt_labels, self.corner_form_anchors,
                                                 self.pos_threshold, self.neg_threshold, self.num_pos, self.num_neg)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_anchors, self.center_variance,
                                                         self.size_variance)

        return locations, labels
