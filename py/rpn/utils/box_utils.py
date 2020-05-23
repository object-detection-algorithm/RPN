# -*- coding: utf-8 -*-

"""
@date: 2020/5/9 下午8:43
@file: box_utils.py
@author: zj
@description: 
"""

import numpy as np
import torch
import math


def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    """Convert regressional location results into boxes in the form of (center_x, center_y, h, w).

    The conversion:
        $$predicted\_center * center_variance = \frac {real\_center - prior\_center} {prior\_hw}$$
        $$exp(predicted\_hw * size_variance) = \frac {real\_hw} {prior\_hw}$$
    We do it in the inverse direction here.
    Args:
        locations (batch_size, num_priors, 4): the regression output. It will contain the outputs as well.
        priors (num_priors, 4) or (batch_size/1, num_priors, 4): prior boxes.
        center_variance: a float used to change the scale of center.
        size_variance: a float used to change of scale of size.
    Returns:
        boxes:  priors: [[center_x, center_y, w, h]]. All the values
            are relative to the image size.
    """
    # priors can have one dimension less.
    if priors.dim() + 1 == locations.dim():
        priors = priors.unsqueeze(0)
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)


def convert_boxes_to_locations(center_form_boxes, center_form_priors, center_variance, size_variance):
    # priors can have one dimension less
    if center_form_priors.dim() + 1 == center_form_boxes.dim():
        center_form_priors = center_form_priors.unsqueeze(0)
    return torch.cat([
        (center_form_boxes[..., :2] - center_form_priors[..., :2]) / center_form_priors[..., 2:] / center_variance,
        torch.log(center_form_boxes[..., 2:] / center_form_priors[..., 2:]) / size_variance
    ], dim=center_form_boxes.dim() - 1)


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)


def assign_anchors(gt_boxes, gt_labels, corner_form_anchors,
                   pos_threshold, neg_threshold, num_pos, num_neg):
    """Assign ground truth boxes and targets to anchors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        anchors (num_anchors, 4): corner form anchors
    Returns:
        boxes (num_anchors, 4): real values for anchors.
        labels (num_anchors): labels for anchors.
    """
    # size: [num_anchors, num_targets]
    # 每行表示单个锚点框与各个标注框的IoU
    # 每列表示单个标注框与各个锚点框的IoU
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_anchors.unsqueeze(1))
    # size: [num_anchors]
    # best_target_per_anchors：每个锚点框计算得到的最高IoU
    # best_target_per_anchor_index：每个锚点框对应最高IoU的标注框下标
    best_target_per_anchors, best_target_per_anchor_index = ious.max(1)
    # size: [num_targets]
    # best_anchor_per_target：每个标注框计算得到的最高IoU
    # best_anchor_per_target_index：每个标注框对应最高IoU的锚点框下标
    best_anchor_per_target, best_anchor_per_target_index = ious.max(0)

    # 确保标注框与最高IoU的锚点框匹配
    for target_index, anchor_index in enumerate(best_anchor_per_target_index):
        best_target_per_anchor_index[anchor_index] = target_index

    # size: [num_anchors]
    # 得到每个锚点框对应标注框的标签/类别
    labels = gt_labels[best_target_per_anchor_index]
    # size: [num_anchors, 4]
    # 得到每个锚点框对应标注框的坐标
    boxes = gt_boxes[best_target_per_anchor_index]

    # 2.0 is used to make sure every target has a anchor assigned
    # 确保每个标注框对应IoU最高的锚点框的阈值大于pos_threshold
    best_target_per_anchors.index_fill_(0, best_anchor_per_target_index, 2)
    # 设置正样本
    labels[best_target_per_anchors >= pos_threshold] = 1
    # 设置负样本 + 不参与训练样本
    labels[best_target_per_anchors < pos_threshold] = -1
    # IoU小于neg_threshold的锚点框设置为背景类别
    labels[best_target_per_anchors < neg_threshold] = 0  # the backgournd id

    # print('pos', sum(labels == 1))
    # 随机采样num_pos个正样本用于训练
    if sum(labels == 1) > num_pos:
        pos_indices = np.where(labels == 1)[0]
        disable_index = np.random.choice(pos_indices, size=(len(pos_indices) - num_pos), replace=False)
        # 　不参与分类损失计算的正样本标签设置为２
        labels[disable_index] = 2
    else:
        num_neg += (num_pos - sum(labels == 1).item())

    # print('neg', num_neg)
    # 随机采样num_neg个负样本用于训练
    neg_indices = np.where(labels == 0)[0]
    if sum(neg_indices) > num_neg:
        disable_index = np.random.choice(
            neg_indices, size=(len(neg_indices) - num_neg), replace=False)
        labels[disable_index] = -1

    return boxes, labels


def assign_priors(gt_boxes, gt_labels, corner_form_priors,
                  iou_threshold):
    """Assign ground truth boxes and targets to priors.

    Args:
        gt_boxes (num_targets, 4): ground truth boxes.
        gt_labels (num_targets): labels of targets.
        priors (num_priors, 4): corner form priors
    Returns:
        boxes (num_priors, 4): real values for priors.
        labels (num_priros): labels for priors.
    """
    # size: [num_priors, num_targets]
    # 每行表示单个先验框与各个标注框的IoU
    # 每列表示单个标注框与各个先验框的IoU
    ious = iou_of(gt_boxes.unsqueeze(0), corner_form_priors.unsqueeze(1))
    # size: [num_priors]
    # best_target_per_prior：每个先验框计算得到的最高IoU
    # best_target_per_prior_index：每个先验框对应最高IoU的标注框下标
    best_target_per_prior, best_target_per_prior_index = ious.max(1)
    # size: [num_targets]
    # best_prior_per_target：每个标注框计算得到的最高IoU
    # best_prior_per_target_index：每个标注框对应最高IoU的先验框下标
    best_prior_per_target, best_prior_per_target_index = ious.max(0)

    # 确保标注框与最高IoU的先验框匹配
    for target_index, prior_index in enumerate(best_prior_per_target_index):
        best_target_per_prior_index[prior_index] = target_index

    # size: [num_priors]
    # 得到每个先验框对应标注框的标签/类别
    labels = gt_labels[best_target_per_prior_index]
    # size: [num_priors, 4]
    # 得到每个先验框对应标注框的坐标
    boxes = gt_boxes[best_target_per_prior_index]

    # 2.0 is used to make sure every target has a prior assigned
    # 确保每个标注框对应IoU最高的先验框的阈值大于iou_threshold
    best_target_per_prior.index_fill_(0, best_prior_per_target_index, 2)
    # IoU小于iou_threshold的先验框设置为背景类别
    labels[best_target_per_prior < iou_threshold] = 0  # the backgournd id

    return boxes, labels


def hard_negative_mining(labels, N_cls, neg_pos_ratio):
    """
    保留所有正样本数目，保持正负样本比为neg_pos_ratio
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.

    Args:
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


def center_form_to_corner_form(locations):
    return torch.cat([locations[..., :2] - locations[..., 2:] / 2,
                      locations[..., :2] + locations[..., 2:] / 2], locations.dim() - 1)


def corner_form_to_center_form(boxes):
    return torch.cat([
        (boxes[..., :2] + boxes[..., 2:]) / 2,
        boxes[..., 2:] - boxes[..., :2]
    ], boxes.dim() - 1)
