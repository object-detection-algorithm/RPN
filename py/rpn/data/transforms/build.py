# -*- coding: utf-8 -*-

"""
@date: 2020/5/14 上午10:10
@file: build.py
@author: zj
@description: 
"""

from .color import PhotometricDistort
from .others import *
from .compose import Compose

from rpn.models.anchors import AnchorBox
from .rpn_target_transform import RPNTargetTransform


def build_transforms(cfg, is_train=True):
    if is_train:
        transform = [
            ConvertFromInts(),
            PhotometricDistort(),
            Expand(cfg.INPUT.PIXEL_MEAN),
            RandomSampleCrop(),
            RandomMirror(),
            ToPercentCoords(),
            Resize(cfg.INPUT.IMAGE_WIDTH, cfg.INPUT.IMAGE_HEIGHT),
            # SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
            Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        ]
    else:
        transform = [
            Resize(cfg.INPUT.IMAGE_WIDTH, cfg.INPUT.IMAGE_HEIGHT),
            # SubtractMeans(cfg.INPUT.PIXEL_MEAN),
            ToTensor(),
            Normalize(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
        ]
    transform = Compose(transform)
    return transform


def build_target_transform(cfg):
    transform = RPNTargetTransform(AnchorBox(cfg)(),
                                   cfg.MODEL.CENTER_VARIANCE,
                                   cfg.MODEL.SIZE_VARIANCE,
                                   cfg.MODEL.POS_THRESOHLD,
                                   cfg.MODEL.NEG_THRESHOLD,
                                   cfg.MODEL.N_CLS,
                                   cfg.MODEL.POS_NEG_RATIO)
    return transform
