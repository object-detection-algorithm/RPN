# -*- coding: utf-8 -*-

"""
@date: 2020/5/22 下午10:33
@file: build.py
@author: zj
@description: 
"""

from .rpn_detector import RPNDetector

_DETECTION_META_ARCHITECTURES = {
    "RPNDetector": RPNDetector
}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
