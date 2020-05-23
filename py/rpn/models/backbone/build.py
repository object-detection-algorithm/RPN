# -*- coding: utf-8 -*-

"""
@date: 2020/5/14 下午2:49
@file: build.py
@author: zj
@description: 
"""

from rpn.models import registry

from .vgg import VGG

def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME]()
