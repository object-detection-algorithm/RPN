# -*- coding: utf-8 -*-

"""
@date: 2020/5/23 上午10:10
@file: normalize.py
@author: zj
@description: 
"""

import torch
import torchvision.transforms as transforms


class Normalize(object):

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, cvimage, boxes=None, labels=None):
        assert isinstance(cvimage, torch.Tensor)
        return transforms.Normalize(self.mean, self.std)(cvimage), boxes, labels
