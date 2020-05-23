# -*- coding: utf-8 -*-

"""
@date: 2020/5/9 下午8:29
@file: to_tensor.py
@author: zj
@description: 
"""

import torch
import torchvision.transforms as transforms
import numpy as np


class ToTensor(object):

    def __call__(self, cvimage, boxes=None, labels=None):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

        return transform(cvimage.astype(np.uint8)), boxes, labels
