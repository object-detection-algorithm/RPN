# -*- coding: utf-8 -*-

"""
@date: 2020/5/9 下午8:28
@file: resize.py
@author: zj
@description: 
"""

import cv2


class Resize(object):
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.width, self.height))
        return image, boxes, labels
