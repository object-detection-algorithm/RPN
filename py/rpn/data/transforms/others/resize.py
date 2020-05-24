# -*- coding: utf-8 -*-

"""
@date: 2020/5/9 下午8:28
@file: resize.py
@author: zj
@description: 
"""

import cv2


class Resize(object):

    def __init__(self, short_side):
        self.short_side = short_side

    # def __init__(self, width=800, height=600):
    #     self.width = width
    #     self.height = height

    def __call__(self, image, boxes=None, labels=None):
        h, w = image.shape[:2]
        if h > w:
            ratio = 1.0 * self.short_side / w
        else:
            ratio = 1.0 * self.short_side / h

        image = cv2.resize(image, (0, 0), fx=ratio, fy=ratio)
        return image, boxes, labels
