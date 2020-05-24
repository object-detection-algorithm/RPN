# -*- coding: utf-8 -*-

"""
@date: 2020/5/20 下午4:12
@file: anchor_box.py
@author: zj
@description: 
"""

from itertools import product
import torch
from rpn.utils import box_utils


class AnchorBox(object):

    def __init__(self, cfg):
        anchor_config = cfg.MODEL.ANCHORS

        self.feature_h = anchor_config.FEATURE_MAP_HEIGHT
        self.feature_w = anchor_config.FEATURE_MAP_WIDTH

        self.stride = anchor_config.STRIDE
        self.sizes = anchor_config.SIZES
        self.aspect_ratios = anchor_config.ASPECT_RATIOS
        self.clip = anchor_config.CLIP

    def __call__(self):
        """Generate RPN Anchor Boxes.
            It returns the center, height and width of the anchors. The values are relative to the image size
            Returns:
                priors (num_anchors, 4): The anchor boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        anchors = []
        # i表示第几行，j表示第几列
        for i, j in product(range(self.feature_h), range(self.feature_w)):
            # unit center x,y
            cx = (j + 0.5) / self.feature_w
            cy = (i + 0.5) / self.feature_h

            # [cx, cy, w, h]
            for size, ratio in product(self.sizes, self.aspect_ratios):
                # 高是较短边
                ratio_form_h = size / self.stride / self.feature_h

                anchors.append([cx, cy, ratio_form_h * ratio, ratio_form_h])

        anchors = torch.tensor(anchors)
        if self.clip:
            # anchors.clamp_(max=1.0, min=0.0)

            corner_form_anchors = box_utils.center_form_to_corner_form(anchors)
            corner_form_anchors.clamp_(max=1.0, min=0.0)

            anchors = box_utils.corner_form_to_center_form(corner_form_anchors)
        return anchors


if __name__ == '__main__':
    from rpn.config import cfg
    import torch

    model = AnchorBox(cfg)
    anchors = model()
    print(anchors.shape)
