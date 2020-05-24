import torch

from rpn.structures.container import Container
from rpn.utils.nms import batched_nms


class PostProcessor:

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def __call__(self, detections, image_h, image_w):
        batches_scores, batches_boxes = detections
        device = batches_scores.device
        batch_size = batches_scores.size(0)
        results = []
        # 逐个图像进行NMS计算
        for batch_id in range(batch_size):
            scores, boxes = batches_scores[batch_id], batches_boxes[batch_id]  # (N, #CLS) (N, 4)
            num_boxes = scores.shape[0]
            num_classes = scores.shape[1]

            boxes = boxes.view(num_boxes, 1, 4).expand(num_boxes, num_classes, 4)
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, num_classes).expand_as(scores)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)

            # remove low scoring boxes
            # 返回非零值的下标
            indices = torch.nonzero(scores > self.cfg.TEST.CONFIDENCE_THRESHOLD).squeeze(1)
            # 获取超过置信度阈值的边界框及置信度和标签
            boxes, scores, labels = boxes[indices], scores[indices], labels[indices]

            # 从分数形式转换为绝对值
            boxes[:, 0::2] *= image_w
            boxes[:, 1::2] *= image_h

            keep = batched_nms(boxes, scores, labels, self.cfg.TEST.NMS_THRESHOLD)
            # keep only topk scoring predictions
            # NMS之后，每个图像仅保留前k个边界框
            keep = keep[:self.cfg.TEST.MAX_PER_IMAGE]
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

            container = Container(boxes=boxes, labels=labels, scores=scores)
            container.img_width = image_w
            container.img_height = image_h
            results.append(container)
        return results
