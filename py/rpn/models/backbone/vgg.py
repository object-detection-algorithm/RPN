import torch.nn as nn
from rpn.models import registry

from torchvision.models import vgg16
import torch


@registry.BACKBONES.register('vgg')
class VGG(nn.Module):

    def __init__(self):
        super().__init__()

        # 取Conv5_3作为特征图
        self.features = vgg16(pretrained=True).features[:30]

        # 使用预训练模型后，从Conv3_1开始进行微调
        for idx, param in enumerate(self.features.parameters()):
            if idx < 8:
                param.requires_grad = False
            # print(idx, param.shape, param.requires_grad)

    def forward(self, x):
        return self.features(x)


if __name__ == '__main__':
    model = VGG()
    # print(model)

    x = torch.randn(1, 3, 800, 600)
    outputs = model(x)
    print(outputs.shape)
