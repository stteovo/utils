import torch
import torch.nn as nn

from torchvision.models import resnet50, resnet18

class ResNet18(nn.Module):
    def __init__(self, in_c, cls=True):
        super().__init__()

        self.cls = cls
        self.backbone = resnet18()
        self.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,  # 新的输入通道数
            out_channels=self.backbone.conv1.out_channels,  # 保持输出通道数不变
            kernel_size=self.backbone.conv1.kernel_size,  # 保持卷积核大小不变
            stride=self.backbone.conv1.stride,  # 保持步长不变
            padding=self.backbone.conv1.padding,  # 保持填充不变
            bias=self.backbone.conv1.bias is not None  # 保持是否有偏置项
        )

        self.cls = nn.Linear(1000, 2)


    def forward(self, x):
        x = self.backbone(x)

        if self.cls:
            x = self.cls(x)

        return x


if __name__ == '__main__':
    model = ResNet18(in_c=1)

    data = torch.randn(1, 1, 768, 768)
    output = model(data)
    pass