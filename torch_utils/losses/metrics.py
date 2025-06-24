""" 
Author: Stte 
Date: 2025-04-03 
Descriptions: z 
"""


import torch.nn as nn
import torch, os
from torch.hub import load_state_dict_from_url
from torch.nn import functional as F
from torchvision.models import resnet50, resnet18

class ResNet18(nn.Module):
    def __init__(self, in_c, b_cls=True, b_as_loss=False):
        super().__init__()

        self.b_cls = b_cls
        self.b_as_loss = b_as_loss
        self.backbone = resnet18()
        self.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,  # 新的输入通道数
            out_channels=self.backbone.conv1.out_channels,  # 保持输出通道数不变
            kernel_size=self.backbone.conv1.kernel_size,  # 保持卷积核大小不变
            stride=self.backbone.conv1.stride,  # 保持步长不变
            padding=self.backbone.conv1.padding,  # 保持填充不变
            bias=self.backbone.conv1.bias is not None  # 保持是否有偏置项
        )
        self.mlp = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
                                nn.Linear(128, 2),
                                nn.Sigmoid()
                        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)

        if self.b_cls:
            cls_logits = self.cls(x)

        if self.b_as_loss:
            return x

        return cls_logits, x
class ResNet50(nn.Module):
    def __init__(self, in_c, b_cls=True, b_as_loss=False):
        super().__init__()

        self.b_cls = b_cls
        self.b_as_loss = b_as_loss
        self.backbone = resnet18()
        self.backbone.conv1 = torch.nn.Conv2d(
            in_channels=1,  # 新的输入通道数
            out_channels=self.backbone.conv1.out_channels,  # 保持输出通道数不变
            kernel_size=self.backbone.conv1.kernel_size,  # 保持卷积核大小不变
            stride=self.backbone.conv1.stride,  # 保持步长不变
            padding=self.backbone.conv1.padding,  # 保持填充不变
            bias=self.backbone.conv1.bias is not None  # 保持是否有偏置项
        )
        self.mlp = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.cls = nn.Sequential(
                                nn.Linear(128, 2),
                                nn.Sigmoid()
                        )


    def forward(self, x):
        x = self.backbone(x)
        x = self.mlp(x)

        if self.b_cls:
            cls_logits = self.cls(x)

        if self.b_as_loss:
            return x

        return cls_logits, x


# ============================ #
#                       #
# ============================ #
def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )
def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )
class MobileNetV1(nn.Module):
    def __init__(self, in_c=3):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(in_c, 32, 2),
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1),

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            # conv_dw(512, 512, 1),
            # conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    
class mobilenet(nn.Module):
    def __init__(self, pretrained, in_c=3):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1(in_c=in_c)
        if pretrained:
            state_dict = load_state_dict_from_url(
                "https://github.com/bubbliiiing/facenet-pytorch/releases/download/v1.0/backbone_weights_of_mobilenetv1.pth",
                model_dir="model_data",
                progress=True)
            self.model.load_state_dict(state_dict)

        del self.model.fc
        del self.model.avg

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x

class Facenet(nn.Module):
    def __init__(self, backbone="mobilenet", in_c=3, dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train",
                 pretrained=False):
        super(Facenet, self).__init__()
        self.mode = mode
        if backbone == "mobilenet":
            self.backbone = mobilenet(pretrained, in_c=in_c)
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            # self.backbone = inception_resnet(pretrained)
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        # todo:为什么是1d?
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        print("num_class==> ", num_classes)
        # if mode == "train":
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.mode == 'predict':
            x = self.backbone(x)
            x = self.avg(x)
            x = x.view(x.size(0), -1)
            x = self.Dropout(x)
            x = self.Bottleneck(x)
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    import time
    start_time = time.perf_counter()

    model = Facenet(backbone="mobilenet", in_c=1, dropout_keep_prob=0.5, embedding_size=128, num_classes=2, mode="train")

    data = torch.randn(4, 1, 768, 768)
    output = model(data)

    end_time = time.perf_counter()
    print("Time: {:.2f} s".format(end_time - start_time))
    pass

class TripletLoss(nn.Module):
    def __init__(self, alpha=0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, A, P, N):
        pos_dist = torch.sqrt(torch.sum(torch.pow(A - P, 2), axis=-1))
        neg_dist = torch.sqrt(torch.sum(torch.pow(A - N, 2), axis=-1))

        # keep_all = (neg_dist - pos_dist < self.alpha).cpu().numpy().flatten()  # 大于这个alpha说明距离已经很远了，没必要管
        keep_all = neg_dist - pos_dist < self.alpha    # 大于这个alpha说明距离已经很远了，没必要管
        hard_triplets = torch.where(keep_all == 1)  # 其实是keep_all == True

        pos_dist = pos_dist[hard_triplets]
        neg_dist = neg_dist[hard_triplets]

        basic_loss = pos_dist - neg_dist + self.alpha
        loss = torch.sum(basic_loss) / torch.max(torch.tensor(1), torch.tensor(len(hard_triplets[0])))  # 向量元素求和变成标量
        return loss


class MaskSmoothLoss(nn.Module):
    def __init__(self, backbone='resnet', loss=nn.MSELoss(), gpu_id_list = [0]):
        super().__init__()
        ENCODER_TYPE = ['resnet', 'facenet', 'facenet_2']
        assert backbone in ENCODER_TYPE, f'Unknown backbone: {backbone}. Only support {ENCODER_TYPE}'

        if backbone == 'resnet':
            self.backbone = ResNet18(in_c=1, b_cls=False, b_as_loss=True).eval()
        elif 'facenet' in backbone:
            self.backbone = Facenet(in_c=1, mode='predict', num_classes=2).eval()
        else:
            assert False, f'Unknown backbone: {backbone}. Only support {ENCODER_TYPE}'

        self.loss = loss
        self.dict_checpoints = {
            'resnet': '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth_resnet.pth',
            'facenet': '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth_facenet.pth',
            'facenet_2': '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth_facenet_2.pth',
        }

        # model_dir = __file__.split('/sugar/')[0] + "/sugar/TORCH_MODEL_ZOO"
        cached_file = self.dict_checpoints[backbone]
        if os.path.exists(cached_file):
            self.backbone.load_state_dict(torch.load(cached_file))
            self.gpu_id_list = gpu_id_list
            self.eval()
        else:
            assert False, f'No Such File ---- {cached_file}'


    def forward(self, input, tar):
        inputFeatures = self.backbone(input)
        with torch.no_grad():
            tarFeatures = self.backbone(tar)

        err_vgg = None
        length = inputFeatures.__len__()
        for i in range(length):
            e = self.loss(inputFeatures[i], tarFeatures[i].detach()) / inputFeatures[i].numel()
            if err_vgg is None:
                err_vgg = e
            else:
                err_vgg += e
        return err_vgg


class MaskDiscriminator(nn.Module):
    def __init__(self, loss=nn.MSELoss(), gpu_id_list = None):
        super().__init__()
        self.backbone = ResNet18(in_c=1, b_cls=False, b_as_loss=True)
        self.loss = loss

        # cached_file = '/data_ssd/ay/切脸数据/checkpoints/MaskSmooth.pth'
        # self.load_state_dict(torch.load(cached_file))
        # self.discriminator.Split()

    def forward(self, input):
        return self.backbone(input)