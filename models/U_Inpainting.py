import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv_pad(*args, **kwargs):
    # kwargs['padding_mode'] = 'circular'
    in_c, out_c, ks, s, p = args
    # p = p * 2
    return nn.Conv2d(in_c, out_c, ks, s, p, **kwargs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(DownBlock, self).__init__()
        self.down = nn.AvgPool2d(2, 2, 0)
        self.conv_left = conv_pad(in_c, out_c, 1, 1, 0)
        self.conv_1 = conv_pad(in_c, out_c, 3, 1, 1)
        self.conv_2 = conv_pad(out_c, out_c, 3, 1, 1)

    def forward(self, x):
        x_half = self.down(x)
        x_left = self.conv_left(x_half)
        x_right = F.relu(x_half)
        x_right = F.relu(self.conv_1(x_right))
        x_right = self.conv_2(x_right)
        return x_left + x_right


class res_conv(nn.Module):
    def __init__(self, in_c, out_c):
        super(res_conv, self).__init__()
        if in_c == out_c:
            self.conv_0 = None
        else:
            self.conv_0 = nn.Conv2d(in_c, out_c, 1, 1, 0)

        self.conv_1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, 3, 1, 1, groups=out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 1, 1, 0),
        )

    def forward(self, x):
        if self.conv_0 is None:
            return F.relu(x + self.conv_1(x))
        else:
            return F.relu(self.conv_0(x) + self.conv_1(x))


class SingleDown(nn.Module):
    def __init__(self, ch):
        super(SingleDown, self).__init__()
        self.down = nn.AvgPool2d(2, 2, 0)
        self.conv_1 = conv_pad(ch, ch, 3, 1, 1)
        self.conv_2 = conv_pad(ch, ch, 3, 1, 1)

    def forward(self, x):
        x_half = self.down(x)
        x_1 = F.relu(x_half)
        x_2 = F.relu(self.conv_1(x_1))
        x_3 = self.conv_2(x_2)
        return x_half + x_3


class ULink(nn.Module):
    def __init__(self, in_c, out_c, mid=None):
        super(ULink, self).__init__()
        if in_c == out_c:
            self.down_block = SingleDown(in_c)
        else:
            self.down_block = DownBlock(in_c, out_c)

        self.mid = mid
        if mid is None:
            ex_ch = 0
        else:
            ex_ch = out_c // 2
        self.conv = conv_pad(out_c + ex_ch, out_c, 3, 1, 1)
        self.up = nn.ConvTranspose2d(out_c, out_c//2, 4, 2, 1)

    def forward(self, x):
        x_half = self.down_block(x)
        if self.mid is not None:
            x_half = self.mid(x_half)
        x_1 = F.relu(x_half)
        x_2 = F.relu(self.conv(x_1))
        x_3 = self.up(x_2)
        return torch.cat([x, x_3], 1)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResBlock, self).__init__()
        self.conv_head = conv_pad(in_c, out_c, 3, 2, 1)
        self.conv_1 = conv_pad(out_c, out_c, 3, 1, 1)
        self.conv_2 = conv_pad(out_c, out_c, 3, 1, 1)

    def forward(self, x):
        x = self.conv_head(x)
        x_1 = F.relu(x)
        x_2 = F.relu(self.conv_1(x_1))
        x_3 = self.conv_2(x_2)
        return x + x_3


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=8):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

def _tensor_2_image(image):
    cpu_var = torch.clip((image + 1) * 127.5, 0, 255).cpu()
    if len(cpu_var.size()) == 4:
        tensor = cpu_var[0]
    else:
        tensor = cpu_var
    arr = tensor.detach().numpy()
    im = arr.transpose(1, 2, 0)
    im = im.astype(np.uint8)
    im = np.ascontiguousarray(im[:, :, ::-1])
    return im

class ScaleNLayer(nn.Module):
    def __init__(self, channel_in,channel_out):
        super(ScaleNLayer, self).__init__()
        self.conv1x1 = nn.Conv2d(channel_in, channel_in, 1, 1, 0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Linear(channel_in, channel_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in, channel_out, bias=False),
            nn.Sigmoid()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(channel_in, channel_in, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel_in, channel_out, bias=False),
            nn.Tanh()
        )
        self.ch_out = channel_out

    def forward(self, x):
        b, c, _, _ = x.size()
        x = self.conv1x1(x)
        y = self.avg_pool(x).view(b, c)
        w = self.fc1(y).view(b, self.ch_out, 1, 1)#*10
        wb = self.fc2(y).view(b, self.ch_out, 1, 1)#*10
        return w, wb


class EnConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=2, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(EnConv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(c2, c2, 1, 1, 0)

    def forward(self, x):
        return self.conv2(self.act(self.conv(x)))


class DeConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=2, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DeConv, self).__init__()
        self.conv = nn.ConvTranspose2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        # self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(c2, c2, 1, 1, 0)

    def forward(self, x):
        return self.conv2(self.act(self.conv(x)))


class ShallowNet(nn.Module):
    """
    Args：
     process diff map
    """

    def __init__(self, in_c=4, out_c=3):
        super(ShallowNet, self).__init__()

        self.layer1 = nn.Sequential(
            conv_pad(in_c, 32, 3, 2, 1),
            nn.ReLU(),
            conv_pad(32, 32, 3, 2, 1),
        )

        self.encode1 = EnConv(32, 64, 3, 2, 1)  # 128
        self.encode2 = EnConv(64, 128, 3, 2, 1)  # 64
        self.encode3 = EnConv(128, 256, 3, 2, 1)  # 32
        self.encode4 = EnConv(256, 512, 3, 2, 1)  # 16

        self.encode5 = nn.Sequential(
            res_conv(512, 512),
            SEBasicBlock(512, 512),
            res_conv(512, 512),
        )
        self.decode4 = DeConv(512, 256, 4, 2, 1)
        self.decode3 = DeConv(256, 128, 4, 2, 1)
        self.decode2 = DeConv(128, 64, 4, 2, 1)
        self.decode1 = DeConv(64, 32, 4, 2, 1)
        self.decode0 = DeConv(32, 32, 4, 2, 1)
        self.affine_4 = nn.Sequential(
            res_conv(32, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_pad(32, out_c, 3, 1, 1),
        )


    def forward(self, x_in, mask):
        x = torch.cat([x_in, mask], dim=1)
        x0 = self.layer1(x)
        e1 = self.encode1(x0)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        d4 = self.decode4(e5 + e4)
        d3 = self.decode3(d4 + e3)
        d2 = self.decode2(d3 + e2)
        d1 = self.decode1(d2 + e1)
        d0 = self.decode0(d1 + x0)

        x = self.affine_4(d0)
        fake_src, mask_pre_sigmoid = x[:, :3], x[:, 3:4]

        mask_out = torch.sigmoid(mask_pre_sigmoid)

        # res = fake_src * mask_out + x_in * (1 - mask_out)
        return fake_src, mask_out


class ShallowNetAot(nn.Module):
    """
    Args：
     process diff map
    """

    def __init__(self, in_c=4, out_c=3):
        super(ShallowNet, self).__init__()

        self.layer1 = nn.Sequential(
            conv_pad(in_c, 32, 3, 2, 1),
            nn.ReLU(),
            conv_pad(32, 32, 3, 2, 1),
        )

        self.encode1 = EnConv(32, 64, 3, 2, 1)  # 128
        self.encode2 = EnConv(64, 128, 3, 2, 1)  # 64
        self.encode3 = EnConv(128, 256, 3, 2, 1)  # 32
        self.encode4 = EnConv(256, 512, 3, 2, 1)  # 16

        self.encode5 = nn.Sequential(
            res_conv(512, 512),
            SEBasicBlock(512, 512),
            res_conv(512, 512),
        )
        self.decode4 = DeConv(512, 256, 4, 2, 1)
        self.decode3 = DeConv(256, 128, 4, 2, 1)
        self.decode2 = DeConv(128, 64, 4, 2, 1)
        self.decode1 = DeConv(64, 32, 4, 2, 1)
        self.decode0 = DeConv(32, 32, 4, 2, 1)
        self.affine_4 = nn.Sequential(
            res_conv(32, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_pad(32, out_c, 3, 1, 1),
        )


    def forward(self, x_in, mask):
        x = torch.cat([x_in, mask], dim=1)
        x0 = self.layer1(x)
        e1 = self.encode1(x0)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        d4 = self.decode4(e5 + e4)
        d3 = self.decode3(d4 + e3)
        d2 = self.decode2(d3 + e2)
        d1 = self.decode1(d2 + e1)
        d0 = self.decode0(d1 + x0)

        x = self.affine_4(d0)
        fake_src, mask_pre_sigmoid = x[:, :3], x[:, 3:4]

        mask_out = torch.sigmoid(mask_pre_sigmoid)

        # res = fake_src * mask_out + x_in * (1 - mask_out)
        return fake_src, mask_out


class ShallowNetExport(nn.Module):
    """
    Args：
     process diff map
    """

    def __init__(self, in_c=4, out_c=3, using_noise=False ):
        super(ShallowNetExport, self).__init__()

        self.layer1 = nn.Sequential(
            conv_pad(in_c, 32, 3, 2, 1),
            nn.ReLU(),
            conv_pad(32, 32, 3, 2, 1),
        )

        self.encode1 = EnConv(32, 64, 3, 2, 1)  # 128
        self.encode2 = EnConv(64, 128, 3, 2, 1)  # 64
        self.encode3 = EnConv(128, 256, 3, 2, 1)  # 32
        self.encode4 = EnConv(256, 512, 3, 2, 1)  # 16

        self.encode5 = nn.Sequential(
            res_conv(512, 512),
            SEBasicBlock(512, 512),
            res_conv(512, 512),
        )
        self.decode4 = DeConv(512, 256, 4, 2, 1)
        self.decode3 = DeConv(256, 128, 4, 2, 1)
        self.decode2 = DeConv(128, 64, 4, 2, 1)
        self.decode1 = DeConv(64, 32, 4, 2, 1)
        self.decode0 = DeConv(32, 32, 4, 2, 1)
        self.end1 = nn.Sequential(
            res_conv(32, 32),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_pad(32, out_c, 3, 1, 1),
        )
        self.end2 = nn.Sequential(
            conv_pad(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.UpsamplingBilinear2d(scale_factor=2),
            conv_pad(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        )

        self.scale_noise = ScaleNLayer(512, 32)
        self.scale_noise_out = ScaleNLayer(512, 3)
        self.using_noise = using_noise
    def forward(self, x_in):
        x = x_in
        x0 = self.layer1(x)
        e1 = self.encode1(x0)
        e2 = self.encode2(e1)
        e3 = self.encode3(e2)
        e4 = self.encode4(e3)
        e5 = self.encode5(e4)
        d4 = self.decode4(e5 + e4)
        d3 = self.decode3(d4 + e3)
        d2 = self.decode2(d3 + e2)
        d1 = self.decode1(d2 + e1)
        d0 = self.decode0(d1 + x0)

        x = d0
        if self.using_noise:
            xh, xw = d0.size()[-2:]
            noise = torch.randn([x.shape[0], x.shape[1], xh, xw], device=x.device)
            noise_scale, noise_baise = self.scale_noise(e5)
            noise_f = noise * noise_scale.expand_as(noise) + noise_baise.expand_as(noise)
            xn = x + noise_f
        else:
            xn = x
        rgb = torch.tanh(self.end1(xn))

        if self.using_noise:
            xh2, xw2 = rgb.size()[-2:]
            noise2 = torch.randn([rgb.shape[0], rgb.shape[1], xh2, xw2], device=rgb.device)
            noise_scale2, noise_baise2 = self.scale_noise_out(e5)
            noise_f2 = noise2 * noise_scale2.expand_as(noise2) + noise_baise2.expand_as(noise2)
            rgb = rgb + noise_f2
        alpha = self.end2(x)

        # out = x_in[:, 0:3]*(1-alpha) + rgb*alpha
        # return out, rgb, alpha

        fake_src, fake_mask = rgb, alpha
        fake_mask = (fake_mask + 1) * 0.5
        fake_mask_alpha = torch.cat([fake_mask, fake_mask, fake_mask], dim=1)
        merged_fake_src = fake_src * fake_mask_alpha + x_in[:, :3] * (1 - fake_mask_alpha)

        return merged_fake_src, fake_src, fake_mask


if __name__ == '__main__':
    import time

    input = torch.rand((1, 3, 512, 512))  # + 192 * 4
    mask = torch.rand((1, 1, 512, 512))

    net = ShallowNet(in_c=4, out_c=4)
    net.eval()
    start_time = time.perf_counter()
    net(input, mask)
    print("Time: {:.3f} ms".format((time.perf_counter() - start_time) * 1000))

    '''测试mac上的cpu时间'''
    if True:
        from utils.torch_utils.misc.base import init_weights
        net.apply(init_weights)
        from utils.torch_utils.engineer.onnx import do_export

        do_export('/root/group-trainee/ay/unet.onnx', net, shape=[512, 512], cuda=True
                  , checkpoint='/data_ssd/ay/checkpoint/neck_color/unet_V6/latest_net_0.pth')
