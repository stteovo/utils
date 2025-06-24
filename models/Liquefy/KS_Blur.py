import torch, math, cv2
import torch.nn as nn
import torch.nn.functional as F
from kornia.utils import create_meshgrid
import numpy as np
from torch.cuda.amp import autocast



class CBAM(nn.Module):
    def __init__(self, in_channel, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # 通道注意力机制
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.mlp = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=in_channel // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=in_channel // reduction, out_features=in_channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # 空间注意力机制
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=kernel_size // 2, bias=False)

    def forward(self, x):
        # 通道注意力机制
        maxout = self.max_pool(x)
        maxout = self.mlp(maxout.reshape(maxout.size(0), -1))
        avgout = self.avg_pool(x)
        avgout = self.mlp(avgout.reshape(avgout.size(0), -1))
        channel_out = self.sigmoid(maxout + avgout)
        channel_out = channel_out.reshape(x.size(0), x.size(1), 1, 1)
        channel_out = channel_out * x
        # 空间注意力机制
        max_out, _ = torch.max(channel_out, dim=1, keepdim=True)
        mean_out = torch.mean(channel_out, dim=1, keepdim=True)
        out = torch.cat((max_out, mean_out), dim=1)
        out = self.sigmoid(self.conv(out))
        out = out * channel_out
        return out


def conv_pad(*args, **kwargs):
    # kwargs['padding_mode'] = 'circular'
    in_c, out_c, ks, s, p = args
    # p = p * 2
    return nn.Conv2d(in_c, out_c, ks, s, p, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, cbam=False):
        super(ResBlock, self).__init__()
        self.conv_head = conv_pad(in_c, out_c, 3, 2, 1)
        self.conv_1 = conv_pad(out_c, out_c, 3, 1, 1)
        self.conv_2 = conv_pad(out_c, out_c, 3, 1, 1)
        if cbam:
            self.ca = CBAM(out_c)
            self.use_ca = True
            # print("channel attention: CBAM")
        else:
            self.ca = None
            self.use_ca = False
            # print("channel attention: None")

    def forward(self, x):
        x = self.conv_head(x)
        x_1 = F.relu(x)
        x_2 = F.relu(self.conv_1(x_1))
        x_3 = self.conv_2(x_2)
        if self.use_ca:
            x_4 = self.ca(x_3)
            return x + x_4
        return x + x_3


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


class UpRes(nn.Module):
    def __init__(self, out_c):
        super(UpRes, self).__init__()
        self.model = nn.ConvTranspose2d(out_c, out_c // 2, 4, 2, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_c, out_c // 2, 1, 1),
            nn.UpsamplingNearest2d(scale_factor=2)
        )

    def forward(self, x):
        return self.fc(x) + F.relu(self.model(x))


class ULink(nn.Module):
    def __init__(self, in_c, out_c, mid=None, cbam=False):
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
        self.up = nn.ConvTranspose2d(out_c, out_c // 2, 4, 2, 1, bias=False)
        if cbam:
            self.ca = CBAM(out_c)
            self.use_ca = True
            # print("channel attention: CBAM")
        else:
            self.ca = None
            self.use_ca = False
            # print("channel attention: None")

    def forward(self, x):
        x_half = self.down_block(x)
        if self.mid is not None:
            x_half = self.mid(x_half)
        x_1 = F.relu(x_half)
        x_2 = F.relu(self.conv(x_1))
        if self.use_ca:
            x_3 = self.ca(x_2)
            x_4 = self.up(x_3)
            return torch.cat([x, x_4], 1)
        x_3 = self.up(x_2)
        return torch.cat([x, x_3], 1)


def base_cbr(in_c, out_c, k, s):
    net = nn.Sequential(
        conv_pad(in_c, out_c, k, s, k // 2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )
    return net


def base_cr(in_c, out_c, k, s):
    net = nn.Sequential(
        conv_pad(in_c, out_c, k, s, k // 2, bias=False),
        nn.ReLU(),
    )
    return net


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=2):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def inverse_sample_grid(grid, std_map=None, max_iters=10, min_error_decrease=1e2):
    """
    inverse sample grid
    Args:
        grid: sample grid to be inverse
        identity: identity sample grid, if None, create based on grid shape

    Returns:
        inverted sample grid
    """

    inv = std_map.clone()
    grid = grid.permute(0, 3, 1, 2)
    # TODO: adjust learning rate here
    prev_error = torch.inf
    curr_error = 1e9
    iter = 0
    while iter < max_iters and (prev_error - curr_error) > min_error_decrease:
        diff = std_map - torch.nn.functional.grid_sample(
            grid, inv, mode='bilinear', padding_mode='border', align_corners=False
        ).permute(0, 2, 3, 1)
        inv = inv + diff
        error = diff.norm(dim=-1).sum()
        prev_error = curr_error
        curr_error = error
        iter += 1
    return inv


def remap_values(remapping, x):
    index = torch.bucketize(x.ravel(), remapping[0])
    return remapping[1][index].reshape(x.shape)


class Remap(nn.Module):
    def __init__(self):
        super(Remap, self).__init__()
        self.std_map = None

    def create_std(self, b, h, w):
        if self.std_map is None or (
                self.std_map.size(0) != b or self.std_map.size(2) != h or self.std_map.size(3) != w):
            y_map, x_map = torch.meshgrid(torch.arange(h), torch.arange(w))
            y_map = y_map / (h - 1) * 2 - 1
            x_map = x_map / (w - 1) * 2 - 1
            self.std_map = torch.cat([x_map[None, :, :, None], y_map[None, :, :, None]], -1).float()
            self.std_map = torch.cat([self.std_map] * b, 0)
        return self.std_map

    def forward(self, image, yx_res, return_list=True):
        b, _, h, w = yx_res.size()
        std_map = self.create_std(b, h, w).to(yx_res.device)
        indxe_map = std_map + yx_res.permute(0, 2, 3, 1)
        # out = F.grid_sample(image, indxe_map, align_corners=True)
        out = remap_values(indxe_map, image)
        return out


class MoveXYCVRemap(nn.Module):
    def __init__(self):
        super(MoveXYCVRemap, self).__init__()
        self.std_map = None

    def create_std(self, b, h, w):
        if self.std_map is None or (
                self.std_map.size(0) != b or self.std_map.size(2) != h or self.std_map.size(3) != w):
            y_map, x_map = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5)
            # y_map = y_map + 1
            # x_map = x_map + 1
            y_map = y_map / h
            x_map = x_map / w

            y_map = y_map * 2 - 1
            x_map = x_map * 2 - 1

            self.std_map = torch.cat([x_map[None, :, :, None], y_map[None, :, :, None]], -1).float()
            self.std_map = torch.cat([self.std_map] * b, 0)
        return self.std_map

    def forward(self, image, yx_res):
        b, _, h, w = yx_res.size()
        std_map = self.create_std(b, h, w).to(yx_res.device)
        indxe_map = std_map - yx_res.permute(0, 2, 3, 1)
        out = F.grid_sample(image, indxe_map, padding_mode='border', align_corners=False)
        # out = grid_sample_v2(image, indxe_map)

        return out


class MoveXY(nn.Module):
    def __init__(self):
        super(MoveXY, self).__init__()
        self.std_map = None

    def create_std(self, b, h, w):
        if self.std_map is None or (
                self.std_map.size(0) != b or self.std_map.size(2) != h or self.std_map.size(3) != w):
            y_map, x_map = torch.meshgrid(torch.arange(h), torch.arange(w))
            # y_map = y_map + 1
            # x_map = x_map + 1
            y_map = y_map / (h - 1) * 2 - 1
            x_map = x_map / (w - 1) * 2 - 1
            self.std_map = torch.cat([x_map[None, :, :, None], y_map[None, :, :, None]], -1).float()
            self.std_map = torch.cat([self.std_map] * b, 0)
        return self.std_map

    def forward(self, image, yx_res):
        b, _, h, w = yx_res.size()
        std_map = self.create_std(b, h, w).to(yx_res.device)
        indxe_map = std_map + yx_res.permute(0, 2, 3, 1)
        out = F.grid_sample(image, indxe_map, align_corners=True)
        # out = grid_sample_v2(image, indxe_map)

        return out


class KSModel(nn.Module):
    def __init__(self, in_c=3, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, cbam=True, tanh_flag=False,
                 is_training=True):
        super(KSModel, self).__init__()

        self.is_training = is_training

        u1 = ULink(mid_c, mid_c, cbam=cbam)
        for i in range(range_num - 1):
            u1 = ULink(mid_c, mid_c, u1, cbam=cbam)
        u4 = ULink(pre_c, mid_c, u1, cbam=cbam)

        self.head = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c, cbam=cbam),
            u4,
            nn.ReLU(),
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c, post_c, 4, 2, 1),
        )

        self.head2 = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c, cbam=cbam),
            u4,
            nn.ReLU(),
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1)
        )

        self.end3 = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, post_c, 1, 1, 0),
            nn.ReLU(),
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c, 1, 1, 0),
        )
        self.endremap = nn.Sequential(
            conv_pad(post_c, post_c, 3, 2, 1),
            nn.ReLU(),
            conv_pad(post_c, 3, 1, 1, 0),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.tanh = tanh_flag
        self.move_xy = MoveXY()
        self.move_xy_for_feat = MoveXY()
        self.down = nn.AvgPool2d(2, 2, 0)
        self.blur_layer = get_gaussian_kernel(15, 3, 2).cuda()

    def forward(self, x_0):
        x_0_d = self.down(x_0)
        x_feat = self.head(x_0_d)
        decoded = self.endremap(x_feat)
        yx_alphamatte = torch.sigmoid(decoded[:, 2:])
        yx_flow = torch.tanh(decoded[:, :2]) * yx_alphamatte
        remap_image = self.move_xy(x_0[:, :3], yx_flow, return_list=False)

        if self.is_training:
            return (remap_image, yx_flow, yx_alphamatte)

        return yx_flow


class KSModelCVRemap(nn.Module):
    def __init__(self, in_c=3, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, cbam=True, tanh_flag=False,
                 is_training=True):
        super(KSModelCVRemap, self).__init__()

        self.is_training = is_training

        u1 = ULink(mid_c, mid_c, cbam=cbam)
        for i in range(range_num - 1):
            u1 = ULink(mid_c, mid_c, u1, cbam=cbam)
        u4 = ULink(pre_c, mid_c, u1, cbam=cbam)

        self.head = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c, cbam=cbam),
            u4,
            nn.ReLU(),
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c, post_c, 4, 2, 1),
        )

        self.head2 = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c, cbam=cbam),
            u4,
            nn.ReLU(),
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1)
        )

        self.end3 = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, post_c, 1, 1, 0),
            nn.ReLU(),
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c, 1, 1, 0),
        )
        self.endremap = nn.Sequential(
            conv_pad(post_c, post_c, 3, 2, 1),
            nn.ReLU(),
            conv_pad(post_c, 3, 1, 1, 0),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.tanh = tanh_flag
        self.down = nn.AvgPool2d(2, 2, 0)
        self.blur_layer = get_gaussian_kernel(15, 3, 2).cuda()
        self.move_xy = MoveXYCVRemap()

    def forward(self, x_0):
        x_0_d = self.down(x_0)
        x_feat = self.head(x_0_d)
        decoded = self.endremap(x_feat)


        yx_alphamatte = torch.sigmoid(decoded[:, 2:])
        yx_flow = torch.tanh(decoded[:, :2]) * yx_alphamatte
        remap_image = self.move_xy(x_0[:, :3], yx_flow)
        # remap_image = batch_remap(x_0[:, :3], yx_flow)

        if self.is_training:
            return (remap_image, yx_flow, yx_alphamatte)

        return yx_flow


class KSModel_blur_down(nn.Module):
    def __init__(self, down_scale=2, in_c=3, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, cbam=False, b_linear_up=True, tanh_flag=False, is_training=True):
        super(KSModel_blur_down, self).__init__()

        self.is_training = is_training
        self.tanh = tanh_flag

        u1 = ULink(mid_c, mid_c)
        for i in range(range_num-1):
            u1 = ULink(mid_c, mid_c, u1)
            #u3 = ULink(mid_c, mid_c, u2)
        u4 = ULink(pre_c, mid_c, u1)
        self.backbone = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c),
            u4,
            nn.ReLU()
        )

        # u1 = ULink(mid_c, mid_c, cbam=cbam)
        # for i in range(range_num - 1):
        #     u1 = ULink(mid_c, mid_c, u1, cbam=cbam)
        # u4 = ULink(pre_c, mid_c, u1, cbam=cbam)
        #
        # self.backbone = nn.Sequential(
        #     conv_pad(in_c, 12, 3, 2, 1),
        #     nn.ReLU(),
        #     ResBlock(12, pre_c),
        #     u4,
        #     nn.ReLU()
        # )
        self.decoder = nn.Sequential(
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, post_c * 2, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            conv_pad(post_c * 2, post_c, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ) if b_linear_up else nn.Sequential(
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1)
        )

        self.head_remap = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c, 1, 1, 0),
            nn.UpsamplingBilinear2d(scale_factor=down_scale)
        )
        self.move_xy = MoveXYCVRemap()
        self.blur_layer = get_gaussian_kernel(9, 1, 2).cuda()
        self.down = nn.AvgPool2d(down_scale, down_scale, 0)

    # def forward(self, x_1, x_2):
    #     with autocast(dtype=torch.float16):
    #         x_0 = torch.cat([x_1, x_2], dim=1)
    def forward(self, x_0):
        with autocast(dtype=torch.float16):
            x_0 = self.down(x_0)
            feat = self.backbone(x_0)
            feat = self.decoder(feat)
            decoded = self.head_remap(feat)

            '''mask处理'''
            yx_alphamatte = torch.tanh(decoded[:, 2:])
            yx_alphamatte = (yx_alphamatte + 1) * 0.5

            yx_flow = torch.tanh(decoded[:, :2]) * yx_alphamatte
            # yx_flow = self.blur_layer(yx_flow)

            if not self.is_training:
                return yx_flow

            remap_image = self.move_xy(x_0[:, :3], yx_flow)

            return remap_image, yx_flow, yx_alphamatte
            # return remap_image, yx_alphamatte


class KSModel_blur(nn.Module):
    def __init__(self, in_c=3, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, cbam=False,
                 b_linear_up=True, tanh_flag=False, is_training=True, b_blur=True, b_return_image=False):
        super(KSModel_blur, self).__init__()

        self.is_training = is_training
        self.b_blur = b_blur
        self.tanh = tanh_flag
        self.b_return_image = b_return_image

        u1 = ULink(mid_c, mid_c)
        for i in range(range_num-1):
            u1 = ULink(mid_c, mid_c, u1)
            #u3 = ULink(mid_c, mid_c, u2)
        u4 = ULink(pre_c, mid_c, u1)
        self.backbone = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c),
            u4,
            nn.ReLU()
        )

        # u1 = ULink(mid_c, mid_c, cbam=cbam)
        # for i in range(range_num - 1):
        #     u1 = ULink(mid_c, mid_c, u1, cbam=cbam)
        # u4 = ULink(pre_c, mid_c, u1, cbam=cbam)
        #
        # self.backbone = nn.Sequential(
        #     conv_pad(in_c, 12, 3, 2, 1),
        #     nn.ReLU(),
        #     ResBlock(12, pre_c),
        #     u4,
        #     nn.ReLU()
        # )
        self.decoder = nn.Sequential(
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, post_c * 2, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReLU(),
            conv_pad(post_c * 2, post_c, 3, 1, 1),
            nn.UpsamplingBilinear2d(scale_factor=2),
        ) if b_linear_up else nn.Sequential(
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1)
        )

        self.head_remap = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c, 1, 1, 0),
            # nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.move_xy = MoveXYCVRemap()
        self.blur_layer = get_gaussian_kernel(9, 1, 2).cuda()

    def forward(self, x_1, x_2):
        x_0 = torch.cat([x_1, x_2], dim=1)
    # def forward(self, x_0):
        feat = self.backbone(x_0)
        feat = self.decoder(feat)
        decoded = self.head_remap(feat)

        '''mask处理'''
        yx_alphamatte = torch.tanh(decoded[:, 2:])
        yx_alphamatte = (yx_alphamatte + 1) * 0.5

        yx_flow = torch.tanh(decoded[:, :2]) * yx_alphamatte
        if self.b_blur:
            yx_flow = self.blur_layer(yx_flow)

        if not self.is_training:
            if self.b_return_image:
                remap_image = self.move_xy(x_0[:, :3], yx_flow)
                return remap_image
            else:
                return yx_flow, yx_alphamatte

        remap_image = self.move_xy(x_0[:, :3], yx_flow)

        return remap_image, yx_flow, yx_alphamatte
        # return remap_image, yx_alphamatte



def to_512_rgb_c4_float(img, fp16):
    if img.ndim == 2:
        img_out = np.expand_dims(img, axis=(0, 1))
        img_out = img_out.astype(np.float32) / 127.5 - 1
    else:
        if img.shape[2] == 3:
            img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif img.shape[2] == 4:
            img_out = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.shape[2] == 5:
            img_reverse = img[:, :, :3][:, :, ::-1]
            img_out = np.concatenate((img_reverse, img[:, :, 3:]), axis=2)
        img_out = np.expand_dims(img_out.transpose(2, 0, 1), axis=0)
        img_out = img_out.astype(np.float32) / 127.5 - 1

    return img_out.astype(np.float16) if fp16 else img_out

if __name__ == '__main__':
    data = torch.randn(1, 4, 768, 768)
    net = KSModel_blur(in_c=4, out_c=3, tanh_flag=True, cbam=False)
    output = net(data)


    hair = cv2.imread('/root/1.png', -1)
    person_mask = cv2.imread('/root/0.png', -1)
    test_src1 = to_512_rgb_c4_float(hair, fp16=False)
    test_src2 = to_512_rgb_c4_float(person_mask, fp16=False)

    net = KSModel_blur(in_c=4, tanh_flag=True, cbam=False).cuda(1)
    state_dict = torch.load('/data_ssd/ay/checkpoint/new512/Hair/wo_cloth/long_768/latest_net_0.pth')
    net.load_state_dict(state_dict)
    net.eval()
    test_src1, test_src2 = torch.from_numpy(test_src1).to(torch.float32).cuda(1), torch.from_numpy(test_src2).to(torch.float32).cuda(1)
    ort_output, ort_output_mask = net(test_src1, test_src2)

    ort_output = np.squeeze(ort_output.detach().cpu().numpy(), axis=0)
    ort_output = ((ort_output + 1) * 127.5).transpose(1, 2, 0)
    # ort_output = ort_output[..., ::-1]
    ort_output = ort_output.astype(np.uint8)
    cv2.imwrite('/root/3.png', ort_output)

    # ort_output = np.squeeze(ort_output.detach().cpu().numpy(), axis=0)
    # ort_output = ort_output.transpose(1, 2, 0)
    # ort_output = ort_output[..., ::-1]
    # ort_output = ort_output.astype(np.float32)
    # ort_output = cv2.GaussianBlur(ort_output, (9, 9), 3).astype(np.float32)

    ort_output_mask = np.squeeze(ort_output_mask.detach().cpu().numpy(), axis=(0, 1))
    ort_output_mask = (ort_output_mask * 255).astype(np.uint8)
    pass

