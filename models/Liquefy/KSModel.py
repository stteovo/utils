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


def get_gaussian_kernel(channels, kernel_size: int, sigma: float, device='cuda'):
    '''该函数对齐的是cv2.GausianBlur效果，结果上每像素误差 <= 1'''
    # 核大小必须是奇数
    assert kernel_size % 2 == 1, "Kernel size must be odd"

    # 生成高斯分布的坐标
    x = torch.arange(kernel_size) - (kernel_size - 1) / 2.0
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_1d = gauss / gauss.sum()

    # 创建二维高斯核，并转换为四维，适配卷积函数
    gs_kernel = torch.outer(kernel_1d, kernel_1d)
    gs_kernel = gs_kernel / gs_kernel.sum()
    gs_kernel = gs_kernel.expand(channels, 1, -1, -1)  # shape: [out_C, 1, H, W]

    '''
    输出nn.Conv2:
        1、padding_mode 不能用默认设置
        2、使用深度可分离卷积，in_C=1，groups=in_C
    
    '''
    gs_Conv2d = torch.nn.Conv2d(channels, channels, kernel_size, stride=1, padding=kernel_size // 2, groups=channels,
                                padding_mode='reflect', bias=False, device=device)
    gs_Conv2d.weight.data = gs_kernel.clone()
    gs_Conv2d.weight.requires_grad = False

    return gs_Conv2d


class MoveXYCVRemap(nn.Module):
    '''
    对齐cv2.remap效果
        1、+0.5：同步remap的像素对齐效果
        2？、归一化时/w，而不是/w-1，没想明白，可能的解释是，对齐opencv中不取右下两边轮廓源像素的值，下附opencv计算变换前后索引对应关系的源码
            sy = std::min(sy, iHiehgtSrc - 2);
            if (sx >= iWidthSrc - 1) {
                fx = 0, sx = iWidthSrc - 2;
            }
            看sx、sy的截断方式
    '''
    def __init__(self):
        super(MoveXYCVRemap, self).__init__()
        self.std_map = None

    def create_std_cv2style(self, b, h, w):
        if self.std_map is None or (
                self.std_map.size(0) != b or self.std_map.size(2) != h or self.std_map.size(3) != w):
            # 初始化uv_map，并归一化
            y_map, x_map = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5)
            y_map = y_map / h * 2 - 1
            x_map = x_map / w * 2 - 1

            # 得到cv2.remap形式的初始化uv_map   (b, h, w, 2)
            self.std_map = torch.stack([x_map, y_map]).permute(1, 2, 0).contiguous().float()
            self.std_map = self.std_map.unsqueeze(0).repeat(b, 1, 1, 1)
        return self.std_map

    def create_std_ytc(self, b, h, w):
        std_map = None
        if std_map is None or (std_map.size(0) != b or std_map.size(2) != h or std_map.size(3) != w):
            y_map, x_map = torch.meshgrid(torch.arange(h), torch.arange(w))
            y_map = y_map / (h - 1) * 2 - 1
            x_map = x_map / (w - 1) * 2 - 1
            std_map = torch.cat([x_map[None, :, :, None], y_map[None, :, :, None]], -1).float()
            std_map = torch.cat([std_map] * b, 0)
        return std_map

    def forward(self, image, real_uv_map):
        b, _, h, w = real_uv_map.size()
        std_map = self.create_std_cv2style(b, h, w).to(real_uv_map.device)
        uv_map = std_map - real_uv_map.permute(0, 2, 3, 1)
        image_out = F.grid_sample(image, uv_map, padding_mode='border', align_corners=False)

        return image_out
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


class KSModel_blur(nn.Module):
    def __init__(self, in_c=3, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, cbam=False, b_linear_up=True, tanh_flag=False,
                 is_training=True, b_blur=True, b_return_image=False):
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
        self.blur_layer = get_gaussian_kernel(2, 7, 1.0)

    def forward(self, *args, **kwargs):
        x_0 = torch.cat(args[:2], dim=1)
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

