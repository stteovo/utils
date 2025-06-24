import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F


def conv_pad(*args, **kwargs):
    # kwargs['padding_mode'] = 'circular'
    in_c, out_c, ks, s, p = args
    # p = p * 2
    return nn.Conv2d(in_c, out_c, ks, s, p, **kwargs)

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


def base_cbr(in_c, out_c, k, s):
    net = nn.Sequential(
        conv_pad(in_c, out_c, k, s, k//2, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )
    return net


def base_cr(in_c, out_c, k, s):
    net = nn.Sequential(
        conv_pad(in_c, out_c, k, s, k//2, bias=False),
        nn.ReLU(),
    )
    return net


class UpModel(nn.Module):
    def __init__(self, mid_c=128, post_c=8, bilinear=False):
        super(UpModel, self).__init__()
        self.head = nn.Sequential(
            nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1)
        )
        if bilinear:
            self.head = nn.Sequential(
                # nn.ConvTranspose2d(mid_c // 2, post_c * 2, 4, 2, 1),
                nn.UpsamplingBilinear2d(scale_factor=2),
                conv_pad(mid_c // 2, post_c * 2, 3, 1, 1),
                nn.ReLU(),

                # nn.ConvTranspose2d(post_c * 2, post_c, 4, 2, 1),
                nn.UpsamplingBilinear2d(scale_factor=2),
                conv_pad(post_c * 2, post_c, 3, 1, 1),
            )

    def __call__(self, x):
        return self.head(x)


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


class up_conv(nn.Module):
    def __init__(self, in_c, out_c, k, stride, pad):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=stride),
            nn.Conv2d(in_c, out_c, 3, 1, 1),
        )
        self.block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.up(x)
        return F.relu(x + self.block(x))


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


class KSModel(nn.Module):
    def __init__(self, in_c=4, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, merge=True, bilinear=False, using_noise=False):
        super(KSModel, self).__init__()
        self.merge = merge
        self.using_noise = using_noise
        u1 = ULink(mid_c, mid_c)
        for i in range(range_num-1):
            u1 = ULink(mid_c, mid_c, u1)
            #u3 = ULink(mid_c, mid_c, u2)
        u4 = ULink(pre_c, mid_c, u1)
        self.head1 = nn.Sequential(
            conv_pad(in_c, 12, 3, 2, 1),
            nn.ReLU(),
            ResBlock(12, pre_c),
            u4,
            nn.ReLU(),
            conv_pad(pre_c + mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
            conv_pad(mid_c // 2, mid_c // 2, 3, 1, 1),
            nn.ReLU(),
        )

        # self.rgb_decoder = UpModel(mid_c, post_c, bilinear)
        # self.alpha_decoder = UpModel(mid_c, post_c, bilinear=True)

        self.end_rgb = nn.Sequential(
            UpModel(mid_c, post_c, bilinear),
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, 3, 1, 1, 0),
            # nn.Tanh()
        )

        self.end_alpha = nn.Sequential(
            UpModel(mid_c, post_c, bilinear=True),
            # nn.Tanh()
        )

        self.scale_noise = ScaleNLayer(mid_c // 2, 64)
        self.scale_noise_out = ScaleNLayer(mid_c // 2, 64)

    def forward(self, x):
        inputs = x
        src, fake_neck_mask_binary, ref = x[:, :3], x[:, 3:4], x[:, 4:]
        x_f = self.head1(x)

        rgb = x_f.clone()
        if self.using_noise:
            xh, xw = x_f.size()[-2:]
            noise = torch.randn([x_f.shape[0], x_f.shape[1], xh, xw], device=x.device)
            noise_scale, noise_baise = self.scale_noise(x_f)
            noise_f = noise * noise_scale.expand_as(noise) + noise_baise.expand_as(noise)
            rgb = x_f + noise_f

        rgb = torch.tanh(self.end_rgb(rgb))

        alpha = self.end_alpha(x_f)

        fake_src_4 = torch.cat([rgb, alpha], 1)
        fake_src_4 = torch.clamp(fake_src_4, -1.0, 1.0)
        fake_src, fake_mask = fake_src_4[:, :3], fake_src_4[:, 3].unsqueeze(dim=1)

        merged_fake_src = fake_src
        if self.merge:
            fake_mask = (fake_mask + 1) * 0.5
            fake_mask_alpha = torch.cat([fake_mask, fake_mask, fake_mask], dim=1)
            merged_fake_src = fake_src * fake_mask_alpha + inputs[:, :3] * (1 - fake_mask_alpha)

        return merged_fake_src, fake_src, fake_mask

        # x = torch.clip(x, -1.0, 1.0)
        # fake_src, fake_mask = x[:, :3], x[:, 3]

        # return fake_src, fake_mask


class KSModel_Diff4(nn.Module):
    def __init__(self, in_c=4, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, b_linear_up=False):
        super(KSModel_Diff4, self).__init__()
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

        self.affine_head = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c, 1, 1, 0),
        )


    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)
        x = self.backbone(x_in)
        x = self.decoder(x)
        x = self.affine_head(x)

        diff, diff_mask_pre_sigmoid = x[:, :3], x[:, 3:]
        diff = diff.clip(-1, 1)
        diff_mask = torch.sigmoid(diff_mask_pre_sigmoid)
        res = x_in[:, :3] + diff * diff_mask

        res = res.clip(-1, 1)
        return diff, diff_mask_pre_sigmoid, res, diff_mask


class KSModel_Diff31(nn.Module):
    def __init__(self, in_c=4, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, b_linear_up=False, b_testing=False, b_diff=True):
        super(KSModel_Diff31, self).__init__()
        self.b_testing = b_testing
        self.b_diff = b_diff
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

        self.affine_3head = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c-1, 1, 1, 0),
        )

        self.affine_1head = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, 1, 1, 1, 0),
        )

    # def forward(self, x, mask):
    #     x_in = torch.cat([x, mask], dim=1)
    def forward(self, x_in):
        feat = self.backbone(x_in)
        feat = self.decoder(feat)

        diff = self.affine_3head(feat)
        diff = diff.clip(-1, 1)

        diff_mask_pre_sigmoid = self.affine_1head(feat)
        diff_mask = torch.sigmoid(diff_mask_pre_sigmoid)

        if self.b_testing:
            return diff, diff_mask

        if self.b_diff:
            res = x_in[:, :3] + diff * diff_mask
        else:
            res = diff * diff_mask + x_in[:, :3] * (1 - diff_mask)

        return diff, diff_mask_pre_sigmoid, res, diff_mask


class KSModel_Diff31_Head2(nn.Module):
    def __init__(self, in_c=4, pre_c=48, mid_c=128, post_c=8, out_c=4, range_num=3, b_linear_up=False, b_testing=False, b_diff=True):
        super(KSModel_Diff31_Head2, self).__init__()
        self.b_testing = b_testing
        self.b_diff = b_diff
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

        self.affine_3head = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, out_c-1, 1, 1, 0),
        )

        self.affine_1head_1st = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, 1, 1, 1, 0),
        )

        self.affine_1head_2nd = nn.Sequential(
            conv_pad(post_c, post_c, 3, 1, 1),
            nn.ReLU(),
            conv_pad(post_c, 1, 1, 1, 0),
        )

    # def forward(self, x, mask):
    #     x_in = torch.cat([x, mask], dim=1)
    def forward(self, x_in):
        feat = self.backbone(x_in)
        feat = self.decoder(feat)

        # 共用rgb
        rgb = self.affine_3head(feat)
        rgb = rgb.clip(-1, 1)

        # 第一个mask
        mask_pre_sigmoid_1st = self.affine_1head_1st(feat)
        mask_1st = torch.sigmoid(mask_pre_sigmoid_1st)

        # 第二个mask
        mask_pre_sigmoid_2nd = self.affine_1head_2nd(feat)
        mask_2nd = torch.sigmoid(mask_pre_sigmoid_2nd)

        if self.b_testing:
            return rgb, mask_1st, mask_2nd

        return rgb, mask_pre_sigmoid_1st, mask_pre_sigmoid_2nd, mask_1st, mask_2nd



if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # input = torch.rand((1, 7, 256, 512))
    # ks = KSModel(in_c=7, using_noise=True)

    input = torch.rand((1, 3, 768, 768))
    mask = torch.rand((1, 1, 768, 768))
    net = KSModel_Diff4(in_c=4)
    output = net(input, mask)


    # ks.load_state_dict(torch.load('/root/wssy/CheckPointsTemp/train_AE_balance_split_lab_hard_large_1536_pretrain_from_v9/latest_net_0.pth'))
    # torch.onnx.export(ks, (input,), '/root/acne_face_(1536+192*4,1280)_beta_0402_v9_c.onnx', input_names=['data'], output_names=['output'],
    #                   opset_version=11)
    # ks.cuda()
    input = input
    ks(input)
    from thop import profile

    p, m = profile(ks, (input,))
    print(p / 1e6, m / 1e6)