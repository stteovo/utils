'''
@ref https://github.com/shrutiphutke/Blind_Omni_Wav_Net
Blind Image Inpainting via Omni-dimensional Gated Attention and Wavelet Queries [CVPRW-23]
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_wavelets import DWTForward, DWTInverse
import matplotlib.pyplot as plt
import torch.autograd


class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d(attention_channel)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)


class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes // groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)

        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)

        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))

        output = output * filter_attention

        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)


class od_attention(nn.Module):
    def __init__(self, channels):
        super(od_attention, self).__init__()

        self.od_conv = ODConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        od_out = self.od_conv(x)

        out = self.conv(x)
        attention = F.gelu(od_out)

        return out * attention


class SSL(nn.Module):
    def __init__(self, channels):
        super(SSL, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 1, dilation=1)
        self.conv5 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 5, dilation=5)
        self.conv7 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 7, dilation=7)
        self.conv9 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels,
                               bias=False)  # conv_block_my(channels, channels, kernel_size = 3, stride=1, padding = 9, dilation=9)

        self.conv_cat = nn.Conv2d(channels * 4, channels, kernel_size=3, padding=1, groups=channels,
                                  bias=False)  # conv_block_my(channels*4, channels, kernel_size = 3, stride = 1, padding = 1, dilation=1)

    def forward(self, x):
        aa = DWTForward(J=1, mode='zero', wave='db3').cuda()
        yl, yh = aa(x)

        yh_out = yh[0]
        ylh = yh_out[:, :, 0, :, :]
        yhl = yh_out[:, :, 1, :, :]
        yhh = yh_out[:, :, 2, :, :]

        conv_rec1 = self.conv5(yl)
        conv_rec5 = self.conv5(ylh)
        conv_rec7 = self.conv7(yhl)
        conv_rec9 = self.conv9(yhh)

        cat_all = torch.stack((conv_rec5, conv_rec7, conv_rec9), dim=2)
        rec_yh = []
        rec_yh.append(cat_all)

        ifm = DWTInverse(wave='db3', mode='zero').cuda()
        Y = ifm((conv_rec1, rec_yh))

        return Y


class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 2, kernel_size=1, bias=False)
        self.query = SSL(channels)
        self.qkv_conv = nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=1, groups=channels * 2, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        k, v = self.qkv_conv(self.qkv(x)).chunk(2, dim=1)
        q = self.query(x)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Inpainting(nn.Module):
    def __init__(self, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48 // 3, 96 // 3, 192 // 3, 384 // 3],
                 num_refinement=4,
                 expansion_factor=2.66):
        super(Inpainting, self).__init__()

        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])

        # self.res_layers = nn.ModuleList([nn.Sequential(*[TransformerBlock_Res(
        #     num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
        #                                zip(num_blocks, num_heads, [384//3, 384//3,384//3,384//3])])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])

        self.skips = nn.ModuleList([od_attention(num_ch) for num_ch in list(reversed(channels))[1:]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)
        out_enc1 = self.encoders[0](fo)
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), self.skips[0](out_enc3)], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), self.skips[1](out_enc2)], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), self.skips[2](out_enc1)], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr)
        return out

if __name__ == '__main__':
    data = torch.randn(1, 3, 512, 512).cuda()
    net = Inpainting().cuda()

    import time
    net.eval()
    start_time = time.perf_counter()
    out = net(data)
    print("Time: {:.3f} ms".format((time.perf_counter() - start_time) * 1000))

    '''测试mac上的cpu时间'''
    if True:
        from utils.torch_utils.misc.base import init_weights
        net.apply(init_weights)

        # from utils.torch_utils.engineer.onnx import do_export, do_export_batch
        # do_export('/root/group-trainee/ay/arcnet_b1_n1_e10_1223.onnx', net, in_c=4, shape=[512, 512], cuda=False
        #           , checkpoint='/data_ssd/ay/checkpoint/neck_color_noise/artunet_b1_n5_e30/latest_net_0.pth')

        # 测试能否导出成onnx
        onnx_fp = 'models/test.onnx'
        torch.onnx.export(
            net,  # 要导出的模型
            data,  # 一个示例输入张量，用于跟踪模型的执行
            onnx_fp,  # 保存ONNX模型的路径
            export_params=True,  # 是否导出参数
            opset_version=11,  # 使用的ONNX算子版本
            do_constant_folding=True,  # 是否进行常量折叠优化
            input_names=['input'],  # 输入张量的名称
            output_names=['output'],  # 输出张量的名称
            dynamic_axes={  # 动态轴（可选，适用于批次大小等动态变化的维度）
                'input': {0: 'batch_size'},  # 输入的第一个维度是批次大小
                'output': {0: 'batch_size'}  # 输出的第一个维度是批次大小
            }
        )