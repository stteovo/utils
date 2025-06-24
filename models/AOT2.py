import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def conv_pad(*args, **kwargs):
    # kwargs['padding_mode'] = 'circular'
    in_c, out_c, ks, s, p = args
    # p = p * 2
    return nn.Conv2d(in_c, out_c, ks, s, p, **kwargs)


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True))

class AOT2(BaseNetwork):
    def __init__(self, rates=(1, 2, 4, 8), block_num=8, out_c=3):  # 1046
        super(AOT2, self).__init__()
        self.rates = rates
        self.block_num = block_num

        self.encoder = nn.Sequential(
            nn.Conv2d(4, 64, 7, padding=3, padding_mode='reflect'),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 255, 4, stride=2, padding=1),
            nn.ReLU(True)
        )

        self.middle = nn.Sequential(*[myAOTBlock(255, self.rates) for _ in range(self.block_num)])

        self.decoder = nn.Sequential(
            UpConv(255, 128),
            nn.ReLU(True),
            UpConv(128, 64),
            nn.ReLU(True),
            nn.Conv2d(64, out_c, 3, stride=1, padding=1)
        )


        self.init_weights()

    def forward(self, x, mask):
        x_in = x[:, :3].clone()
        x = torch.cat([x, mask], dim=1)
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)

        fake_src, mask_pre_sigmoid = x[:, :3], x[:, 3:]
        fake_src = torch.tanh(fake_src)
        mask = torch.sigmoid(mask_pre_sigmoid)
        res = fake_src * mask + x_in * (1 - mask)

        return fake_src, mask_pre_sigmoid, res


class myAOTBlock(nn.Module):
    def __init__(self, dim, rates):
        super(myAOTBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                'block{}'.format(str(i).zfill(2)),
                nn.Sequential(
                    nn.Conv2d(dim, dim//len(rates), 3, dilation=rate, padding=rate, padding_mode='reflect'),
                    nn.ReLU(True)))
        self.fuse = nn.Conv2d(dim, dim, 3, dilation=1, padding=1, padding_mode='reflect')
        self.gate = nn.Conv2d(dim, dim, 3, dilation=1, padding=1, padding_mode='reflect')

    def forward(self, x):
        out = [self.__getattr__(f'block{str(i).zfill(2)}')(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = self.gate(x)
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask

if __name__ == '__main__':
    net = AOT2(rates=(1, 4, 8), block_num=2, out_c=4)
    mask = torch.randn(1, 1, 512, 512)
    img = torch.randn(1, 3, 512, 512)

    out = net(img, mask)
    pass