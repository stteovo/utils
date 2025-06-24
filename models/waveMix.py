import torch
import torch.nn as nn
import wavemix
from wavemix import Level1Waveblock, Level2Waveblock, Level3Waveblock, DWTForward


class WaveMixModule(nn.Module):
    def __init__(
            self,
            *,
            depth,
            mult=2,
            ff_channel=16,
            final_dim=16,
            dropout=0.,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, int(final_dim / 2), 3, 1, 1),
            nn.Conv2d(int(final_dim / 2), final_dim, 3, 2, 1),
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Level1Waveblock(mult=mult, ff_channel=ff_channel, final_dim=final_dim, dropout=dropout))
            # self.layers.append(Waveblock(mult = mult, ff_channel = ff_channel, final_dim = final_dim, dropout = dropout))

        self.depthconv = nn.Sequential(
            nn.Conv2d(final_dim, final_dim, 5, groups=final_dim, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(final_dim)
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(final_dim * 2, int(final_dim / 2), 4, stride=2, padding=1),
            nn.BatchNorm2d(int(final_dim / 2))
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(int(final_dim / 2) + 3, 3, 1),
        )

    def forward(self, img, mask):

        x = torch.cat([img, mask], dim=1)

        x = self.conv(img)

        skip1 = x

        for attn in self.layers:
            x = attn(x) + x

        x = self.depthconv(x)

        x = torch.cat([x, skip1], dim=1)  # skip connection

        x = self.decoder1(x)

        x = torch.cat([x, img], dim=1)  # skip connection

        x = self.decoder2(x)

        return x


class WavePaint(nn.Module):
    def __init__(
            self,
            *,
            num_modules=1,
            blocks_per_module=7,
            mult=4,
            ff_channel=16,
            final_dim=16,
            dropout=0.,

    ):
        super().__init__()

        self.wavemodules = nn.ModuleList([])
        for _ in range(num_modules):
            self.wavemodules.append(
                WaveMixModule(depth=blocks_per_module, mult=mult, ff_channel=ff_channel, final_dim=final_dim,
                              dropout=dropout))

    def forward(self, img, mask):
        x = img

        for module in self.wavemodules:
            x = module(x, 1 - mask) + x

        x = x * mask + img

        return x


if __name__ == '__main__':
    import time

    net = WavePaint(num_modules=8,
            blocks_per_module=4,
            mult=4,
            ff_channel=128,
            final_dim=128,
            dropout=0.5,).cuda()

    '''测试在gpu上的耗时'''
    input = torch.randn(1, 3, 512, 512).cuda()
    mask = torch.randn(1, 1, 512, 512).cuda()
    start_time = time.perf_counter()
    output = net(input, mask)
    print("Time: {:.3f} ms".format((time.perf_counter() - start_time) * 1000))

    '''测试mac上的cpu时间'''
    if True:
        from utils.torch_utils.misc.base import init_weights
        net.apply(init_weights)
        from utils.torch_utils.engineer.onnx import do_export

        do_export('/root/group-trainee/ay/wave.onnx', net, shape=[512, 512], checkpoint=None, cuda=True)




