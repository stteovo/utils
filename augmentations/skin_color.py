import numpy as np
import cv2
import os
from utils.visulization.realTimeVisual import visualize


def add_3dlut(img, lut):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lut = cv2.cvtColor(lut, cv2.COLOR_BGR2RGB)

    red = img[:, :, 0] / 255
    green = img[:, :, 1] / 255
    blue = img[:, :, 2] / 255 * 63.0
    quad1y = np.floor(np.floor(blue) / 8.0)
    quad1x = np.floor(blue) - quad1y * 8.0
    quad2y = np.floor(np.ceil(blue) / 8.0)
    quad2x = np.ceil(blue) - quad2y * 8.0
    texpos1x = (512 * (quad1x * 0.125 + 0.0009765625 + (0.125 - 0.001953125) * red)).astype(int)
    texpos1y = (512 * (quad1y * 0.125 + 0.0009765625 + (0.125 - 0.001953125) * green)).astype(int)
    texpos2x = (512 * (quad2x * 0.125 + 0.0009765625 + (0.125 - 0.001953125) * red)).astype(int)
    texpos2y = (512 * (quad2y * 0.125 + 0.0009765625 + (0.125 - 0.001953125) * green)).astype(int)
    color1 = lut[texpos1y, texpos1x]
    color2 = lut[texpos2y, texpos2x]
    alpha = blue - blue.astype(int)
    alpha = np.dstack([alpha, alpha, alpha])
    out = np.clip(color1 * alpha + color2 * (1 - alpha), 0, 255)
    out = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return out


def alpha_merge(foreground, background, alpha):
    alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)
    return img_out.clip(0, 255).astype(np.uint8)


def test_add_3dlut(img, lut_path='/root/group-trainee/ay/mines/datasets/look_up_table/filter/filter 41.png'):
    lut = cv2.imread(lut_path, -1)
    img = add_3dlut(img, lut)

    return img


if __name__ == '__main__':
    # dir = '/root/group-trainee/ay/tmp/lut_model/'
    # img_neck = cv2.imread('/root/group-trainee/ay/version1/dataset/a_online512/train/complete/GT/33.png', -1)[:, :, :3]
    # # background = np.ones_like(img_neck[:, :, :3]) * 255
    # # img_neck = alpha_merge(img_neck[:, :, :3], background, img_neck[:, :, 3])
    # a = img_neck
    #
    # res_dir = '/root/group-trainee/ay/tmp/results/'
    # if not os.path.exists(res_dir):
    #     os.makedirs(res_dir)
    # for fn in os.listdir(dir):
    #     fp = os.path.join(dir, fn)
    #     lut = cv2.imread(fp, -1)
    #     img_out = add_3dlut(a, lut)
    #
    #     res_fp = os.path.join(res_dir, fn)
    #     cv2.imwrite(res_fp, img_out)
    #     pass

    visualize(test_add_3dlut)



