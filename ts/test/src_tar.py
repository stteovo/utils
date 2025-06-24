import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import os
import glob
import numpy as np
import pathlib
import shutil

def to_uint8(image):
    return np.clip(image, 0, 255).astype(np.uint8)


# 基于src和dst构建中间的线性光图层， src + 线性光图层 -> dst
def build_linear_layer(src, dst):
    return to_uint8((dst.astype(np.float32) - src.astype(np.float32)) * 0.5 + 128)


# 基于src和dst构建中间的柔光图层， src + 柔光图层 -> dst
def build_soft_light_layer(src, dst):
    diff_img = build_linear_layer(src, dst)

    src_norm = src.astype(np.float32) / 255.0
    dst_norm = dst.astype(np.float32) / 255.0
    diff_norm = diff_img.astype(np.float32) / 255.0

    mask_by_src = (src_norm == 0.0) | (src_norm == 1.0) | (src_norm == dst_norm)
    mask_less_0_5 = (diff_norm < 0.5) & (~mask_by_src)
    mask_other_wise = (diff_norm >= 0.5) & (~mask_by_src)

    soft_light_norm = diff_norm.copy()
    soft_light_norm[mask_by_src] = 0.5
    soft_light_norm[mask_less_0_5] = (dst_norm[mask_less_0_5] -
                                      src_norm[mask_less_0_5] * src_norm[mask_less_0_5]) / \
                                     (2.0 * src_norm[mask_less_0_5] -
                                      2.0 * src_norm[mask_less_0_5] * src_norm[mask_less_0_5])

    soft_light_norm[mask_other_wise] = (dst_norm[mask_other_wise] +
                                        np.sqrt(src_norm[mask_other_wise]) -
                                        2.0 * src_norm[mask_other_wise]) / \
                                       (2.0 * np.sqrt(src_norm[mask_other_wise]) -
                                        2.0 * src_norm[mask_other_wise])

    return to_uint8(soft_light_norm * 255)

if __name__ == '__main__':
    root = '/root/group-inspect2-data/证件照换装/中间结果/剥离脖子阴影/原图'
    src_dir = os.path.join(root, 'src')
    dst_dir = os.path.join(root, 'dst')
    save_dir = os.path.join(root,'gray_inverse')

    for sub_dir in [src_dir, dst_dir, save_dir]:
        os.makedirs(sub_dir, exist_ok=True)

    for fn in os.listdir(src_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post!= '':
            src = cv2.imread(os.path.join(src_dir, pre + '.png'))
            dst = cv2.imread(os.path.join(dst_dir, pre + '.jpg'))

            grayLayer = build_soft_light_layer(src, dst)
            save_fn = os.path.join(save_dir, pre + '.png')
            cv2.imwrite(save_fn, grayLayer)