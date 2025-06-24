import os, cv2
import numpy as np
from tqdm import tqdm

from lollipop.parsing.human_parsing.HumanParsing import HumanParsing


def morph(mask, kernel_szie=10, operation=cv2.MORPH_DILATE, shape=cv2.MORPH_RECT, binary=False):
    # 增大假脖子区域，遮挡阴影
    if shape == 'auto':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(kernel_szie, kernel_szie))
        t, b = kernel_szie // 3 - 1, kernel_szie * 2 // 3 + 1
        kernel[0:t] = 0
        kernel[b:] = 0
    else:
        kernel = cv2.getStructuringElement(shape, ksize=(kernel_szie, kernel_szie))
    mask = cv2.morphologyEx(mask, operation, kernel, iterations=1)
    if binary:
        mask = np.where(mask > 50, 255, 0)

    return mask.astype(np.uint8)


def alpha_merge(foreground, background, alpha):
    if len(alpha.shape) == 2:
        alpha = cv2.merge([alpha, alpha, alpha])

    alpha = alpha.astype(np.float64) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)

    return img_out.clip(0, 255).astype(np.uint8)


def get_face(img_path, hp_model):
    image = cv2.imread(img_path, -1)
    bg = np.ones_like(image, np.uint8) * 255

    instance_mask_all, part_sem_segs = hp_model(image, instance_flag=True, human_parsing_flag=True)
    m = instance_mask_all[0] * np.float32(1 / 255)
    face_skin_mask = (np.sum(part_sem_segs[[0, 1, 2, 3, 4]], axis=0) * m).astype(np.uint8)  # 获取人脸skin mask
    pict_merge = alpha_merge(image, bg, face_skin_mask)

    return pict_merge



if __name__ == '__main__':
    img_dir = '/root/group-trainee/ay/version1/dataset/results/双下巴/'
    hp_model = HumanParsing(True)

    for fn in tqdm(os.listdir(img_dir)):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            img_path = os.path.join(img_dir, fn)
            img_merged = get_face(img_path, hp_model)

            save_fn = os.path.join(img_dir, 'merged_images', fn)
            cv2.imwrite(save_fn, img_merged)


