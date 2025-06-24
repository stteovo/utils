import cv2, os
import numpy as np
from utils.ts.rawEngin.ParamList import Random_Dark2
from utils.ts.rawEngin import TsreRandomParams as tsre_rp
from utils.ts.rawEngin.base import *
import matplotlib.pyplot as plt

def alpha_merge(foreground, background, alpha):
    alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)

    return img_out.clip(0, 255).astype(np.uint8)


def fakeNeckWinkles(image, image_dark, face_skin_mask, mask):
    image_h, image_w = image.shape[:2]
    x, y, w, h = cv2.boundingRect(face_skin_mask)
    x_align = x + w // 2
    y_top = y + h

    x, y, w, h = cv2.boundingRect(mask)
    start_x = (x + w//2) - x_align
    end_x = start_x + image_w
    start_y = y - y_top + 200
    end_y = start_y + image_h
    winkle_mask = mask[start_y: end_y, start_x: end_x]

    img_out = alpha_merge(image_dark, image, winkle_mask)
    return img_out


if __name__ == '__main__':
    try:
        import Tetrachromacy as tetra
        print("Init Raw Engine (PyRawEngine Version = %s) ... " % (tetra.__version__))
    except BaseException as be:
        msg = "开始前请先阅读 Readme.md，并确认正确 install 了 Tetra \n"
        print(msg, be)
    raw_engine_obj = tetra.RawEngine()
    raw_engine_obj.InitEnv(type=tetra.EngineType.kSIMD, support_raw=False)


    mask_fn = '/data_ssd/materials/颈纹素材/横穿脖子的颈纹素材/neckwrinkle_mask-4_shadow_plus.tif'
    mask = cv2.imread(mask_fn, -1)[:, :, 0]
    src_dir = '/data_ssd/ay/neck_color/a_stage2/'
    src_image_dir = os.path.join(src_dir, 'smodel')
    src_mask_dir = os.path.join(src_dir, 'face_skin_mask')
    for fn in os.listdir(src_image_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != ':':
            img_fp = os.path.join(src_image_dir, fn)
            image = cv2.imread(img_fp, -1)[:, :, :3]
            face_skin_mask_fp = os.path.join(src_mask_dir, fn)
            face_skin_mask = cv2.imread(face_skin_mask_fp, -1)
            params_wb_fcn = Random_Dark2()
            params_dict = params_wb_fcn()
            image_dark = ProcessByBGR(raw_engine_obj, image, params_dict)
            image_out = fakeNeckWinkles(image, image_dark, face_skin_mask, mask)
            image_out = image_out[:, :, ::-1]
            image = image[:, :, ::-1]

            # 创建一个包含两个子图的图形窗口
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # 在第一个子图中显示第一张图像
            ax1.imshow(image)
            ax1.set_title('Image 1')
            ax1.axis('off')  # 关闭坐标轴

            # 在第二个子图中显示第二张图像
            ax2.imshow(image_out)
            ax2.set_title('Image 2')
            ax2.axis('off')  # 关闭坐标轴

            # 调整子图之间的间距
            plt.tight_layout()

            # 获取当前图形管理器
            manager = plt.get_current_fig_manager()
            # 设置窗口的位置和大小
            # 注意：不同平台的图形管理器可能有不同的 API，以下是Linux平台（使用 Qt5 后端）的示例
            manager.window.setGeometry(300, 400, 1800, 1200)

            # 显示图形
            plt.show()