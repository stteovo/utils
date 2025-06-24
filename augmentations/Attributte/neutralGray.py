import os

import cv2

from utils.tools.General import *
from utils.tools.General import _2tuple
from utils.tools.layerBlend import blend_soft_light

from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop import FacePoint
from skimage.util import view_as_windows


def surface_blur(src, radius=5, threshold=25):
    # 创建一个与输入图像相同大小的输出图像
    dst = np.zeros_like(src, dtype=np.float32)

    # 计算边界宽度
    border = (radius - 1) // 2

    # 使用 copyMakeBorder 处理图像边界
    padded_src = cv2.copyMakeBorder(src, border, border, border, border, cv2.BORDER_REFLECT)

    # 获取图像的高度和宽度
    h, w, _ = src.shape

    # 定义邻域的大小
    window_shape = (2 * border + 1, 2 * border + 1)

    # 遍历每个通道
    for k in range(3):
        # 提取当前通道的数据
        channel = padded_src[:, :, k].astype(np.float32)

        # 获取中心像素
        center = channel[border:-border, border:-border]

        # 使用 view_as_windows 创建邻域视图
        neighborhood = view_as_windows(channel, window_shape, step=1)

        # 将邻域视图展平为 (h, w, num_neighbors) 形状
        neighborhood = neighborhood.reshape(h, w, -1)

        # 计算中心像素与邻域内像素的差异
        diffs = np.abs(center[:, :, np.newaxis] - neighborhood)

        # 计算权重矩阵
        weights = 1.0 - (diffs / (2.5 * threshold))
        weights[weights < 0] = 0

        # 累加权重和加权像素值
        weighted_sum = np.sum(neighborhood * weights, axis=2)
        total_weights = np.sum(weights, axis=2)

        # 计算最终的模糊结果
        dst[:, :, k] = weighted_sum / (total_weights + 1e-6)  # 防止除以零

    # 将结果转换回 uint8 类型
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    return dst


def high_pass_filter(image, ks, sigma=0, thre=25):
    '''1）高斯模糊'''
    # blurred = cv2.GaussianBlur(image, (ks, ks), sigma)

    '''2）中值滤波'''
    # blurred = cv2.medianBlur(image, ks)

    '''3）表面模糊'''
    blurred = surface_blur(src, radius=ks, threshold=thre)

    """ 应用高反差保留滤波器 """
    high_pass = cv2.addWeighted(image, 1, blurred, -1, 127)

    return high_pass


'''提取neck mask内阴影'''
def extract_neck_shadow(image, mask):
    # 确保掩码和图像尺寸相同
    if image.shape[:2] != mask.shape:
        print("Error: Image and mask must have the same size.")
        return

    # 将掩码转换为三通道，以便与图像相乘
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 转换为灰度图像，并应用高反差保留滤波器
    gray_neck_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    high_pass_neck = high_pass_filter(gray_neck_image, 13, sigma=0, thre=25)
    # high_pass_neck[neck_mask == 0] = 127

    # 阈值化处理，分离阴影部分
    # 形态学操作，去除噪声
    # _, shadow_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # 将阴影掩码应用于原始图像，突出显示阴影部分
    high_pass_neck = cv2.cvtColor(high_pass_neck, cv2.COLOR_GRAY2BGR)
    shadow_highlighted = blend_soft_light(image, high_pass_neck)

    # 保存结果
    cv2.imwrite('/data_ssd/ay/a_DEBUG/1.png', image)
    cv2.imwrite('/data_ssd/ay/a_DEBUG/2.png', shadow_highlighted)
    cv2.imwrite('/data_ssd/ay/a_DEBUG/3.png', high_pass_neck)

    return

def high_pass(src, human_parsing_model, face_point_model, ksize):
    human_instance_seg, part_sem_seg = human_parsing_model(src, instance_flag=True, human_parsing_flag=True)
    fg_seg = np.sum(human_instance_seg, axis=0).astype(np.uint8)
    m = fg_seg * np.float32(1 / 255)
    neck_mask = (part_sem_seg[5] * m).astype(np.uint8)  # 获取neck mask
    hair_mask = (part_sem_seg[0] * m).astype(np.uint8)  # 获取neck mask

    '''计算人脸点'''
    face_points = face_point_model(src)
    if len(face_points) != 1:
        return
    face_points = face_points[0]

    '''转灰度图， 并进行高斯模糊处理'''
    src = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    dst = high_pass_filter(src, ksize, sigma=0)

    return dst, neck_mask, hair_mask, face_points
    # src_ = blend_soft_light(src, dst)
    # return src, src_

if __name__ == '__main__':
    root_dir = '/data_ssd/ay/a_DEBUG/高反差保留抓取阴影'
    human_parsing_model = HumanParsing(True)
    face_point_model = FacePoint()

    shadow_save_dir = os.path.join(root_dir, 'shadows')
    shadow_neck_dir = os.path.join(root_dir, 'neck_mask')
    shadow_hair_dir = os.path.join(root_dir, 'hair_mask')
    face_pts_dir = os.path.join(root_dir, 'face_pts')
    os.makedirs(shadow_save_dir, exist_ok=True)
    os.makedirs(shadow_neck_dir, exist_ok=True)
    os.makedirs(shadow_hair_dir, exist_ok=True)
    os.makedirs(face_pts_dir, exist_ok=True)

    shadow_img_dir = os.path.join(root_dir, 'shadow_imgs')
    for fn in tqdm(os.listdir(shadow_img_dir)):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            img_path = os.path.join(shadow_img_dir, fn)
            src = cv2.imread(img_path, -1)[:, :, :3]
            res = high_pass(src, human_parsing_model, face_point_model, 21)

            # neck_mask = cv2.imread(os.path.join(shadow_neck_dir, pre + '.png'), cv2.IMREAD_GRAYSCALE)
            # res = extract_neck_shadow(src, neck_mask)

            if res is not None:
                dst, neck_mask, hair_mask, face_points = res
                fn = pre + '.png'
                cv2.imwrite(os.path.join(shadow_save_dir, fn), dst, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(os.path.join(shadow_neck_dir, fn), neck_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(os.path.join(shadow_hair_dir, fn), hair_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                np.save(os.path.join(face_pts_dir, pre + '.npy'), face_points)