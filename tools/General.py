import os, cv2
import numpy as np
from datetime import datetime
from tqdm import tqdm

'''
    图像变换
'''
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


def _2tuple(t):
    return t if isinstance(t, tuple) else (t, t)



'''
    矩阵变换
'''
def affine_points(points, M):
    if not isinstance(points, np.ndarray):
        points = np.array(points, np.float32)
    if points.ndim == 1:
        points = points[None, :]

    points = np.concatenate([points.transpose(1, 0), np.ones(points.shape[0], dtype=M.dtype)[None]])
    M = np.concatenate([M, np.array([0, 0, 1], dtype=M.dtype)[None]])
    points = np.matmul(M, points)
    points = points[:2].transpose(1, 0)

    return points.astype(np.int32)

def inverse_2x3mat(M_affine):

    inverse_affine_matrix = np.vstack([M_affine, [0, 0, 1]])
    inverse_affine_matrix = np.linalg.inv(inverse_affine_matrix)[:2]

    return inverse_affine_matrix




'''Log 信息'''
def get_datetime(mode=0):
    # 获取当前日期时间
    current_datetime = datetime.now()

    # 格式化为 "MM/DD/YYYY"
    if mode == 0:
        formatted_datetime = current_datetime.strftime('%m/%d/%Y')

    # 格式化为 "YYYY-MM-DD HH:MM:SS"
    elif mode == 1:
        formatted_datetime = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    # 格式化为 "HH:MM:SS"
    elif mode == 2:
        formatted_datetime = current_datetime.strftime('%H:%M:%S')

    # 格式化为 "YYYY-MM-DDTHH:MM:SS.ssssss"（ISO 8601 格式）
    elif mode == 3:
        formatted_datetime = current_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')
        # print(f"ISO 8601 格式的当前日期时间: {iso_formatted_datetime}")

    # 格式化为 "Day, Month DD, YYYY HH:MM:SS AM/PM"
    elif mode == 4:
        formatted_datetime = current_datetime.strftime('%A, %B %d, %Y %I:%M:%S %p')
        # print(f"带星期几的当前日期时间: {formatted_datetime_with_day}")

    return formatted_datetime