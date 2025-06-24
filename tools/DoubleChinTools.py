from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop import FacePoint
import os, cv2
import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit

from utils.tools.General import morph
from lollipop import FaceRotate

def get_face_neck_mask():
    pass

def get_hand_mask():
    pass

def fit_quadratic_curve(points):
    # 将点转换为numpy数组
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    # 构造设计矩阵
    A = np.vstack([x**2, x, np.ones(len(x))]).T

    # 使用最小二乘法求解二次曲线的系数
    coeffs = np.linalg.lstsq(A, y, rcond=None)[0]
    return coeffs

def f_2(x, a, b, c):
    return a * x * x + b * x + c


def get_jaw_mask(face_points, face_neck_mask):
    jaw_mask = np.zeros_like(face_neck_mask, np.uint8)
    face_points = face_points[118: 133]

    # coeffs = fit_quadratic_curve(face_points)
    coeffs = curve_fit(f_2, face_points[:, 0], face_points[:, 1])[0]

    xes = np.linspace(int(face_points[0][0]), int(face_points[-1][0]), 300)
    yes = coeffs[0] * xes ** 2 + coeffs[1] * xes + coeffs[2]
    # 绘制连接线
    for i in range(len(xes) - 1):
        cv2.line(jaw_mask, (int(xes[i]), int(yes[i])), (int(xes[i + 1]), int(yes[i + 1])), 255, 3)


    jaw_mask[face_neck_mask == 0] = 0
    jaw_mask = morph(jaw_mask, 5)
    jaw_mask = cv2.GaussianBlur(jaw_mask, (31, 31), 0)


    # mask_ref = cv2.imread('/data_ssd/doublechin/data/train_717/jaw_mask/BQT-500人拍摄2307_0_2302.png', -1)
    # a_diff = cv2.absdiff(mask_ref, jaw_mask)

    return jaw_mask


def get_face_rotate_model(model_image_path, pad=None, resize=None, dx=0, dy=0):
    model_image = cv2.imread(model_image_path)
    if model_image is None:
        assert False, "get_face_rotate_model —— 读取人脸转正的模板图像出错！！！"

    if pad is not None:
        model_image = np.pad(model_image, pad)

    if resize is not None:
        model_image = cv2.resize(model_image, resize)
        # model_image = cv2.resize(model_image, (768 * 2, 768 * 2))

    # dx, dy = 0, 0  # dx=100 向右偏移量, dy=50 向下偏移量
    if dx != 0 or dy != 0:
        MAT = np.float32([[1, 0, dx], [0, 1, dy]])  # 构造平移变换矩阵
        dst = cv2.warpAffine(model_image, MAT, (model_image.shape[1], model_image.shape[0]),  borderValue=(0, 0, 0))    # 设置白色填充
    else:
        dst = model_image

    face_rotate_model = FaceRotate(dst, False)
    return face_rotate_model


if __name__ == '__main__':
    face_point_model = FacePoint()
    hp_model = HumanParsing(True)
    person_path = '/data_ssd/doublechin/data/train_717/org/BQT-500人拍摄2307_0_2302.png'
    img_person = cv2.imread(person_path, -1)

    face_points_all = face_point_model(img_person)
    instance_mask_all, part_sem_segs = hp_model(img_person, instance_flag=True, human_parsing_flag=True)

    m = instance_mask_all[0] * np.float32(1 / 255)
    face_neck_mask = (np.sum(part_sem_segs[[1, 2, 4, 5]], axis=0) * m).astype(np.uint8)

    face_points = face_points_all[0]
    jaw_mask = get_jaw_mask(face_points, face_neck_mask)
    pass
