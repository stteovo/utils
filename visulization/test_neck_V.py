import cv2, math
import random
import numpy as np


def modify_real_neck_V(real_neck_mask, face_mask, H=512, W=512):
    n_x1, n_y1, n_w, n_h = cv2.boundingRect(real_neck_mask)
    f_x1, f_y1, f_w, f_h = cv2.boundingRect(face_mask)

    # 求出脖子最上边缘与脸的交点
    face_mask_top_row = face_mask[n_y1, :]
    l_x1 = np.argmax(face_mask_top_row > 0) + 10
    r_x1 = np.argmax((W - face_mask_top_row[::-1]) > 0) + 10
    l_pt1, r_pt1 = (l_x1, n_y1), (r_x1, n_y1)

    # 脖子外轮廓的转折点
    ratio_wave1 = np.random.uniform(-0.05, 0.05)
    ratio_wave2 = np.random.uniform(-0.05, 0.05)
    top_h1 = int((0.4 + ratio_wave1) * n_h)
    top_h2 = int((0.4 + ratio_wave2) * n_h)
    l_pt2, r_pt2 = (l_x1, n_y1 + top_h1), (r_x1, n_y1 + top_h2)

    # 脖子外轮廓最低点
    ratio_wave = np.random.uniform(-0.05, 0.05)
    left_w = int((0.5 + ratio_wave) * n_w)
    mid_lowest_pt = (l_x1 + left_w, n_y1 + n_h)

    cnt_pts = [l_pt1, l_pt2, mid_lowest_pt, r_pt2, r_pt1]
    model_neck_mask = np.zeros_like(real_neck_mask)
    cv2.fillPoly(model_neck_mask, [np.array(cnt_pts, np.int32).reshape(-1, 1, 2)], (255, 255, 255))

    return model_neck_mask


def get_random_point(pt, radius_max, radius_min=5, angles=[0, 360]):
    radius = np.random.randint(radius_min, radius_max)

    # 随机生成一个角度，范围从 0 到 90 度
    angle = random.uniform(angles[0], angles[1])
    # 将角度转换为弧度
    radians = math.radians(angle)

    # 由于我们只需要上半圆，所以 y 坐标需要调整为正值
    x = pt[0] + radius * math.cos(radians)
    y = pt[1] + radius * math.sin(radians)

    x = int(x)
    y = int(y)

    return [x, y]

def draw_random_polygon(pt, radius=100, size=(512, 512)):
    n = np.random.randint(3, 6)
    pts, angles_all = [], [[135, 315], [135, 315], [0, 90], [90, 180], [0, 45]]
    for i in range(n):
        pt_rnd = get_random_point(pt, radius, angles=angles_all[i])
        pts.append(pt_rnd)
    pts = np.array(pts).astype(np.int32)
    mask = np.zeros(size, np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    return mask


if __name__ == '__main__':
    from realTimeVisual import visualize

    test_dir = '/root/group-trainee/ay/version1/dataset/a_online512/test_Stage2_no_shadow_white/real_neck_mask'
    visualize(modify_real_neck_V, test_dir)

