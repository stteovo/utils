import os, cv2
from lollipop import FacePoint
from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lolite.pose.pose_est.PoseEstimate import PoseEstimate

from tqdm import tqdm
import numpy as np
from utils.tools.General import alpha_merge, morph
from scipy.interpolate import splprep, splev
# from skimage.filters import guided_filter
from sklearn.linear_model import RANSACRegressor
from scipy.interpolate import splprep, splev


# B样条基函数的迭代计算
def b_spline_basis(t, i, k, knots):
    """计算B样条基函数"""
    if k == 0:
        return 1.0 if knots[i] <= t < knots[i + 1] else 0.0
    coef1 = (t - knots[i]) / (knots[i + k] - knots[i]) if knots[i + k] > knots[i] else 0
    coef2 = (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) if knots[i + k + 1] > knots[i + 1] else 0
    return coef1 * b_spline_basis(t, i, k - 1, knots) + coef2 * b_spline_basis(t, i + 1, k - 1, knots)

# 生成 B 样条曲线
def generate_b_spline(control_points, degree, num_points=100):
    n = len(control_points) - 1
    knots = np.concatenate((
        np.zeros(degree + 1),                 # 前 degree+1 个节点为 0
        np.linspace(0, 1, n - degree + 1),   # 中间均匀分布的节点
        np.ones(degree + 1)                  # 后 degree+1 个节点为 1
    ))
    t_values = np.linspace(0, 1, num_points)
    curve = []

    for t in t_values:
        basis_values = np.array([b_spline_basis(t, i, degree, knots) for i in range(len(control_points))])
        point = np.sum(basis_values[:, None] * control_points, axis=0)
        curve.append(point)

    return np.array(curve)

# 快速检查凹性
def check_concavity(curve):
    x, y = curve[:, 0], curve[:, 1]
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # 曲率公式
    curvature = dx * ddy - dy * ddx
    return np.all(curvature <= 0)

# 调整控制点
def adjust_control_points(control_points, degree, max_iterations=100, learning_rate=1.0):
    n_points = len(control_points)
    for iteration in range(max_iterations):
        curve = generate_b_spline(control_points, degree)
        if check_concavity(curve):
            print(f"满足凹性约束，迭代次数: {iteration}")
            break

        # 更新控制点
        for i in range(1, n_points - 1):  # 避免调整首尾控制点
            control_points[i][1] -= learning_rate * (i / n_points)

    return control_points


# Catmull-Rom 样条
def catmull_rom_spline(points, n_points=100):
    tck, u = splprep(points.T, s=0, k=3)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck)
    return np.column_stack((x_new, y_new))


# 贝塞尔曲线
def bezier_curve(points, n_points=100):
    t = np.linspace(0, 1, n_points)
    curve = np.zeros((n_points, 2))
    for i in range(len(points)):
        curve += np.outer((1 - t) ** (len(points) - 1 - i) * t ** i, points[i])
    return curve

# 滑动平均
def moving_average(points, window_size=3):
    weights = np.repeat(1.0, window_size) / window_size
    smoothed_x = np.convolve(points[:, 0], weights, 'same')
    smoothed_y = np.convolve(points[:, 1], weights, 'same')
    return np.column_stack((smoothed_x, smoothed_y))


# # 使用RANSAC算法来识别离群点
# ransac = RANSACRegressor()
# ransac.fit(x_data[:, np.newaxis], y_data)
# inlier_mask = ransac.inlier_mask_
# outlier_mask = np.logical_not(inlier_mask)
#
# # 过滤掉离群点
# x_filtered = x_data[inlier_mask]
# y_filtered = y_data[inlier_mask]
#
# # 计算B样条曲线
# tck, u = splprep([x_filtered, y_filtered], s=0)
#
# # 生成用于绘制曲线的新参数值
# u_fine = np.linspace(0, 1, 100)
#
# # 使用B样条插值计算新点
# x_fine, y_fine = splev(u_fine, tck)


def getFaceX(face_skin_mask_binary, l_y_lst, r_y_lst):
    l_row, r_row = face_skin_mask_binary[l_y_lst, :], face_skin_mask_binary[r_y_lst, :]
    l_xes = np.argmax(l_row, axis=1)
    r_xes = face_skin_mask_binary.shape[1] - np.argmax(r_row[:, ::-1], axis=1)

    l_pts = np.hstack([l_xes[:, None], l_y_lst[:, None]])
    r_pts = np.hstack([r_xes[:, None], r_y_lst[:, None]])
    return l_pts, r_pts


def facepoint2Contour(img_path, face_point_model, hp_model, pose_model=None):
    img = cv2.imread(img_path, -1)
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    assert isinstance(img, np.ndarray)
    background = np.ones_like(img, np.uint8)

    face_points = face_point_model(img)
    if len(face_points) == 0:
        print('未检测到人脸')
        return None

    if len(face_points) > 1:
        print('检测到多个人脸')
        return None

    face_points = face_points[0]
    # face_points = catmull_rom_spline(face_points, len(face_points) * 2)
    face_points_standard = face_points[114:137]

    contour_results = pose_model(img)
    if len(contour_results['pose_point']) == 0:
        return None
    contour_pts = np.array(contour_results['pose_point'])
    neck_idxes = [20, 41]
    cnt_neck_pts = contour_pts[0, neck_idxes]

    instance_mask_all, part_sem_segs = hp_model(img, instance_flag=True, human_parsing_flag=True)
    if instance_mask_all.shape[0] != 1:
        return None
    m = instance_mask_all[0] * np.float32(1 / 255)
    face_skin_neck_mask = (np.sum(part_sem_segs[[1, 2, 3, 4, 5]], axis=0) * m).astype(np.uint8)  # 获取人脸skin mask
    face_skin_mask = (np.sum(part_sem_segs[[1, 2, 3, 4]], axis=0) * m).astype(np.uint8)  # 获取人脸skin mask
    ear_mask = (part_sem_segs[3] * m).astype(np.uint8)  # 获取人脸skin mask
    neck_mask = (part_sem_segs[5] * m).astype(np.uint8)  # 获取人脸skin mask
    # y = int((cnt_neck_pts[0][1] + cnt_neck_pts[1][1]) / 2 - 0.1 * abs(cnt_neck_pts[0][0] - cnt_neck_pts[1][0]))

    # 根据左右耳朵的位置确定起止点
    num_ear_mask = (ear_mask > 0).astype(np.int32).sum()
    mid_x = int((face_points[73][0] + face_points[83][0]) / 2)
    l_ear_mask, r_ear_mask = np.copy(ear_mask), np.copy(ear_mask)
    l_ear_mask[:, mid_x:] = 0
    r_ear_mask[:, :mid_x] = 0
    l_ear_mask = np.where(l_ear_mask > 127, 255, 0).astype(np.uint8)
    if (l_ear_mask > 120).astype(np.uint8).sum() < 1000:
        return None
    r_ear_mask = np.where(r_ear_mask > 127, 255, 0).astype(np.uint8)
    _, ly, _, lh = cv2.boundingRect(l_ear_mask)
    _, ry, _, rh = cv2.boundingRect(r_ear_mask)
    ly, ry = ly + lh, ry + rh
    l_num_ear_mask = (l_ear_mask > 0).astype(np.int32).sum()
    r_num_ear_mask = (r_ear_mask > 0).astype(np.int32).sum()
    if l_num_ear_mask < num_ear_mask * 0.2:
        ly = face_points[113][1]
    if r_num_ear_mask < num_ear_mask * 0.2:
        ry = face_points[137][1]
    # 根据脖子的位置确定
    l_neck_mask, r_neck_mask = np.copy(neck_mask), np.copy(neck_mask)
    l_neck_mask[:, mid_x:] = 0
    r_neck_mask[:, :mid_x] = 0
    l_neck_mask = np.where(l_neck_mask > 200, 255, 0).astype(np.uint8)
    r_neck_mask = np.where(r_neck_mask > 200, 255, 0).astype(np.uint8)
    _, ly_n, _, _ = cv2.boundingRect(l_neck_mask)
    _, ry_n, _, _ = cv2.boundingRect(r_neck_mask)

    # 确定人脸点的范围：两端最外围点的高度要大于两边脖子的最高点
    l_idx = 114
    while(face_points[l_idx][1] < ly_n):
        l_idx += 1
    l_idx -= 1
    r_idx = 136
    while (face_points[r_idx][1] < ry_n):
        r_idx -= 1
    r_idx += 1


    # 截取两边（不包含脖子区域的）自画人脸点
    n_pts = 15
    l_y_lst, r_y_lst = np.linspace(ly, face_points[l_idx][1], n_pts),  np.linspace(ry, face_points[r_idx][1], n_pts)
    l_y_lst, r_y_lst = l_y_lst.astype(np.int32), r_y_lst.astype(np.int32)
    face_skin_neck_mask_binary = np.where(face_skin_neck_mask > 127, 255, 0).astype(np.uint8)
    l_pts, r_pts = getFaceX(face_skin_neck_mask_binary, l_y_lst, r_y_lst)
    # 若衔接点相邻过近，需要剔除
    if l_pts[-1][0] - face_points[l_idx][0] < 20 or abs(l_pts[-1][1] - face_points[l_idx][1]) < 20:
        l_pts = l_pts[:-1]

    if r_pts[-1][0] - face_points[r_idx][0] < 20 or abs(r_pts[-1][1] - face_points[r_idx][1]) < 20:
        r_pts = r_pts[:-1]

    # 截取合适范围的人脸点
    face_idxes = list(range(l_idx, r_idx + 1))
    face_points = face_points[face_idxes]

    # 拼接人脸点和自画人脸点
    face_points = face_points.astype(np.int32)
    l_pts = l_pts.astype(np.int32)
    r_pts = r_pts.astype(np.int32)
    img_cpy = np.copy(img)
    for pt in face_points:
        cv2.drawMarker(img_cpy, pt, (0, 0, 255), cv2.MARKER_STAR, 2, 2)
    for pt in l_pts:
        cv2.drawMarker(img_cpy, pt, (0, 255, 0), cv2.MARKER_STAR, 2, 2)
    for pt in r_pts:
        cv2.drawMarker(img_cpy, pt, (0, 255, 0), cv2.MARKER_STAR, 2, 2)
    face_points = np.vstack([l_pts, face_points, r_pts[::-1]])
    print(face_points)

    # face_points = catmull_rom_spline(face_points, len(face_points) * 2)

    # # 调整控制点
    # adjusted_control_points = adjust_control_points(face_points.astype(np.float32), 3)
    # # 生成最终曲线
    # face_points = generate_b_spline(adjusted_control_points, 3)
    # face_points = face_points.astype(np.int32)

    # 使用 B 样条进行拟合
    # 提取 x 和 y 坐标
    x, y = face_points[:, 0], face_points[:, 1]# 去重
    unique_indices = np.unique(x, return_index=True)[1]
    x = x[unique_indices]
    y = y[unique_indices]
    try:
        tck, u = splprep([x, y], s=30)  # s 为平滑因子，值越大越平滑
    except:
        print(img_path)
    new_u = np.linspace(0, 1, 500)  # 生成更多的点
    face_points = splev(new_u, tck)
    face_points = np.squeeze(cv2.merge(face_points), axis=1)
    face_points = face_points.astype(np.int32)

    # 剔除x坐标相同的点
    unique_points = []
    seen_x = set()
    for point in face_points:
        if point[0] not in seen_x:
            unique_points.append(point)
            seen_x.add(point[0])
    face_points = np.array(unique_points)
    face_points = face_points[:-1, ]
    # # 获取 y 值的排序索引
    # sorted_indices = np.argsort(face_points[:, 1])
    # # 根据排序索引对坐标数组进行排序
    # face_points = face_points[sorted_indices]
    z_img_pt = np.copy(img)
    for pt in face_points:
        cv2.drawMarker(z_img_pt, pt, (255, 255, 0), cv2.MARKER_STAR, 2, 2)


    # 画出人脸轮廓
    face_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(face_mask, [face_points], 255, 8)
    face_mask = np.where(np.logical_and(face_skin_neck_mask < 100, face_mask > 0), 0, face_mask).astype(np.uint8)

    # face_mask_standard = np.zeros(img.shape[:2], np.uint8)
    # face_points_standard = face_points_standard.astype(np.int32)
    # cv2.fillPoly(face_mask_standard, [face_points_standard], 255, 8)
    # face_mask_standard = np.where(np.logical_and(face_skin_neck_mask < 100, face_mask_standard > 0), 0, face_mask_standard).astype(np.uint8)
    # face_mask_standard = cv2.GaussianBlur(face_mask_standard, (11, 11), 0)
    # val_image_3 = alpha_merge(img, background, face_mask_standard)

    # face_mask = cv2.absdiff(face_mask_standard, face_mask)

    # y = int(max(cnt_neck_pts[0][1], cnt_neck_pts[1][1]) - 0.2 * abs(cnt_neck_pts[0][0] - cnt_neck_pts[1][0]))
    y = max(ly, ry) + 30
    # thre = int(0.1 * abs(cnt_neck_pts[0][0] - cnt_neck_pts[1][0]))
    face_skin_neck_mask[y:, ] = 0
    face_mask[:int(face_points[0][1] - 30), ] = 0
    # face_mask = cv2.GaussianBlur(face_mask, (5, 5), 0)
    mask_all = cv2.add(face_skin_neck_mask, face_mask)
    mask_all = cv2.GaussianBlur(mask_all, (5, 5), 0)
    val_image = alpha_merge(img, background, mask_all)
    val_image_2 = alpha_merge(img, background, face_skin_neck_mask)

    return face_mask, val_image, img


if __name__ == '__main__':
    src_dir = '/root/group-trainee/ay/version1/测试数据/haed'
    save_dir = os.path.join(src_dir, 'res')
    os.makedirs(save_dir, exist_ok=True)
    face_point_model = FacePoint()
    hp_model = HumanParsing(True)
    pose_model = PoseEstimate()

    for fn in tqdm(os.listdir(src_dir)):
        fn = '099.jpg'
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            img_fp = os.path.join(src_dir, fn)
            res = facepoint2Contour(img_fp, face_point_model, hp_model, pose_model)
            if res is not None:
                face_mask, val_image, img_pt = res

                mask_save_fn = os.path.join(save_dir, pre + '_3.png')
                pt_save_fn = os.path.join(save_dir, pre + '_2.png')
                img_save_fn = os.path.join(save_dir, pre + '_1.png')
                cv2.imwrite(mask_save_fn, face_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(pt_save_fn, val_image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                cv2.imwrite(img_save_fn, img_pt, [cv2.IMWRITE_PNG_COMPRESSION, 0])


