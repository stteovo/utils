from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop.parsing.face_parsing.FaceParsing import FaceParsing
from lollipop import FacePoint
import cv2, os, shutil
import numpy as np


def alpha_merge(foreground, background, alpha):
    alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)
    return img_out.clip(0, 255).astype(np.uint8)


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


def calc_ear_shift(ear_mask_single, y_mid, W):
    row = ear_mask_single[y_mid]
    l, r = np.argmax(row), W - np.argmax(row[::-1])

    shift = (r - l) // 3 * 2

    return shift


def process_single(img_path, human_parsing_model, face_point_model):
    img = cv2.imread(img_path, -1)
    H, W = img.shape[:2]


    human_instance_seg, part_sem_seg = human_parsing_model(img, instance_flag=True, human_parsing_flag=True)

    '''
    根据人脸转正所用的warp矩阵来调整human parsing分割结果的分辨率  
    human parsing获取人脸mask、控制点，控制点用来后续贴图时的定位
    人脸控制点，为脸部缺失补充脖子区域，减缓脸部伪影
    '''
    human_instance_seg, part_sem_seg = human_instance_seg[0], part_sem_seg
    m = human_instance_seg * np.float32(1 / 255)


    ear_mask = (part_sem_seg[3] * m).astype(np.uint8)
    beard_mask = (part_sem_seg[4] * m).astype(np.uint8)
    face_no_skin_mask = (part_sem_seg[2] * m).astype(np.uint8)

    ''''''
    mask1 = np.full_like(human_instance_seg, 254, np.uint8)
    if human_instance_seg is None:
        a = 1
    mask1 = cv2.subtract(mask1, morph(human_instance_seg, 100))


    face_points = face_point_model(img)[0]

    '''ear'''
    ear_mask_ = np.copy(ear_mask)
    x1, y1, x2, y2, x_mid = int(face_points[112][0]), int(face_points[112][1]), int(face_points[138][0]), int(face_points[138][1]), int(face_points[58][0])

    l_ear_mask, r_ear_mask = np.copy(ear_mask), np.copy(ear_mask)
    l_ear_mask[:, x_mid:] = 0
    r_ear_mask[:, :x_mid] = 0
    l_shift, r_shift = calc_ear_shift(l_ear_mask, y1, W), calc_ear_shift(r_ear_mask, y2, W)
    ear_mask_[:, :x1-l_shift] = 0
    ear_mask_[:, x2+r_shift:] = 0
    mask2 = cv2.subtract(ear_mask, ear_mask_)


    '''face'''
    idxes = [158, 0, 63, 70, 36, 9, 148]


    pt2 = np.array([face_points[0][0], face_points[62][1]], np.float32)
    pt3 = np.array([face_points[63][0], (face_points[63][1] + face_points[58][1]) / 2], np.float32)
    pt4 = np.array([face_points[70][0], (face_points[70][1] + face_points[58][1]) / 2], np.float32)
    pt5 = np.array([face_points[9][0], face_points[69][1]], np.float32)
    pts = [
        face_points[158], face_points[0], pt2, pt3,
        pt4, pt5, face_points[9], face_points[148]
    ]

    pts = np.array(pts, np.int32)
    mask3 = np.zeros_like(ear_mask, np.uint8)
    cv2.fillPoly(mask3, [pts.reshape(-1, 1, 2)], 255)


    '''face no skin'''
    mask4 = np.copy(face_no_skin_mask)
    mask4 = morph(mask4, 20, cv2.MORPH_ERODE)


    mask = cv2.add(mask1, mask2)
    mask = cv2.add(mask, mask3)
    mask = cv2.add(mask, mask4)
    mask = cv2.subtract(mask, beard_mask)
    mask = np.where(mask < 100, 0, 255).astype(np.uint8)


    black = np.dstack([np.zeros_like(ear_mask), np.zeros_like(ear_mask), np.zeros_like(ear_mask)])
    img_out = alpha_merge(black, img, mask)

    return img_out, mask


def process_single_(img, human_instance_seg, part_sem_seg, face_points):
    H, W = img.shape[:2]


    '''
    根据人脸转正所用的warp矩阵来调整human parsing分割结果的分辨率  
    human parsing获取人脸mask、控制点，控制点用来后续贴图时的定位
    人脸控制点，为脸部缺失补充脖子区域，减缓脸部伪影
    '''
    m = human_instance_seg * np.float32(1 / 255)


    ear_mask = (part_sem_seg[3] * m).astype(np.uint8)
    beard_mask = (part_sem_seg[4] * m).astype(np.uint8)
    face_no_skin_mask = (part_sem_seg[2] * m).astype(np.uint8)

    ''''''
    mask1 = np.full_like(human_instance_seg, 254, np.uint8)
    if human_instance_seg is None:
        a = 1
    mask1 = cv2.subtract(mask1, morph(human_instance_seg, 100))


    '''ear'''
    ear_mask_ = np.copy(ear_mask)
    x1, y1, x2, y2, x_mid = int(face_points[112][0]), int(face_points[112][1]), int(face_points[138][0]), int(face_points[138][1]), int(face_points[58][0])

    l_ear_mask, r_ear_mask = np.copy(ear_mask), np.copy(ear_mask)
    l_ear_mask[:, x_mid:] = 0
    r_ear_mask[:, :x_mid] = 0
    l_shift, r_shift = calc_ear_shift(l_ear_mask, y1, W), calc_ear_shift(r_ear_mask, y2, W)
    ear_mask_[:, :x1-l_shift] = 0
    ear_mask_[:, x2+r_shift:] = 0
    mask2 = cv2.subtract(ear_mask, ear_mask_)


    '''face'''
    idxes = [158, 0, 63, 70, 36, 9, 148]


    pt2 = np.array([face_points[0][0], face_points[62][1]], np.float32)
    pt3 = np.array([face_points[63][0], (face_points[63][1] + face_points[58][1]) / 2], np.float32)
    pt4 = np.array([face_points[70][0], (face_points[70][1] + face_points[58][1]) / 2], np.float32)
    pt5 = np.array([face_points[9][0], face_points[69][1]], np.float32)
    pts = [
        face_points[158], face_points[0], pt2, pt3,
        pt4, pt5, face_points[9], face_points[148]
    ]

    pts = np.array(pts, np.int32)
    mask3 = np.zeros_like(ear_mask, np.uint8)
    cv2.fillPoly(mask3, [pts.reshape(-1, 1, 2)], 255)


    '''face no skin'''
    mask4 = np.copy(face_no_skin_mask)
    mask4 = morph(mask4, 20, cv2.MORPH_ERODE)


    mask = cv2.add(mask1, mask2)
    mask = cv2.add(mask, mask3)
    mask = cv2.add(mask, mask4)
    mask = cv2.subtract(mask, beard_mask)
    mask = np.where(mask < 100, 0, 255).astype(np.uint8)


    black = np.dstack([np.zeros_like(ear_mask), np.zeros_like(ear_mask), np.zeros_like(ear_mask)])
    img_out = alpha_merge(black, img, mask)

    return img_out, mask

def process_single_v2(img, human_instance_seg, part_sem_seg, face_points=None):
    H, W = img.shape[:2]


    '''
    根据人脸转正所用的warp矩阵来调整human parsing分割结果的分辨率  
    human parsing获取人脸mask、控制点，控制点用来后续贴图时的定位
    人脸控制点，为脸部缺失补充脖子区域，减缓脸部伪影
    '''
    m = human_instance_seg * np.float32(1 / 255)


    mask = (np.sum(part_sem_seg[[1, 2, 3, 4, 14]], axis=0) * m).astype(np.uint8)

    mask = morph(mask, 40, cv2.MORPH_ERODE)
    mask = np.where(mask > 200, 255, 0).astype(np.uint8)
    bg_mask = np.where(human_instance_seg > 0, 0, 255).astype(np.uint8)
    bg_mask = morph(bg_mask, 60, cv2.MORPH_ERODE)
    mask = cv2.add(bg_mask, mask)

    black = np.dstack([np.zeros_like(mask), np.zeros_like(mask), np.zeros_like(mask)])
    img_out = alpha_merge(black, img, mask)

    return img_out, mask


def cut(img, face_points):
    H, W = img.shape[:2]

    l, r = face_points[165][0], face_points[189][0]
    t, b = face_points[177][1], face_points[125][1]

    w = abs(l - r) // 4
    h = abs(t- b) // 5

    l, r = max(0, l-w), min(W, r+w)
    t, b = max(0, t), min(H, b+h)

    return [int(t), int(b), int(l), int(r)]


def cut_and_process(img_path, face_point_model, human_parsing_model):
    img = cv2.imread(img_path, -1)
    H, W = img.shape[:2]

    all_face_points = face_point_model(img)
    human_instance_segs, part_sem_segs = human_parsing_model(img, instance_flag=True, human_parsing_flag=True)


    img_outs, mask_outs = [], []
    if human_instance_segs.shape[0] == 1:
        human_instance_seg, face_points = human_instance_segs[0], all_face_points[0]

        # extension = cut(img, face_points)
        # img_ = img[extension[0]:extension[1], extension[2]:extension[3]]
        # face_points[:, 0] -= extension[2]
        # face_points[:, 1] -= extension[0]
        # human_instance_seg = human_instance_seg[extension[0]:extension[1], extension[2]:extension[3]]
        # part_sem_segs = [part_sem_seg[extension[0]:extension[1], extension[2]:extension[3]] for part_sem_seg in part_sem_segs]
        img_out, mask = process_single_v2(img, human_instance_seg, part_sem_segs, face_points)

        img_outs.append(img_out)
        mask_outs.append(mask)

    else:
        face_maks_points = []
        for face_points in all_face_points:
            mask_ = np.zeros((H, W), np.uint8)
            cv2.fillPoly(mask_, [face_points[109:164].reshape(-1, 1, 2).astype(np.int32)], 255)
            face_maks_points.append(mask_)

        img_outs = []
        for human_instance_seg in human_instance_segs:
            m = human_instance_seg * np.float32(1 / 255)
            face_mask_seg = (np.sum(part_sem_segs[[1, 2, 3, 4, 5]], axis=0) * m).astype(np.uint8)

            idx, max_num = 0, 0
            for i in range(len(face_maks_points)):
                num = np.logical_and(face_mask_seg > 100, face_maks_points[i] > 100).astype(np.int32).sum()

                if num > max_num:
                    idx = i

            face_points = all_face_points[idx]
            img_out, mask = process_single_v2(img, human_instance_seg, part_sem_segs, face_points)

            img_outs.append(img_out)
            mask_outs.append(mask)


    return img_outs, mask_outs



if __name__ == '__main__':
    face_points_model = FacePoint()
    human_parsing_model = HumanParsing(True)
    root = '/data_ssd/ay/neck_color/阴影修图/'

    src_dir = os.path.join(root, '原图')
    save_dir = os.path.join(root, '原图（打码后）')
    save_mask_dir = os.path.join(root, '打码mask')

    for fn in os.listdir(src_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            fp = os.path.join(src_dir, fn)
            save_fn = pre + '.png'
            save_fp = os.path.join(save_dir, save_fn)
            save_mask_fp = os.path.join(save_mask_dir, save_fn)

            img_outs, mask_outs = cut_and_process(fp, face_points_model, human_parsing_model)

            if len(img_outs) == 1:
                cv2.imwrite(save_fp, img_outs[0])
                cv2.imwrite(save_mask_fp, mask_outs[0])

            else:
                for i in range(len(img_outs)):
                    fn_ = fn[:-4] + '_' + str(i) + '.png'
                    save_fp = os.path.join(save_dir, fn_)
                    save_mask_fp = os.path.join(save_mask_dir, fn_)
                    cv2.imwrite(save_fp, img_outs[i])
                    cv2.imwrite(save_mask_fp, mask_outs[i])