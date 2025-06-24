import shutil

from utils.ts.FaceRotate import FaceRotate
from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop.parsing.face_parsing.FaceParsing import FaceParsing
from lollipop import FacePoint
import json, cv2, os
import numpy as np
from tqdm import tqdm

from utils.tools.layerBlend import blend_soft_light

# scale_head = 脸颊宽：领口宽
class PreProcess:
    def __init__(self, model_face_pt_path, model_image_path, H=2000, W=2000):
        model_img_person = cv2.imread(model_image_path, -1)
        assert isinstance(model_img_person, np.ndarray), f"该文件不是图像文件——{model_image_path}"
        self.H, self.W = model_img_person.shape[0], model_img_person.shape[1]
        self.model_img_person = self.alpha_merge(model_img_person[:, :, :3], np.full([self.H, self.W, 3], 255, np.uint8), model_img_person[:, :, 3:])

        model_face_pts = np.load(model_face_pt_path)
        assert isinstance(model_face_pts, np.ndarray), f"模板人脸点加载有误——{model_face_pt_path}"
        self.face_rotate = FaceRotate(model_image=[model_face_pts, H, W])

        self.face_point_idx = [0, 1, 4, 3, 18, 5, 22, 7, 8, 9, 10, 12, 13, 14, 36, 14, 16, 17, 36, 40, 63,
                               58, 70, 57, 73, 23, 83, 25, 109, 114, 28, 125, 30, 136, 141]
        self.face_point_idx = list(range(117, 133))

        self.human_parsing_model = HumanParsing(True)  # 脸部mask 模型

        self.model_neck_mask = cv2.imread('/root/group-trainee/ay/version1/main/model/model_fake_neck_mask.png', -1)

    # alpha融合
    def alpha_merge(self, foreground, background, alpha):
        alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
        foreground = foreground.astype(np.float64)

        img_out = foreground * alpha + background * (1 - alpha)
        return img_out.clip(0, 255).astype(np.uint8)

        # 边缘平滑，消除画笔抠图时，羽化对目标边缘提取的影响（仅针对人工抠图）


    def center_wrap(self, img, tx, ty=0):
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        rows, cols = img.shape[0], img.shape[1]
        img_out = cv2.warpAffine(img, M, (cols, rows))
        return img_out

        # 等比缩放

    def morph(self, mask, kernel_szie=10, operation=cv2.MORPH_DILATE, shape=cv2.MORPH_RECT, binary=False):
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

    def affine_point(self, points, M):
        if not isinstance(points, np.ndarray):
            points = np.array(points, np.float32)
        points = np.concatenate([points.transpose(1, 0), np.ones(points.shape[0], dtype=M.dtype)[None]])
        M = np.concatenate([M, np.array([0, 0, 1], dtype=M.dtype)[None]])
        points = np.matmul(M, points)
        points = points[:2].transpose(1, 0)

        return points.astype(np.int32)

    def to_512_rgb_c4_float(self, img):
        if img.ndim == 2:
            img_out = np.expand_dims(img, axis=(0, 1))
            img_out = img_out.astype(np.float32) / 127.5 - 1
        else:
            img_out = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_out = np.expand_dims(img_out.transpose(2, 0, 1), axis=0)
            img_out = img_out.astype(np.float32) / 127.5 - 1

        return img_out


    def __call__(self, person_path, gray_path, shadow_face_pt_path):
        img_person_org = cv2.imread(person_path, -1)[:, :, :3]
        assert isinstance(img_person_org, np.ndarray), f"该文件不是图像文件——{person_path}"
        
        shadow_face_pts = np.load(shadow_face_pt_path)
        assert isinstance(shadow_face_pts, np.ndarray), f"阴影原图人脸点加载有误——{shadow_face_pt_path}"
        img_shadow = cv2.imread(gray_path, -1)
        assert isinstance(img_shadow, np.ndarray), f"该文件不是图像文件——{gray_path}"

        black = np.zeros_like(img_person_org[:, :, 0], np.uint8)
        black_background = np.dstack([black, black, black])

        '''人脸对齐'''
        # 1）变换素材
        # img_shadow = cv2.cvtColor(img_shadow, cv2.COLOR_GRAY2BGR)
        self.face_rotate(img_shadow, src_fp=shadow_face_pts, idxes=self.face_point_idx)
        rotate_M = self.face_rotate.rotate_M[0]
        img_shadow = cv2.warpAffine(img_shadow, rotate_M, (self.W, self.H), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        img_shadow = cv2.GaussianBlur(img_shadow, (21, 21), 0)
        img_out = blend_soft_light(self.model_img_person, img_shadow)

        img_shadow = img_shadow[:, :, 0]
        img_shadow_inverse = cv2.flip(img_shadow, 1)
        # img_shadow_all = np.full_like(img_shadow, 127, np.uint8)
        # img_shadow_all[:, :1000] = img_shadow_inverse[:, :1000]
        # img_shadow_all[:, 1000:] = img_shadow[:, 1000:]
        # limit_shadow = cv2.imread('/root/group-inspect2-data/证件照换装/中间结果/剥离脖子阴影/model_fake_neck_shadow_org.png')
        # img_shadow_all[limit_shadow[:, :, 0] == 127] = 127
        # img_shadow_all[limit_shadow[:, :, 0] == 128] = 127

        return img_shadow, img_shadow_inverse, img_out


def alpha_merge(foreground, background, alpha):
    alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)
    return img_out.clip(0, 255).astype(np.uint8)


if __name__ == '__main__':
    model_image_path = '/root/group-trainee/ay/version1/main/model/sucai/woman/woman.png'
    model_face_points_path = '/root/group-trainee/ay/version1/main/model/sucai/woman/woman.npy'
    infer = PreProcess(model_face_points_path, model_image_path)

    root_dir = '/data_ssd/ay/a_DEBUG/高反差保留抓取阴影/'
    face_pts_dir = os.path.join(root_dir, 'face_pts')
    gray_dir = os.path.join(root_dir, 'gray_inverse')
    shadow_img_dir = os.path.join(root_dir, 'shadow_imgs')

    save_gray_dir = os.path.join(root_dir, 'gray_model')
    save_val_img_dir = os.path.join(root_dir, 'gray_model_val')
    os.makedirs(save_val_img_dir, exist_ok=True)
    os.makedirs(save_gray_dir, exist_ok=True)
    for person_fn in tqdm(os.listdir(shadow_img_dir)):
        pre, _ = os.path.splitext(person_fn)
        if '.' not in pre and _ != '':
            shadow_img_path = os.path.join(shadow_img_dir, person_fn)
            gray_path = os.path.join(gray_dir, pre + '.png')
            shadow_face_pt_path = os.path.join(face_pts_dir, pre + '.npy')

            gray_out, gray_out_inverse, img_out = infer(shadow_img_path, gray_path, shadow_face_pt_path)
            save_fp = os.path.join(save_gray_dir, pre+'.png')
            save_fp_inverse = os.path.join(save_gray_dir, pre+'_inverse.png')
            cv2.imwrite(save_fp, gray_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(save_fp_inverse, gray_out_inverse, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            shutil.copyfile(model_image_path, os.path.join(save_val_img_dir, pre + '_1.png'))
            cv2.imwrite(os.path.join(save_val_img_dir, pre + '_2.png'), img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])