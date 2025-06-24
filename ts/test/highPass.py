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
    def __init__(self):
        ''' mask模型定义 '''
        self.face_point_model = FacePoint()  # 人脸点模型
        self.face_point_idx = [0, 1, 4, 3, 18, 5, 22, 7, 8, 9, 10, 12, 13, 14, 36, 14, 16, 17, 36, 40, 63,
                               58, 70, 57, 73, 23, 83, 25, 109, 114, 28, 125, 30, 136, 141]
        self.face_point_idx = list(range(117, 133))
        self.human_parsing_model = HumanParsing(True)  # 脸部mask 模型

        # face_points_index = np.arange(1, 77, 4)
        # lst1 = [190, 191, 192, 193]
        # lst2 = list(range(77, 165, 1))
        # face_points_index = np.append(face_points_index, lst2 + lst1)
        # face_points_index = np.ascontiguousarray(np.arange(194))

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


    def __call__(self, person_path, model_face_pt_path, shadow_material_path):
        img_person_org = cv2.imread(person_path, -1)[:, :, :3]
        assert isinstance(img_person_org, np.ndarray), f"该文件不是图像文件——{person_path}"
        H, W = img_person_org.shape[:2]

        model_face_pts = np.load(model_face_pt_path)
        assert isinstance(model_face_pts, np.ndarray), f"模板人脸点加载有误——{person_path}"
        img_shadow = cv2.imread(shadow_material_path, -1)
        assert isinstance(img_shadow, np.ndarray), f"该文件不是图像文件——{person_path}"

        black = np.zeros_like(img_person_org[:, :, 0], np.uint8)
        black_background = np.dstack([black, black, black])

        '''人脸对齐'''
        # 1) 计算人脸点
        face_points = self.face_point_model(img_person_org)
        if len(face_points) != 1:
            return
        face_points = face_points[0]
        # 2）变换素材
        face_rotate = FaceRotate(model_image=[face_points, H, W])
        # img_shadow = cv2.cvtColor(img_shadow, cv2.COLOR_GRAY2BGR)
        face_rotate(img_shadow, src_fp=model_face_pts, idxes=self.face_point_idx)
        rotate_M = face_rotate.rotate_M[0]
        img_shadow = cv2.warpAffine(img_shadow, rotate_M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(127, 127, 127))
        img_shadow = img_shadow[:, :, 0]
        # img_shadow_l = cv2.flip(img_shadow, 1)
        # img_shadow_all = np.full_like(img_shadow, 127, np.uint8)
        # img_shadow_all[:, :1000] = img_shadow_l[:, :1000]
        # img_shadow_all[:, 1000:] = img_shadow[:, 1000:]
        # limit_shadow = cv2.imread('/root/group-inspect2-data/证件照换装/中间结果/剥离脖子阴影/model_fake_neck_shadow_org.png')
        # img_shadow_all[limit_shadow[:, :, 0] == 127] = 127
        # img_shadow_all[limit_shadow[:, :, 0] == 128] = 127
        img_shadow_all = cv2.GaussianBlur(img_shadow, (21, 21), 0)
        cv2.imwrite('/data_ssd/ay/shadows/model_shadow.png', img_shadow, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        img_out = blend_soft_light(img_person_org, img_shadow)

        human_instance_seg, part_sem_seg = self.human_parsing_model(img_person_org, instance_flag=True, human_parsing_flag=True)
        fg_seg = np.sum(human_instance_seg, axis=0).astype(np.uint8)
        m = fg_seg * np.float32(1 / 255)
        neck_mask = (part_sem_seg[5] * m).astype(np.uint8)  # 获取neck mask
        other_skin_mask = (part_sem_seg[6] * m).astype(np.uint8)  # 获取neck mask

        # face_skin_mask = (np.sum(part_sem_seg[[1, 2, 3, 4, 6]], axis=0) * m).astype(np.uint8)  # 获取人脸skin mask
        # person_mask = (np.sum(part_sem_seg[[1, 2, 3, 4, 5, 6,]], axis=0) * m).astype(np.uint8)  # 获取人脸skin mask
        #
        # # 构建训练数据
        # background = np.full_like(img_person_org, 255, np.uint8)
        # img_out = self.alpha_merge(img_person_org, background, person_mask)
        # real_neck_mask = cv2.add(neck_mask, other_skin_mask)

        img_out = self.alpha_merge(img_out, img_person_org, neck_mask)


        return img_out
        # return img_out, real_neck_mask, face_skin_mask, other_skin_mask, face_points, rotate_M


def alpha_merge(foreground, background, alpha):
    alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
    foreground = foreground.astype(np.float64)

    img_out = foreground * alpha + background * (1 - alpha)
    return img_out.clip(0, 255).astype(np.uint8)


def sythetic_train_data(src_dir, trg_dir, perfect_dir=None):
    model_img_path = '/root/group-trainee/ay/version1/main/model/woman.png'
    p2 = '/root/group-trainee/ay/version1/main/model/model_cloth.png'
    p0 = '/root/group-trainee/ay/version1/main/model/model_back_cloth.png'


    sub_dirs = ['GT', 'real_neck_mask', 'face_skin_mask', 'other_skin_mask', 'face_points_org', 'rotate_M']
    # 获取存储GT、 fake neck mask的路径
    final_dirs = []
    for dir_ in sub_dirs:
        final_dir = os.path.join(trg_dir, dir_)
        final_dirs.append(final_dir)
        if not os.path.exists(final_dir):
            os.makedirs(final_dir)
    gt_dir, real_neck_mask_dir, face_skin_mask_dir, other_skin_mask_dir, face_points_org_dir, rotateM_dir = final_dirs

    trg_fns = os.listdir(gt_dir)
    for person_fn in tqdm(os.listdir(src_dir)):
        prefix, _ = os.path.splitext(person_fn)
        if '.' not in prefix and person_fn not in trg_fns:
            person_path = os.path.join(src_dir, person_fn)

            # 存图
            fn = person_fn
            p_fn = prefix + '.npy'
            gt_path = os.path.join(gt_dir, fn)
            real_neck_mask_path = os.path.join(real_neck_mask_dir, fn)
            face_skin_mask_path = os.path.join(face_skin_mask_dir, fn)
            other_skin_mask_path = os.path.join(other_skin_mask_dir, fn)
            face_points_org_path = os.path.join(face_points_org_dir, p_fn)
            rotateM_path = os.path.join(rotateM_dir, p_fn)

            infer = PreProcess(model_img_path=model_img_path)
            img, real_neck_mask, face_skin_mask, other_skin_mask, face_points_org, rotate_M = infer(person_path, p2, p0)

if __name__ == '__main__':
    infer = PreProcess()
    material_fn1 = '0173.png'
    material_fn2 = '0173.npy'
    root_dir = '/data_ssd/ay/a_DEBUG/高反差保留抓取阴影'
    src_dir = os.path.join(root_dir, 'src_imgs')
    res_dir = os.path.join(src_dir, 'results')

    shadow_dir = "/root/group-inspect2-data/证件照换装/中间结果/剥离脖子阴影/原图/gray_inverse/"
    face_pts_dir = os.path.join(root_dir, 'face_pts')
    # shadow_img_dir = os.path.join(root_dir, 'shadow_imgs')
    for person_fn in tqdm(os.listdir(src_dir)):
        prefix, _ = os.path.splitext(person_fn)
        if '.' not in prefix and _ != '':
            person_fn = 'woman.png'
            person_path = os.path.join(src_dir, person_fn)

            shadow_path = os.path.join(shadow_dir, material_fn1)
            model_face_pt_path = os.path.join(face_pts_dir, material_fn2)

            img_out = infer(person_path, model_face_pt_path, shadow_path)
            save_fp = os.path.join(res_dir, prefix+'_2.png')
            cv2.imwrite(save_fp, img_out, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            shutil.copyfile(person_path, os.path.join(res_dir, prefix + '_1.png'))