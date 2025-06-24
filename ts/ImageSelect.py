from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop import FacePoint

import os, time, cv2, shutil
import numpy as np
from tqdm import tqdm

from utils.tools.General import alpha_merge, morph, affine_points, inverse_2x3mat


def model(img_person, face_point_model):
    ages = face_point_model.get_attr(img_person, face_point_model.AttrKey.kAttrAge)
    males = face_point_model.get_attr(img_person, face_point_model.AttrKey.kAttrGenderMale)
    females = face_point_model.get_attr(img_person, face_point_model.AttrKey.kAttrGenderFemale)
    lsts = []
    for i in range(len(ages)):
        lst = 0
        if ages[i] < 15:
            lst = 0
        else:
            if males[i] > females[i]:
                lst = 1
            else:
                lst = 2
        lsts.append(lst)

    return lsts


'''
计算性别：
    0：未知
    1：男性
    2：女性
'''
def get_gender(img_person, face_point_model, b_single=True):
    males = face_point_model.get_attr(img_person, face_point_model.AttrKey.kAttrGenderMale)
    females = face_point_model.get_attr(img_person, face_point_model.AttrKey.kAttrGenderFemale)

    # img_person里只有一张人脸
    if b_single:
        if males[0] > females[0]:
            return 0
        elif males[0] > females[0]:
            return 1
        else:
            return 2

    # img_person里有多张人脸
    n = len(males)
    genders = [0] * n
    for i in range(n):
        if males[i] < females[i]:
            genders[i] = 1

    return genders


''''
分类图片并根据类别索引存储在不同的文件夹
'''
def process_and_save(src_dir, sub_dirs, cls_func, *args, **kwargs):
    # 构建存储路径
    save_dirs = []
    for sub_dir in sub_dirs:
        save_dir = os.path.join(src_dir, sub_dir)
        os.makedirs(save_dir, exist_ok=True)
        save_dirs.append(save_dir)

    n = len(sub_dirs)
    cls_nums = [0] * n
    for fn in tqdm(os.listdir(src_dir)):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post!= '':
            src_fp = os.path.join(src_dir, fn)

            '''分类图片'''
            img = cv2.imread(src_fp, -1)
            assert isinstance(img, np.ndarray), f'图像为空：{src_fp}'

            cls_idx = cls_func(img, face_point_model=kwargs['face_point_model'])
            cls_nums[cls_idx] += 1
            save_fp = os.path.join(save_dirs[cls_idx], fn)
            shutil.copyfile(src_fp, save_fp)

    for i in range(n):
        print(f'类别：{sub_dirs[i]}, 图片数量：{cls_nums[i]}')


if __name__ == '__main__':
    src_dir = '/data_ssd/ay/头发分类/dataset/imgs_mark'
    face_point_model = FacePoint()

    genders = process_and_save(src_dir, ['male', 'female', 'others'], get_gender, face_point_model=face_point_model)
