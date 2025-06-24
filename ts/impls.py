from lollipop import FacePoint
import os, shutil, cv2
import numpy as np
from tqdm import tqdm

from utils.ts.FaceRotate import myFaceRotate
from utils.tools.General import alpha_merge, morph, affine_points


def face_rotate_mine(model_image_path, face_points_model, src_img, face_points_indexes):
    if not isinstance(src_img, np.ndarray):
        print('输入图像不是np.ndarray类型！！！')
        return None

    '''人脸点检测'''
    src_face_points = face_points_model(src_img)
    if src_face_points is None or len(src_face_points) == 0:
        print('输入图像中未检测到人脸！！！')
        return None
    src_face_points = src_face_points[0]

    '''人脸转正'''
    face_rotate_model = myFaceRotate(model_image_path)
    img_frt = face_rotate_model(src_img, src_fp=src_face_points, idxes=face_points_indexes)

    return img_frt


'''模板图像固定'''
def test_fix():
    face_points_model = FacePoint()
    model_image_path = '/root/group-trainee/ay/version1/main/model/sucai/woman/woman.png'
    face_points_indexes = [0, 1, 4, 3, 18, 5, 22, 7, 8, 9, 10, 12, 13, 14, 36, 14, 16, 17, 36, 40, 63,
                           58, 70, 57, 73, 23, 83, 25, 109, 114, 28, 125, 30, 136, 141]

    src_dir = '/data_ssd/ay/ID_DATA/肩宽点/img_'
    save_dir = '/data_ssd/ay/ID_DATA/肩宽点/imgs_frt'
    exclude_dir = os.path.join(src_dir, '不符合条件的图片')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exclude_dir, exist_ok=True)
    num = 0
    for fn in tqdm(os.listdir(src_dir)):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            src_path = os.path.join(src_dir, fn)
            src_img = cv2.imread(src_path, -1)

            img_frt = face_rotate_mine(model_image_path, face_points_model, src_img, face_points_indexes)
            if img_frt is None:
                shutil.move(src_path, os.path.join(exclude_dir, fn))
                continue
            save_path = os.path.join(save_dir, fn)
            cv2.imwrite(save_path, img_frt, [cv2.IMWRITE_JPEG_QUALITY, 100])
            num += 1

    print(f'转正结束，共转正{num}张图片。')


'''模板图像固定'''
def test_unfixed():
    face_points_model = FacePoint()
    face_points_indexes = [0, 1, 4, 3, 18, 5, 22, 7, 8, 9, 10, 12, 13, 14, 36, 14, 16, 17, 36, 40, 63,
                           58, 70, 57, 73, 23, 83, 25, 109, 114, 28, 125, 30, 136, 141]

    src_dir = '/root/group-inspect2-data/证件照换装/中间结果/all_test/原图'
    model_image_dir = '/root/group-inspect2-data/证件照换装/中间结果/all_test/output'
    save_dir = '/root/group-inspect2-data/证件照换装/中间结果/all_test/转正后原图'
    exclude_dir = os.path.join(src_dir, '不符合条件的图片')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(exclude_dir, exist_ok=True)
    num = 0
    for fn in tqdm(os.listdir(src_dir)):
        # fn = ''
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            src_path = os.path.join(src_dir, fn)
            src_img = cv2.imread(src_path, -1)[:, :, :3]
            model_image_path = os.path.join(model_image_dir, fn)

            img_frt = face_rotate_mine(model_image_path, face_points_model, src_img, face_points_indexes)
            save_path = os.path.join(save_dir, fn)
            if img_frt is None:
                print(src_path)
            cv2.imwrite(save_path, img_frt, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            num += 1

    print(f'转正结束，共转正{num}张图片。')



if __name__ == '__main__':

    test_unfixed()