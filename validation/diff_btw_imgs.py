import cv2, os
import numpy as np

'''
计算两个目录中，图片的差异：
    compare_dirs：   需要对比的图片
    ref_fn_dir：     提供原始文件名的目录
    post：           compare_dirs中文件，需要在原始文件名后添加的后缀
'''
def calc_diffs(compare_dirs, ref_fn_dir=None, post_fixes=None, format='.png', diff_save_dir=None):
    if diff_save_dir is not None:
        os.makedirs(diff_save_dir, exist_ok=True)

    for fn in os.listdir(ref_fn_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            # 构造文件名，读取需要对比的图像
            compare_imgs = []
            for i in range(len(compare_dirs)):
                compare_fn = pre + post_fixes[i] + format
                compare_fn = os.path.join(compare_dirs[i], compare_fn)
                compare_img = cv2.imread(compare_fn, -1)
                compare_imgs.append(compare_img)

            # 计算diff
            for i in range(1, len(compare_imgs)):
                diff_img = cv2.absdiff(compare_imgs[i], compare_imgs[i-1])
                diff_img = np.where(diff_img > 0, 255, 0).astype(np.uint8)
                if diff_save_dir is not None:
                    diff_fn = os.path.join(diff_save_dir, pre + '_diff' + post)
                    cv2.imwrite(diff_fn, diff_img)


if __name__ == '__main__':
    compare_dirs = [
        '/data_ssd/doublechin/data/train_717/stage1',
        '/data_ssd/doublechin/data/train_717/stage2'
    ]
    ref_fn_dir = '/data_ssd/doublechin/data/train_717/org'

    diff_save_dir = None
    diff_save_dir = '/data_ssd/doublechin/data/train_717/diff_s12'

    calc_diffs(compare_dirs, ref_fn_dir=ref_fn_dir, post_fixes=['_1', '_2_1'], format='.png', diff_save_dir=diff_save_dir)