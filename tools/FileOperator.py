import shutil, os, cv2
import numpy as np
from tqdm import tqdm

def is_image_file(fn, post_fixes=['jpg', 'jpeg', 'tif', 'bmp', 'png']):
    pre, post = os.path.splitext(fn)
    if '.' not in pre and post.lower() in post_fixes:
        return True

    return False


def process_sameFn_from_refDir(src_dir, ref_dir, trg_dirs=None, conditions=[True, '.png'], file_op='cpy',
                            i=0):
    '''
    复制src_dir中，所有与ref_dir同名的图片，到trg_dirs中
    :param i: 计数
    :param conditions:list-[pre_only, src_fn_postfix, ]:
                                1、pre_only=True只需要pre相同即可;
                                2、src_fn_postfix若不为空，说明src_dir中所有文件后缀名相同;若为空，则需添加新的逻辑
    :param file_ops = ['cpy', 'cut', 'delete']:
                                'delete':安全起见，本质上是把src_dir中的文件，移到一个临时文件夹(src+'/file_to_delete)

    :return:
    '''
    file_ops = ['cpy', 'cut', 'delete']
    assert file_op in file_ops, f'参数file_op必须{file_ops}在中取值'

    if trg_dirs is not None:
        for trg_dir in trg_dirs:
            os.makedirs(trg_dir, exist_ok=True)
    else:
        if file_op == 'delete':
            tmp_dir = os.path.join(src_dir, 'file_to_delete_no')
            os.makedirs(tmp_dir, exist_ok=True)


    for fn in tqdm(os.listdir(ref_dir)):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            i += 1

            # 根据conditions，改造fn来得到，src_fn
            src_fn = fn
            if conditions[0]:
                if conditions[1] != None:
                    src_fn = pre + conditions[1]
                else:
                    pass
            else:   # 只pre相同还不够，需要进一步处理ref_fn
                src_fn = pre + '_原图.jpg'
                if not os.path.exists(os.path.join(src_dir, src_fn)):
                    src_fn = pre + '_原图.JPG'

            src_fp = os.path.join(src_dir, src_fn)
            if not os.path.exists(src_fp):
                i-=1
                continue

            # 复制src_dir中文件至trg_dir
            if file_op == 'cpy':
                for trg_dir in trg_dirs:
                    trg_fp = os.path.join(trg_dir, src_fn)
                    shutil.copy(src_fp, trg_fp)

            elif file_op == 'cut':
                for trg_dir in trg_dirs:
                    trg_fp = os.path.join(trg_dir, src_fn)
                    shutil.move(src_fp, trg_fp)

            elif file_op == 'delete':
                tmp_fp = os.path.join(tmp_dir, src_fn)
                shutil.move(src_fp, tmp_fp)


    print(f'实际操作文件数：{i:>5}个')


if __name__ == '__main__':
    src_dir = '/data_ssd/ay/neck_color/a_fake_data/smodel/'
    ref_dir = '/data_ssd/ay/neck_color/a_fake_data/train_v1/smodel/no'
    trg_dirs = [
        '/data_ssd/ay/neck_color/a_fake_data/smodel/delete/'
    ]

    process_sameFn_from_refDir(src_dir, ref_dir, trg_dirs=None, conditions=[True, '.png'], file_op='delete')
