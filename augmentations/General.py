import numpy as np
import cv2, os


def generate_random_affine_matrix(translate_range=(-0.1, 0.1), angle_range=(-10, 10), scale_range=(0.9, 1.1),
                                  shear_range=None, dtype=np.float32, p_angle=0.15, p_shear=0.15, ref_center=None, keep_ratio=False):
    """
    生成一个随机的仿射变换矩阵。

    :param angle_range: 旋转角度范围 (degree)，元组形式 (min_angle, max_angle)。
    :param scale_range: 缩放范围，元组形式 (min_scale, max_scale)。
    :param translate_range: 平移范围，元组形式 (min_translation, max_translation)，单位为图像宽度或高度的比例。
    :param shear_range: 剪切角度范围 (degree)，元组形式 (min_shear, max_shear)。
    :param ref_center: 旋转+缩放的仿射矩阵的参考点，若为None，则使用原点
    :param keep_ratio: 只保证scale_x、scale_y是否相等，不影响shear

    :return: 仿射变换矩阵 (3x3)。
    """
    # 生成随机角度
    angle = np.random.uniform(*angle_range) if np.random.uniform() < p_angle else 0
    # 生成随机缩放因子
    scale_x = np.random.uniform(*scale_range)
    scale_y = scale_x if keep_ratio else np.random.uniform(*scale_range)
    # 生成随机平移因子
    translate_x = np.random.uniform(*translate_range)
    translate_y = np.random.uniform(*translate_range)


    '''构造仿射变换矩阵'''
    if ref_center is not None:
        # 生成旋转+缩放的仿射矩阵，围绕中心点
        center_x, center_y = ref_center
        affine_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle, scale_x)
        affine_matrix[0][2] = translate_x
        affine_matrix[1][2] = translate_y
    else:
        radians = np.deg2rad(angle)
        affine_matrix = np.array([
            [np.cos(radians) * scale_x, np.sin(radians) * scale_x, translate_x],
            [-np.sin(radians) * scale_y, np.cos(radians) * scale_y, translate_y],
        ])


    # 生成随机剪切角度
    if shear_range is not None and np.random.uniform() < p_shear:
        shear = np.random.uniform(*shear_range)
        shear_radians = np.deg2rad(shear)
        # 添加剪切部分
        affine_matrix[0][1] += np.tan(shear_radians)
        affine_matrix[1][0] += np.tan(shear_radians)

    affine_matrix = np.vstack([affine_matrix, [0, 0, 1]])

    return affine_matrix.astype(dtype)


'''
叠加高斯噪声
'''
def get_gaussian_noise(input_shape, mean=0, stddev=1, depth=1, dtype=np.float32):
    if depth == 1:
        noise = np.random.normal(mean, stddev, input_shape)
        return noise.astype(dtype)

    assert input_shape % 2**depth == 0, "input_size、depth不匹配 !!!"

    noises = np.zeros([depth, *input_shape], dtype=dtype)
    for i in range(depth):
        up_scale = pow(2, i)
        cur_shape = [input_shape[0] // up_scale, input_shape[1] // up_scale]
        noises[i] = np.random.normal(mean, stddev, cur_shape)
        '''根据i的值进行不同程度的上菜样'''
        if i > 1:
            noises[i] = cv2.resize(noises[i], None, fx=up_scale, fy=up_scale, interpolation=cv2.INTER_LINEAR)

    noise = np.sum(noises, axis=0) / depth
    return noise.astype(dtype)


if __name__ == '__main__':
    image = cv2.imread("/data_ssd/ay/a_DEBUG/1.png", -1).astype(np.float32)
    noise = get_gaussian_noise([512, 512, 3], 0, 15, depth=1)
    dst = cv2.add(image, noise)
    dst = np.clip(dst, 0, 255).astype(np.uint8)
    cv2.imwrite('/data_ssd/ay/a_DEBUG/gaussian_noise.png', noise)