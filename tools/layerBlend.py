import numpy as np
import cv2

np.seterr(divide='ignore', invalid='ignore')


# "颜色"叠加
def blend_color(base_img, blend_img):
    base_hsl = cv2.cvtColor(base_img, cv2.COLOR_BGR2HSV)
    blend_hsl = cv2.cvtColor(blend_img, cv2.COLOR_BGR2HSV)

    base_h, base_s, base_v = cv2.split(base_hsl)
    blend_h, blend_s, blend_v = cv2.split(blend_hsl)

    res_img_hsl = cv2.merge([blend_h, blend_s, base_v])
    return cv2.cvtColor(res_img_hsl, cv2.COLOR_HSV2BGR)


# "正片叠底"叠加
def blend_multiply(base_img, blend_img):
    # 将图像转换为浮点数
    base_img = base_img.astype(float)
    blend_img = blend_img.astype(float)

    blended_image = cv2.multiply(base_img, blend_img / 255)

    # 将结果转换回整数类型
    blended_image = blended_image.astype(np.uint8)

    return blended_image


# "滤色"叠加
def blend_screen(base_img, blend_img):
    # 将图像转换为浮点数
    base_img = base_img.astype(float)
    blend_img = blend_img.astype(float)

    blended_image = cv2.multiply((255 - base_img), (255 - blend_img))

    blended_image = 255 - blended_image / 255
    # 将结果转换回整数类型
    blended_image = blended_image.astype(np.uint8)

    return blended_image


# "强光"叠加
def blend_hard_light(base_img, blend_img):
    # 将图像转换为浮点数
    base_img = base_img.astype(float) / 255
    blend_img = blend_img.astype(float) / 255

    blended_image = np.zeros_like(base_img)
    mask = base_img < 0.5
    blended_image[mask] = 2 * base_img[mask] * blend_img[mask]
    blended_image[~mask] = 1 - 2 * (1 - base_img[~mask]) * (1 - blend_img[~mask])

    # 将结果转换回0-255范围内的整数
    blended_image = (blended_image * 255).astype(np.uint8)

    return blended_image


# "柔光"叠加
def blend_soft_light(base_img, blend_img):
    # 将图像转换为浮点数
    base_img = base_img.astype(float) / 255
    blend_img = blend_img.astype(float) / 255

    blended_image = np.zeros_like(base_img)
    mask = base_img < 0.5
    blended_image[mask] = 2 * base_img[mask] * blend_img[mask] + base_img[mask] ** 2 * (1 - 2 * blend_img[mask])
    blended_image[~mask] = 2 * base_img[~mask] * (1 - blend_img[~mask]) + \
                           np.sqrt(base_img[~mask]) * (2 * blend_img[~mask] - 1)

    # 将结果转换回0-255范围内的整数
    blended_image = (blended_image * 255).astype(np.uint8)

    return blended_image


def blend_operation(base_img, blend_img, blend_type="multiply"):
    if blend_type == "multiply":
        blend_img = blend_multiply(base_img, blend_img)
    elif blend_type == "color":
        blend_img = blend_color(base_img, blend_img)
    elif blend_type == "screen":
        blend_img = blend_screen(base_img, blend_img)
    elif blend_type == "hard light":
        blend_img = blend_hard_light(base_img, blend_img)
    elif blend_type == "soft light":
        blend_img = blend_soft_light(base_img, blend_img)
    else:
        blend_img = base_img
    return blend_img


def blend_with_color_layer(image, color, mask, blend_type="multiply"):
    coord_list = np.where(mask > 0)
    if coord_list[0].size < 1:
        return image
    # coord_y_list = coord_list[0]
    # coord_x_list = coord_list[1]
    left_pos = np.min(coord_list[1])
    right_pos = np.max(coord_list[1])
    top_pos = np.min(coord_list[0])
    bottom_pos = np.max(coord_list[0])

    roi_img = image[top_pos:bottom_pos+1, left_pos:right_pos+1, :]
    color_layer = np.full(roi_img.shape, color, dtype=np.uint8)
    blend_img = blend_operation(base_img=roi_img, blend_img=color_layer, blend_type=blend_type)

    output_img = image.copy()
    if mask is None:
        output_img[top_pos:bottom_pos+1, left_pos:right_pos+1, :] = blend_img
    else:
        roi_mask = mask[top_pos:bottom_pos + 1, left_pos:right_pos + 1, :]
        output_img[top_pos:bottom_pos+1, left_pos:right_pos+1, :] = \
            image[top_pos:bottom_pos+1, left_pos:right_pos+1, :] * (1.0 - roi_mask) + blend_img * roi_mask
    return np.clip(output_img, 0, 255).astype(np.uint8)


if __name__ == '__main__':
    pass