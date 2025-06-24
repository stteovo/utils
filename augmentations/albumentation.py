import albumentations as A
import cv2
import numpy as np
from utils.visulization.realTimeVisual import visualize
augmentation = A.Compose([
    # A.OneOf([

    # ], p=0.2),

    # 模糊退化
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.Blur(blur_limit=(3, 7), p=0.5),
        A.MedianBlur(blur_limit=(3, 7), p=0.5),
        A.MotionBlur(blur_limit=(3, 7), p=0.5),
    ], p=0.2),

    # 退化
    A.OneOf([
        A.GaussNoise(var_limit=(0.0, 75.0), p=0.5),
        A.ImageCompression(quality_lower=30, quality_upper=70, p=0.5),
    ], p=0.2),

    # 颜色
    A.HueSaturationValue(hue_shift_limit=(-2, 2), sat_shift_limit=(-30, 30), val_shift_limit=0, p=0.3),

    # 亮度
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.1), contrast_limit=(-0.2, 0.1), p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], p=0.2),

    ]
    # , additional_targets={'image2': 'image'}
    )

def test():
    # 定义变换
    augmentation = A.Compose([
        # 模糊退化
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            A.Blur(blur_limit=7, p=0.5),
            A.MedianBlur(blur_limit=7, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.5),
        ], p=1.0),

        # # 退化
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # A.JpegCompression(quality_lower=70, quality_upper=100, p=0.5),
        #
        # # 颜色
        # A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        #
        # # 亮度
        # A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=0.5),
        # A.RandomGamma(gamma_limit=(80, 120), p=0.5),

    ], additional_targets={'image2': 'image'})

    # 加载图像
    image = cv2.imread('path_to_your_image.png')  # 替换为你的图像路径
    image2 = cv2.imread('path_to_another_image.png')  # 替换为另一个图像路径

    # 应用变换
    augmented = augmentation(image=image, image2=image2)

    # 获取变换后的图像
    transformed_image = augmented['image']
    transformed_image2 = augmented['image2']


def process_func(img):
    # 定义变换
    global augmentation

    # 应用变换
    augmented = augmentation(image=img[:, :, :3])

    return augmented['image']


if __name__ == '__main__':
    visualize(process_func)