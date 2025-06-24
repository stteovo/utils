# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
from utils.ts.rawEngin import TsreRandomParams as tsre_rp
from utils.ts.rawEngin.base import *


def Random_WB():
    # For JPG CCT = 5001, Tint = 10
    return tsre_rp.TsreRandomParams([
        ["Temperature", 3000, 7500, 1.0],
        ["Tint", 0, 20, 1.0],
    ])


def Random_WB_Exposure():
    # For JPG CCT = 5001, Tint = 10
    return tsre_rp.TsreRandomParams([
        ["Temperature", 3000, 7500, 1.0],
        ["Tint", 0, 20, 1.0],
        ["Exposure", -20, 0, 1.0],
    ])


def Random_Calibration():
    return tsre_rp.TsreRandomParams([
        ["RedPrimaryHue", -100, 100, 0.5],
        ["RedPrimarySaturation", -100, 100, 0.5],
        ["GreenPrimaryHue", -100, 100, 1.0],
        ["GreenPrimarySaturation", -100, 100, 0.5],
        ["BluePrimaryHue", -100, 100, 1.0],
        ["BluePrimarySaturation", -100, 100, 0.5],
    ])

def Random_UnderSaturation():
    return tsre_rp.TsreRandomParams([
        ["RedPrimarySaturation", -30, 0, 0.5],
        ["GreenPrimarySaturation", -30, 0, 0.5],
        ["BluePrimarySaturation", -30, 0, 0.5],
    ])

def Random_ReddishFace_For_RedCheek():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -10, 0, 0.1],
        ["Saturation", 5, 40, 1.0],
        ["Vibrance", 0, 100, 1.0],

        ["HSLTunnerHueRed", 20, 80, 0.5],
        ["HSLTunnerHueOrange", -80, -10, 1.0],
        ["HSLTunnerHueYellow", -20, 0, 0.5],

        ["HSLTunnerSaturationRed", 20, 70, 0.6],
        ["HSLTunnerSaturationOrange", 20, 85, 1.0],
        ["HSLTunnerSaturationYellow", 20, 50, 0.6],

        ["HSLTunnerLuminanceRed", -50, -5, 0.5],
        ["HSLTunnerLuminanceOrange", -50, -5, 1.0],
        ["HSLTunnerLuminanceYellow", -40, -5, 0.5],
    ])


def Random_ReddishFace_For_MosquitoBites():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -10, 0, 0.1],
        ["Saturation", 0, 10, 0.75],
        ["Vibrance", 30, 70, 0.75],

        ["HSLTunnerHueRed", 30, 100, 0.5],
        ["HSLTunnerHueOrange", -100, -50, 1.0],
        ["HSLTunnerHueYellow", -40, -20, 0.5],

        ["HSLTunnerSaturationRed", 40, 85, 0.6],
        ["HSLTunnerSaturationOrange", 45, 95, 1.0],
        ["HSLTunnerSaturationYellow", 40, 60, 0.6],

        ["HSLTunnerLuminanceRed", -65, -35, 0.5],
        ["HSLTunnerLuminanceOrange", -65, -35, 1.0],
        ["HSLTunnerLuminanceYellow", -65, -35, 0.5],
    ])

def Random_Body_For_Vitiligo():
    return tsre_rp.TsreRandomParams([
        ["Exposure",      0,  30, 0.1],
        ["Saturation", -75, -20, 0.75],
        ["Vibrance",     -100, -85, 0.75],

        ["HSLTunnerSaturationRed",    -95, -75, 0.6],
        ["HSLTunnerSaturationOrange", -95, -75, 1.0],
        ["HSLTunnerSaturationYellow", -90, -75, 0.6],

        ["HSLTunnerLuminanceRed",    5, 45, 0.5],
        ["HSLTunnerLuminanceOrange", 5, 45, 1.0],
        ["HSLTunnerLuminanceYellow", 5, 45, 0.5],
    ])

def Random_DarkSkin_For_PandaArm():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -30, -10, 1.0],
        ["Saturation", -10, 10, 0.1],
        ["Vibrance", 30, 45, 0.2],

        ["HSLTunnerSaturationRed", None, None, None],
        ["HSLTunnerHueRed", -20, 20, 0.5],
        ["HSLTunnerLuminanceRed", None, None, None],

        ["HSLTunnerHueOrange", -70, 70, 1.0],
        ["HSLTunnerSaturationOrange", -5, 20, 1.0],
        ["HSLTunnerLuminanceOrange", -10, 0, 1.0],
    ])


def Random_DeepDarkSkin_For_PandaEyes():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -45, -35, 1.0],
        ["Saturation", 30, 45, 1.0],
        ["Vibrance", 25, 35, 1.0],

        ["HSLTunnerSaturationRed", 10, 35, 0.7],
        ["HSLTunnerSaturationOrange", 10, 35, 0.7],
        ["HSLTunnerSaturationYellow", 10, 35, 0.7],

        ["HSLTunnerLuminanceRed", -40, -20, 0.5],
        ["HSLTunnerLuminanceOrange", -40, -20, 0.5],
        ["HSLTunnerLuminanceYellow", -40, -20, 0.5],
    ])


def Random_DeepDarkSkin_For_ContourFlaw():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -45, -35, 1.0],
        ["Saturation", 30, 45, 1.0],
        ["Vibrance", 25, 35, 1.0],

        ["HSLTunnerSaturationRed", 10, 35, 0.7],
        ["HSLTunnerSaturationOrange", 10, 35, 0.7],
        ["HSLTunnerSaturationYellow", 10, 35, 0.7],

        ["HSLTunnerLuminanceRed", -40, -20, 0.5],
        ["HSLTunnerLuminanceOrange", -40, -20, 0.5],
        ["HSLTunnerLuminanceYellow", -40, -20, 0.5],
    ])


def Random_NeckColor():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -25, -5, 1.0],
        ["Saturation", 10, 30, 1.0],
        ["Vibrance", 10, 30, 1.0],

        ["HSLTunnerSaturationRed", 10, 35, 0.7],
        ["HSLTunnerSaturationOrange", 10, 50, 0.7],
        ["HSLTunnerSaturationYellow", 10, 50, 0.7],

        ["HSLTunnerLuminanceRed", -20, 0, 0.5],
        ["HSLTunnerLuminanceOrange", -20, 0, 0.5],
        ["HSLTunnerLuminanceYellow", -20, 0, 0.5],
    ])


def Random_Dark():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -90, -30, 1.0],
        # ["Saturation", -100, -50, 0.75],
        # ["Vibrance", -100, -50, 0.75],

        ["HSLTunnerHueRed", 30, 60, 1.0],
        ["HSLTunnerSaturationRed", -90, -50, 1.0],
        # ["HSLTunnerLuminanceRed", None, None, None],

        ["HSLTunnerHueOrange", 30, 60, 1.0],
        ["HSLTunnerSaturationOrange", -90, -50, 1.0],
        # ["HSLTunnerLuminanceOrange", -10, 0, 1.0],
    ])

def Random_Dark2():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -70, 0, 1.0],
    ])

def Random_Skin():
    return tsre_rp.TsreRandomParams([
        ["Exposure", -30, 20, 1.0],
        ["Saturation", 0, 30, 0.5],
        ["Contrast", 0, 50, 1.0],

        ["HSLTunnerHueRed", -10, 10, 0.5],
        ["HSLTunnerSaturationRed", 0, 30, 0.5],
        ["HSLTunnerLuminanceRed", -10, 10, 0.5],

        ["HSLTunnerHueOrange", -10, 10, 0.5],
        ["HSLTunnerSaturationOrange", 0, 30, 0.5],
        ["HSLTunnerLuminanceOrange", -10, 10, 0.5],
    ])


if __name__ == '__main__':

    try:
        import Tetrachromacy as tetra
        print("Init Raw Engine (PyRawEngine Version = %s) ... " % (tetra.__version__))
    except BaseException as be:
        msg = "开始前请先阅读 Readme.md，并确认正确 install 了 Tetra \n"
        print(msg, be)
    raw_engine_obj = tetra.RawEngine()
    raw_engine_obj.InitEnv(type=tetra.EngineType.kSIMD, support_raw=False)

    params_wb_fcn = Random_Skin()

    print(params_wb_fcn)

    params_dict = params_wb_fcn()

    print(params_dict)

    import cv2
    import numpy as np
    # from DataAug.TetraAug.TetraAugEngine import *

    # 使用 CV2 打开图像
    # src_img = cv2.imread("/data_ssd/ay/a_DEBUG/7_1.jpg")

    src_dir = "/data_ssd/ay/neck_color/a_fake_data/train_v1/smodel/yes_PixCake/"
    for fn in os.listdir(src_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != ':':
            fp = os.path.join(src_dir, fn)
            src_img = cv2.imread(fp, -1)[:, :, :3]
            res_img = ProcessByBGR(raw_engine_obj, src_img, params_dict)

            src_img = src_img[:, :, ::-1]
            res_img = res_img[:, :, ::-1]

            # 创建一个包含两个子图的图形窗口
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            # 在第一个子图中显示第一张图像
            ax1.imshow(src_img)
            ax1.set_title('Image 1')
            ax1.axis('off')  # 关闭坐标轴

            # 在第二个子图中显示第二张图像
            ax2.imshow(res_img)
            ax2.set_title('Image 2')
            ax2.axis('off')  # 关闭坐标轴

            # 调整子图之间的间距
            plt.tight_layout()

            # 获取当前图形管理器
            manager = plt.get_current_fig_manager()
            # 设置窗口的位置和大小
            # 注意：不同平台的图形管理器可能有不同的 API，以下是Linux平台（使用 Qt5 后端）的示例
            manager.window.setGeometry(300, 400, 1800, 1200)

            # 显示图形
            plt.show()

    # cv2.imwrite("/data_ssd/ay/a_DEBUG/7_1.png", res_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    # src_rgb_img = src_img[:, :, ::-1]
    # res_rgb_img = ProcessByRGB(raw_engine_obj, src_rgb_img, params_dict)
    # tmp_res = res_rgb_img[:, :, ::-1]
    #
    # show_img = cv2.hconcat([res_img, tmp_res])
    #
    # diff = res_img.astype(np.float32) - tmp_res.astype(np.float32)
    # sum_val = sum(sum(diff))
    # print(sum_val)



    # cv2.imshow('TetraAugEngine-test', show_img)
    # cv2.waitKey()
