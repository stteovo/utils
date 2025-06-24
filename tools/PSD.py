import os
import shutil
from utils.tools.General import *
import os
from psd_tools import PSDImage
from PIL import Image


def export_layer(psd, layer_name, output_path):
    """
    导出 PSD 文件中的特定图层并保存为 PNG 文件。

    :param psd: PSDImage 对象
    :param layer_name: 要导出的图层名称
    :param output_path: 输出文件路径
    :return: None
    """
    # 查找目标图层
    target_layer = None
    for layer in psd:
        if layer.name == layer_name or layer.name == '下颌光影塑造（中性灰）_2_2':
            target_layer = layer
            break

    b_exist = False
    for layer in target_layer:
        if '中性灰' in layer.name:
            target_layer = layer
            b_exist = True
            break
    if not b_exist:
        target_layer = target_layer[0]

    if target_layer is None:
        print(f"未找到图层 '{layer_name}'")
        return

    # 将图层转换为 PIL 图像
    pil_image = target_layer.topil()

    # 保存为 PNG 文件
    pil_image.save(output_path)
    print(f"已导出图层 '{layer_name}' 到 {output_path}")


def batch_export_layers(input_folder, output_folder, layer_name):
    """
    批量导出指定文件夹中所有 PSD 文件中的某一特定图层。

    :param input_folder: 包含 PSD 文件的输入文件夹路径
    :param output_folder: 输出文件夹路径
    :param layer_name: 要导出的图层名称
    :return: None
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有 PSD 文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            pre, post = os.path.splitext(file)
            if '.' not in pre and post.lower() == '.psd':
                psd_path = os.path.join(root, file)
                output_path = os.path.join(output_folder, os.path.splitext(file)[0] + '.png')

                try:
                # 打开 PSD 文件
                    psd = PSDImage.open(psd_path)

                    # 导出特定图层
                    export_layer(psd, layer_name, output_path)
                except Exception as e:
                    print(f"处理文件 {file} 时发生错误: {e}")


if __name__ == '__main__':
    src_dir = '/data_ssd/doublechin/data/3已定稿/'
    save_dir = '/data_ssd/materials/jaw_shadow/'
    batch_export_layers(src_dir, save_dir, '下颌光影塑造（中性灰）')