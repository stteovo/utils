import shutil, cv2, os
import numpy as np
from tqdm import tqdm

def to_uint8(img):
    return img.clip(0, 255).astype(np.uint8)

def ProcessByBGR(raw_engine_obj, input_img, engine_params):
    # 注意 Tetra 的输入输出都是 RGB，不接受其他格式
    src_img = input_img[:, :, ::-1]

    export_img = raw_engine_obj.ExportResult(src_img, repx_params=engine_params)
    assert export_img is not None, "出现错误, 请联系 @FinnChou"

    # RGB2BGR
    export_img = export_img[:, :, ::-1]
    # uint16 -> uint8
    export_img = (export_img.astype(np.float32) / 65535.0 * 255.0)

    return export_img.clip(0, 255).astype(np.uint8)

def ProcessByRGB(raw_engine_obj, input_img, engine_params):
    export_img = raw_engine_obj.ExportResult(input_img, repx_params=engine_params)
    assert export_img is not None, "出现错误, 请联系 @FinnChou"

    # uint16 -> uint8
    export_img = (export_img.astype(np.float32) / 65535.0 * 255.0)

    return to_uint8(export_img)
