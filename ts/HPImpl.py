import os
import numpy as np
import cv2
import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from lollipop.parsing.human_parsing.m2fp import add_m2fp_config


TSHP_PARSING_CATEGORIES = [
    {'id': 1, 'name': 'hair', 'isthing': 1, 'color': [255, 0, 255]},
    {'id': 2, 'name': 'face_skin', 'isthing': 1, 'color': [250, 219, 20]},
    {'id': 3, 'name': 'face_no_skin', 'isthing': 1, 'color': [196, 136, 136]},
    {'id': 4, 'name': 'ear', 'isthing': 1, 'color': [186, 231, 255]},
    {'id': 5, 'name': 'beard', 'isthing': 1, 'color': [89, 56, 19]},
    {'id': 6, 'name': 'neck', 'isthing': 1, 'color': [250, 140, 22]},
    {'id': 7, 'name': 'other_skin', 'isthing': 1, 'color': [147, 235, 191]},
    {'id': 8, 'name': 'arm_left', 'isthing': 1, 'color': [250, 84, 28]},
    {'id': 9, 'name': 'arm_right', 'isthing': 1, 'color': [255, 255, 0]},
    {'id': 10, 'name': 'hand_left', 'isthing': 1, 'color': [39, 107, 9]},
    {'id': 11, 'name': 'hand_right', 'isthing': 1, 'color': [21, 53, 184]},
    {'id': 12, 'name': 'leg_left', 'isthing': 1, 'color': [114, 46, 209]},
    {'id': 13, 'name': 'leg_right', 'isthing': 1, 'color': [192, 168, 231]},
    {'id': 14, 'name': 'cloth', 'isthing': 1, 'color': [45, 231, 231]},
    {'id': 15, 'name': 'belongings', 'isthing': 1, 'color': [255, 255, 255]},
]


def setup_cfg(config_file, opts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_m2fp_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


class HumanParsingImpl(object):
    def __init__(self, config_file=None, weights_file=None, max_size=1024, device='cuda', fp16=False):
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'configs/matting/m2fp_R101.yaml')
        if weights_file is None:
            weights_file = os.path.join(os.path.dirname(__file__), 'weights/tshp_v0.1.pth')
        opts = [
            'MODEL.DEVICE', device,
            'MODEL.WEIGHTS', weights_file,
        ]
        cfg = setup_cfg(config_file, opts)
        self.cfg = cfg

        self.device = device
        assert max_size % cfg.MODEL.M2FP.SIZE_DIVISIBILITY == 0
        self.max_size = max_size
        self.fp16 = fp16

        self.metadata = self.get_metadata()

        self.model = build_model(cfg.clone())
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        if self.device != 'cpu' and self.fp16:
            self.model.half()

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ['RGB', 'BGR'], self.input_format

    @staticmethod
    def get_metadata():
        metadata = {
            'categories': TSHP_PARSING_CATEGORIES,
            'thing_classes': [k['name'] for k in TSHP_PARSING_CATEGORIES],
            'thing_colors': [k['color'] for k in TSHP_PARSING_CATEGORIES],
            'num_parsing': len(TSHP_PARSING_CATEGORIES),
        }
        return metadata

    def run(self, img, refine_matting=False, merge_matting=True, return_ori_size=True):
        """
        human parsing + human matting
        :param img: np.uint8图片
        :param refine_matting: 是否为matting结果增加半透明效果，如果matting模型本身支持则无需开启此选项
        :param merge_matting: 是否将matting结果融合至parsing
        :param return_ori_size: 返回结果是否resize回原始图分辨率，否则只是保留模型输出的分辨率
        :return:
            result: {
                "fg_seg": 前景分割结果，np.uint8数组，shape=(h, w)
                "human_instance_seg": 人体实例分割结果，np.uint8数组，shape=(n, h, w)，n为人数
                "part_sem_seg": part语义分割结果，np.uint8数组，shape=(15, h, w)，类别定义参考self.metadata
                "alpha": matting结果，只有在merge_matting=False的时候才有此字段，np.uint8数组，shape=(h, w)
            }
        """
        oh, ow = img.shape[0:2]
        s = self.max_size/max(ow, oh)
        nw, nh = int(round(ow*s)), int(round(oh*s))

        interpolation = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        img2 = cv2.resize(img, (nw, nh), interpolation=interpolation)
        if self.input_format == 'RGB':
             img2 = img2[..., ::-1]
        img2_t = torch.as_tensor(img2.transpose(2, 0, 1).copy())

        input = {'image': img2_t}
        with torch.no_grad():
            result = self.model([input])[0]

        if refine_matting:
            result = self._refine_matting(result)

        if merge_matting:
            result = self._merge_parsing_and_matting(result)

        result2 = {
            'fg_seg': (result['fg_seg']*255).astype(np.uint8),
            'human_instance_seg': (result['human_instance_seg']*255).astype(np.uint8),
            'part_sem_seg': (result['part_sem_seg']*255).astype(np.uint8),
        }
        if not merge_matting:
            result2['alpha'] = (result['alpha']*255).astype(np.uint8)
        result = result2

        # TODO: 牺牲一点精度损失，换取更高的数据压缩率
        _, filter = cv2.threshold(result['fg_seg'], 5, 1, cv2.THRESH_BINARY)
        result['part_sem_seg'] *= filter

        if return_ori_size:
            if len(result['human_instance_seg']) > 0:
                human_instance_seg = np.stack([cv2.resize(x, (ow, oh)) for x in result['human_instance_seg']], axis=0)
            else:
                human_instance_seg = np.empty((0, oh, ow), dtype=result['human_instance_seg'].dtype)
            result2 = {
                'fg_seg': cv2.resize(result['fg_seg'], (ow, oh)),
                'human_instance_seg': human_instance_seg,
                'part_sem_seg': np.stack([cv2.resize(x, (ow, oh)) for x in result['part_sem_seg']], axis=0),
            }
            if not merge_matting:
                result2['alpha'] = cv2.resize(result['alpha'], (ow, oh))
            result = result2

        return result

    def get_skin_mask(self, result):
        skin = np.sum(result['part_sem_seg'][[1,3,5,6,7,8,9,10,11,12]], axis=0)
        fg_seg = result['fg_seg']*np.float32(1/255)
        return (skin*fg_seg).astype(np.uint8)

    def get_face_neck_mask(self, result):
        face_neck = np.sum(result['part_sem_seg'][[1, 4, 5]], axis=0)
        fg_seg = result['fg_seg']*np.float32(1/255)
        return (face_neck*fg_seg).astype(np.uint8)

    def get_hair_mask(self, result):
        hair = result['part_sem_seg'][0]
        fg_seg = result['fg_seg']*np.float32(1/255)
        return (hair*fg_seg).astype(np.uint8)

    def visualize(self, img, result):
        colors = [color[::-1] for color in self.metadata['thing_colors']]

        fg_seg = result['fg_seg'] / np.float32(255)
        human_instance_seg = result['human_instance_seg'] / np.float32(255)
        part_sem_seg = result['part_sem_seg'] / np.float32(255)

        part = np.sum(part_sem_seg[..., None]*np.array(colors)[:, None, None], axis=0).astype(np.uint8)
        part = (img*(1-fg_seg[..., None])+part*fg_seg[..., None]).astype(np.uint8)
        alpha = 0.7
        part = cv2.addWeighted(img, 1-alpha, part, alpha, 0)

        if len(human_instance_seg) > 0:
            colors2 = [colors[(i+1) % len(colors)] for i in range(len(human_instance_seg))]
            human = np.sum(human_instance_seg[..., None]*np.array(colors2)[:, None, None], axis=0).astype(np.uint8)
        else:
            human = np.zeros_like(img, dtype=np.uint8)
        alpha = 0.7
        human = cv2.addWeighted(img, 1-alpha, human, alpha, 0)

        return np.hstack([img, part, human])

    def _refine_matting(self, result):
        """
        parsing背景上衣物的半透明效果迁移给matting
        """
        fg_seg = result['fg_seg']
        part_sem_seg = result['part_sem_seg']
        alpha = result['alpha']

        bg = 1.0-fg_seg
        cloth = part_sem_seg[13]*fg_seg
        mask = ((cloth > 0.1) & (bg > 0.1) & (bg+cloth > 0.5)).astype(np.float32)
        # 形态学操作避免不透明衣物边缘误伤
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=3)
        mask = cv2.dilate(mask, kernel, iterations=5)
        # 羽化过渡更自然
        mask = cv2.blur(mask, (5, 5))
        alpha = fg_seg*mask+alpha*(1-mask)

        result2 = dict(result)
        result2['alpha'] = alpha

        return result2

    def _merge_parsing_and_matting(self, result):
        """
        matting结果融合至parsing每个instance
        """
        fg_seg = result['fg_seg']
        human_instance_seg = result['human_instance_seg']
        part_sem_seg = result['part_sem_seg']
        alpha = result['alpha']

        if len(human_instance_seg) == 0 or len(human_instance_seg) > 8:
            pass
        elif len(human_instance_seg) == 1:
            human_instance_seg = alpha[None].copy()
            fg_seg = alpha.copy()
        else:
            # 前景置信度低的像素直接采纳离它最近的高置信度像素的人体实例分割结果
            unknown = ((fg_seg < 0.1)*255).astype(np.uint8)
            dst, labels = cv2.distanceTransformWithLabels(unknown, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
            zero_xys = np.stack(np.where(unknown == 0)[::-1], axis=1)
            map_xy = zero_xys[labels-1]
            map_xy = np.float32(map_xy)
            human_instance_seg = np.stack([
                cv2.remap(seg, map_xy, map2=None, interpolation=cv2.INTER_LINEAR)
                for seg in human_instance_seg
            ], axis=0)

            # 归一化
            human_instance_seg += 1e-6
            human_instance_seg *= alpha/np.sum(human_instance_seg, axis=0)
            fg_seg = alpha.copy()

        result = {
            'fg_seg': fg_seg,
            'human_instance_seg': human_instance_seg,
            'part_sem_seg': part_sem_seg,
        }

        return result

if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    from lollipop.parsing.human_parsing.human_parsing_impl import HumanParsingImpl
    # root = Path('/data_ssd/ay/头发分类/dataset/train_2/')
    # root = Path('/data_ssd/ay/头发分类/dataset/1/')
    root = Path('/data_ssd/ay/头发分类/dataset/test/')
    hair_mask_dir = root / 'hair_mask'
    os.makedirs(hair_mask_dir, exist_ok=True)

    model = HumanParsingImpl()
    for img_path in tqdm(root.glob('*.jpg')):
        img = cv2.imread(str(img_path), -1)
        if img is None:
            continue

        '''human parsing'''
        result = model.run(img)
        hair_mask = model.get_hair_mask(result)
        mask_path = hair_mask_dir / f'{img_path.stem}.png'
        cv2.imwrite(mask_path, hair_mask)
