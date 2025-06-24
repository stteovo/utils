#!/usr/bin/python2.7
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from lollipop.warping import umeyama_alignment
from lollipop import FacePoint


def is_image(model_image):
    if isinstance(model_image, np.ndarray):
        if model_image.dtype != np.uint8:
            return False
        if len(model_image.shape) == 3 and model_image.shape[-1] in (3, 4):
            return True
        elif len(model_image.shape) == 2:
            return True
        return False
    else:
        return False



class FaceRotate:
    '''
    Examples::
    >>> frt = FaceRotate(tar_image)
    >>> #warp the src
    >>> output = frt(src_image)
    >>>
    >>> #warp the list
    >>> src_part, mask_part = frt([src_image, src_mask])
    >>> # src_part.shape == tar_image.shape
    >>>
    >>> # backfill the part to src
    >>> restore = frt.backfill(src_image, src_part)
    >>> # restore.shape == src_image.shape
    '''
    def __init__(self, model_image=None, face_index=0):
        super(FaceRotate, self).__init__()
        self.rotate_M = None

        self.fpm = FacePoint()
        if model_image is None:
            self.model_image_data = None
        else:
            if isinstance(model_image, str):
                model_image = cv2.imread(model_image)

            if is_image(model_image):
                tar_points = self.fpm(model_image)[face_index]
                th, tw = model_image.shape[:2]
            elif isinstance(model_image, (tuple, list)):
                if len(model_image) == 2:
                    tar_points, tar_shape = model_image
                    th, tw = tar_shape[:2]
                elif len(model_image) == 3:
                    tar_points, th, tw = model_image
                else:
                    assert False
            else:
                assert False

            self.model_image_data = [tar_points, th, tw]

    def select_points(self, points, index=None):
        if index:
            return points[index]
        return points

    def __call__(self, src_image, src_fp=None, idxes=None):
        if isinstance(src_image, list):
            org_src = src_image[0]
        else:
            org_src = src_image
        if src_fp is None:
            src_fp = self.fpm(org_src)
            if src_fp is None or len(src_fp) == 0:
                return None
            src_fp = src_fp[0]

        tarp, th, tw = self.model_image_data
        r, t, c, M = umeyama_alignment(self.select_points(src_fp, index=idxes), self.select_points(tarp, index=idxes))
        M = np.array([
            [c * r[0, 0], c * r[0, 1], t[0]],
            [c * r[1, 0], c * r[1, 1], t[1]],
        ]).astype(np.float32)
        part_alpha = cv2.warpAffine(np.zeros_like(org_src[:, :, :1], np.uint8) + 255, M[:2], (tw, th))
        self.rotate_M = [M, part_alpha]

        if isinstance(src_image, list):
            return [cv2.warpAffine(image, M[:2], (tw, th)) for image in src_image]
        else:
            output = cv2.warpAffine(org_src, M[:2], (tw, th))
            return output

    def dewarp(self, src_image, partImage, border_color=0):
        if self.rotate_M is None:
            assert False, 'Please execute function __call__() before dewarp().'
        sh, sw = src_image.shape[:2]
        M, part_alpha = self.rotate_M
        M3 = np.eye(3)
        M3[:2] = M
        output = cv2.warpAffine(partImage, np.linalg.inv(M3)[:2], (sw, sh), borderValue=border_color)
        return output

    def backfill(self, src_image, partImage):
        part_one = cv2.merge([partImage, self.rotate_M[1]])
        rebuild_part = self.dewarp(src_image, part_one)
        bgr = rebuild_part[:, :, :-1]
        alpha = rebuild_part[:, :, -1:].astype(np.float32) / 255

        output = np.clip(bgr + (1 - alpha) * src_image, 0, 255).astype(np.uint8)
        return output


class EyeRotate(FaceRotate):
    eye_index = np.array([51, 52, 53, 54, 55, 56, 57, 58, 59]).astype(np.int64)

    def __init__(self, model_image=None, face_index=0, pad_v=16):
        super(EyeRotate, self).__init__(model_image, face_index)
        x, y, w, h = cv2.boundingRect(self.model_image_data[0][self.eye_index])
        self.model_image_data[0] = self.model_image_data[0] - np.array([x - pad_v, y - pad_v])[np.newaxis]
        self.model_image_data[1] = h + pad_v * 2
        self.model_image_data[2] = w + pad_v * 2

    def select_points(self, points):
        return points[self.eye_index].T


class FaceRotateList(FaceRotate):
    def __init__(self, model_image=None, face_index=0):
        super(FaceRotateList, self).__init__(model_image, face_index)
        self.rotate_list = []

    def __call__(self, src_image, src_fp=None):
        if isinstance(src_image, list):
            org_src = src_image[0]
        else:
            org_src = src_image
        if src_fp is None:
            src_fp = self.fpm(org_src)
            if src_fp is None or len(src_fp) == 0:
                return None
        tarp, th, tw = self.model_image_data
        self.rotate_list = []
        for src_ in src_fp:
            r, t, c, M = umeyama_alignment(self.select_points(src_), self.select_points(tarp))
            M = np.array([
                [c * r[0, 0], c * r[0, 1], t[0]],
                [c * r[1, 0], c * r[1, 1], t[1]],
            ]).astype(np.float32)

            part_alpha = cv2.warpAffine(np.zeros_like(org_src[:, :, :1], np.uint8) + 255, M[:2], (tw, th))
            self.rotate_list.append((M, part_alpha))

        if isinstance(src_image, list):
            return [[cv2.warpAffine(image, M[:2], (tw, th)) for M, _ in self.rotate_list] for image in src_image]
        else:
            output = [cv2.warpAffine(org_src, M[:2], (tw, th)) for M, _ in self.rotate_list]
            return output

    def dewarp(self, src_image, partImage, border_color=0):
        print('FaceRotateList 不提供单独的逆转正过程')
        assert False, NotImplemented

    def backfill(self, src_image, partImage):
        sh, sw = src_image.shape[:2]
        output = src_image
        for part_, rebuild_data in zip(partImage, self.rotate_list):
            part_one = cv2.merge([part_, rebuild_data[1]])

            M, part_alpha = rebuild_data
            M3 = np.eye(3)
            M3[:2] = M
            rebuild_part = cv2.warpAffine(part_one, np.linalg.inv(M3)[:2], (sw, sh), borderValue=0)

            bgr = rebuild_part[:, :, :-1]
            alpha = rebuild_part[:, :, -1:].astype(np.float32) / 255
            output = np.clip(bgr + (1 - alpha) * output, 0, 255).astype(np.uint8)
        return output


class myFaceRotate:
    '''
    Examples::
    >>> frt = FaceRotate(tar_image)
    >>> #warp the src
    >>> output = frt(src_image)
    >>>
    >>> #warp the list
    >>> src_part, mask_part = frt([src_image, src_mask])
    >>> # src_part.shape == tar_image.shape
    >>>
    >>> # backfill the part to src
    >>> restore = frt.backfill(src_image, src_part)
    >>> # restore.shape == src_image.shape
    '''
    def __init__(self, model_image=None, face_index=0):
        super(myFaceRotate, self).__init__()
        self.rotate_M = None

        self.fpm = FacePoint()
        if model_image is None:
            self.model_image_data = None
        else:
            if isinstance(model_image, str):
                model_image = cv2.imread(model_image)

            if is_image(model_image):
                tar_points = self.fpm(model_image)[face_index]
                th, tw = model_image.shape[:2]
            elif isinstance(model_image, (tuple, list)):
                if len(model_image) == 2:
                    tar_points, tar_shape = model_image
                    th, tw = tar_shape[:2]
                elif len(model_image) == 3:
                    tar_points, th, tw = model_image
                else:
                    assert False
            else:
                assert False

            self.model_image_data = [tar_points, th, tw]

    def select_points(self, points, index=None):
        if index:
            return points[index]
        return points

    def __call__(self, src_image, src_fp=None, idxes=None):
        if isinstance(src_image, list):
            org_src = src_image[0]
        else:
            org_src = src_image
        if src_fp is None:
            src_fp = self.fpm(org_src)
            if src_fp is None or len(src_fp) == 0:
                return None
            src_fp = src_fp[0]

        tarp, th, tw = self.model_image_data
        r, t, c, M = umeyama_alignment(self.select_points(src_fp, index=idxes), self.select_points(tarp, index=idxes))
        M = np.array([
            [c * r[0, 0], c * r[0, 1], t[0]],
            [c * r[1, 0], c * r[1, 1], t[1]],
        ]).astype(np.float32)
        part_alpha = cv2.warpAffine(np.zeros_like(org_src[:, :, :1], np.uint8) + 255, M[:2], (tw, th))
        self.rotate_M = [M, part_alpha]

        if isinstance(src_image, list):
            return [cv2.warpAffine(image, M[:2], (tw, th)) for image in src_image]
        else:
            output = cv2.warpAffine(org_src, M[:2], (tw, th))
            return output

    def dewarp(self, src_image, partImage, border_color=0):
        if self.rotate_M is None:
            assert False, 'Please execute function __call__() before dewarp().'
        sh, sw = src_image.shape[:2]
        M, part_alpha = self.rotate_M
        M3 = np.eye(3)
        M3[:2] = M
        output = cv2.warpAffine(partImage, np.linalg.inv(M3)[:2], (sw, sh), borderValue=border_color)
        return output

    def backfill(self, src_image, partImage):
        part_one = cv2.merge([partImage, self.rotate_M[1]])
        rebuild_part = self.dewarp(src_image, part_one)
        bgr = rebuild_part[:, :, :-1]
        alpha = rebuild_part[:, :, -1:].astype(np.float32) / 255

        output = np.clip(bgr + (1 - alpha) * src_image, 0, 255).astype(np.uint8)
        return output


if __name__ == '__main__':
    import os
    from lollipop import lollipop_root_path, FacePoint
    fpm = FacePoint()
    B = cv2.imread(os.path.join(lollipop_root_path, 'model_2.jpg'))
    A = cv2.imread(os.path.join(lollipop_root_path, 'model.jpg'))
    print('推荐使用方式:')
    frt = FaceRotate(B)

    face_part = frt(A)

    fp = fpm(A)[0]
    M = frt.rotate_M[0]
    fp = fp @ M[:, :2].T + M[:, -1]
    fpm.draw_point(face_part, fp)
    cv2.imshow("A", A)
    cv2.imshow(",", face_part)
    cv2.waitKey()

    rebuild_image = frt.backfill(A, face_part)

    cv2.imshow('A', A)
    cv2.imshow('B', B)
    cv2.imshow('face_part', face_part)
    cv2.imshow('rebuild_image', rebuild_image)
    cv2.waitKey()

    print('其他使用方式')
    if True:
        # 以path构建转正模版
        frt = FaceRotate(os.path.join(lollipop_root_path, 'model_2.jpg'), face_index=1)
        face_part = frt(A)
        rebuild_image = frt.backfill(A, face_part)

    if True:
        # 以人脸点+目标尺寸构建转正模板
        tar_data = [fpm(B)[0], B.shape]
        frt = FaceRotate(tar_data)

    if True:
        # 以人脸点+目标尺寸构建转正模板
        h, w = B.shape[:2]
        tar_data = [fpm(B)[0], h, w]
        frt = FaceRotate(tar_data)

    if True:
        # 多个对象同步转正
        face_part, face_gray = frt([A, cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)])
        cv2.imshow('face_part', face_part)
        cv2.imshow('face_gray', face_gray)

    if True:
        # 五官转正
        ert = EyeRotate(B)
        ert.fpm.draw_point(A, ert.fpm(A)[0][ert.eye_index])
        eye_part = ert(A)
        cv2.imshow('eye_part', eye_part)

    cv2.imshow('rebuild_image', rebuild_image)
    cv2.waitKey()

    frt = FaceRotateList(A)
    face_part = frt(B)
    rebuild_image = frt.backfill(B, [(part * 0.5).astype(np.uint8) for part in face_part])

    [cv2.imshow('face_part' + str(i), face_) for i, face_ in enumerate(face_part)]
    cv2.imshow('rebuild_image', rebuild_image)
    cv2.waitKey()
