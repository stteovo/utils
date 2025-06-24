from lollipop.parsing.human_parsing.HumanParsing import HumanParsing
from lollipop import FacePoint
import os, cv2
import numpy as np


'''
为人脸点匹配相应的分割结果：
    返回值：一个list，其值为对应的hp结果索引，若该人脸点无对应的hp结果，list中对应位置的值为None
'''
def match_fp_hp(instance_mask_all, part_segs, face_points_all, thre_min_intersect=300):
    match_list = [None] * len(face_points_all)
    ms = instance_mask_all * np.float32(1 / 255)
    face_num_max = 0
    for i in range(len(face_points_all)):
        # 根据人脸点画出人脸mask
        face_points = face_points_all[i]
        face_mask_pt = np.zeros_like(instance_mask_all[0], np.uint8)
        cv2.fillPoly(face_mask_pt, [face_points[109:164].astype(np.int32)], 255, 4)

        # 画出的mask与hp结果进行匹配
        for j in range(instance_mask_all.shape[0]):
            m = ms[j]
            face_mask_hp = (np.sum(part_segs[[1, 2]], axis=0) * m).astype(np.uint8)

            # 根据相交面积的阈值和最大值来确定匹配结果
            face_num_max_ = np.logical_and(face_mask_pt > 0, face_mask_hp > 0).astype(np.uint8).sum()
            if face_num_max_ > face_num_max and face_num_max_ > thre_min_intersect:
                match_list[i] = j

    return match_list


def match_hp_pose(img_person, instance_mask_all, part_segs, pose_points_all, thre_min_intersect=300):
    num_instance = instance_mask_all.shape[0]
    if num_instance == 0:
        return None

    match_list = [None] * num_instance
    face_num_max = 0
    for i in range(num_instance):
        ms = instance_mask_all * np.float32(1 / 255)
        face_mask_hp = (np.sum(part_segs[[1, 2]], axis=0) * ms[i]).astype(np.uint8)

        # 画出的mask与hp结果进行匹配
        for j in range(len(pose_points_all)):
            # 根据轮廓点画出大概人脸mask
            pose_face_points = pose_points_all[j, [20, 41, 42]].astype(np.int32)
            face_mask_pt = np.zeros_like(instance_mask_all[0], np.uint8)
            for pt in pose_face_points:
                cv2.drawMarker(img_person, pt, (0, 0, 255), cv2.MARKER_CROSS, 3, 3)
            cv2.fillPoly(face_mask_pt, [pose_face_points], 255, 4)

            # 根据相交面积的阈值和最大值来确定匹配结果
            face_num_max_ = np.logical_and(face_mask_pt > 0, face_mask_hp > 0).astype(np.uint8).sum()
            if face_num_max_ > face_num_max and face_num_max_ > thre_min_intersect:
                match_list[i] = j

    return match_list