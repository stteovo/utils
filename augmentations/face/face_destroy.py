import numpy as np
import cv2, os, random
from lollipop import liquefy, liquefy_image, liquefy_display


class FaceDestroy:
    def __init__(self):

        self.size = [2000, 2000]

    '''液化'''

    def liquefy_image(self, image, from_point2d, to_point2d, r_max=-1, pressure=100, ratio=(1, 1), face_mask=None,
                      bm=cv2.BORDER_CONSTANT, bv=(255, 255, 255)):
        h, w = image.shape[:2]
        xy_map = cv2.merge(np.meshgrid(range(w), range(h)))
        xy_map = liquefy(xy_map, from_point2d, to_point2d, r_max, pressure, ratio, face_mask)
        output = cv2.remap(image, xy_map, None, cv2.INTER_LINEAR, borderMode=bm, borderValue=bv)
        return output

    def hole_cloth_liquefy_neck(self, img, pre_cloth_mask, to_shift_max=(15, 15), ratio_range=(0.5, 1.0), dir=None,
                                i=None):
        h, w = pre_cloth_mask.shape[:2]
        l_pt, r_pt = self.calc_points(pre_cloth_mask)
        l_x, r_x = l_pt[0], r_pt[0]
        l_y, r_y = np.argmax(pre_cloth_mask[:, l_x]), np.argmax(pre_cloth_mask[:, r_x])

        x_shift, y_shift = np.random.randint(3, to_shift_max[0]), np.random.randint(3, to_shift_max[1])
        ratio_num1, ratio_num2 = np.random.uniform(ratio_range[0], ratio_range[1]), np.random.uniform(ratio_range[0],
                                                                                                      ratio_range[1])
        ratio1, ratio2 = (ratio_num1, ratio_num1), (ratio_num2, ratio_num2)
        if np.random.uniform() < 0.3:
            ratio_num11, ratio_num22 = np.random.uniform(ratio_range[0], ratio_range[1]), np.random.uniform(
                ratio_range[0], ratio_range[1])
            ratio1, ratio2 = (ratio_num11, ratio_num1), (ratio_num22, ratio_num2)

        from_ = [[l_x, l_y], [r_x, r_y]]
        to_ = [[l_x - x_shift, l_y - y_shift], [r_x + x_shift, r_y - y_shift]]

        img_ = self.liquefy_image(img, np.array(from_[0]), np.array(to_[0]), ratio=ratio1)
        img_ = self.liquefy_image(img_, np.array(from_[1]), np.array(to_[1]), ratio=ratio2)

        # if i and dir:
        #     cv2.imwrite(os.path.join(dir, str(i) + '.png'), img)
        #     cv2.imwrite(os.path.join(dir, str(i) + '_liq.png'), img_)

        return img_

    def affine_point(self, points, M):
        if not isinstance(points, np.ndarray):
            points = np.array(points, np.float32)
        if points.ndim == 1:
            points = points[None, :]

        points = np.concatenate([points.transpose(1, 0), np.ones(points.shape[0], dtype=M.dtype)[None]])
        M = np.concatenate([M, np.array([0, 0, 1], dtype=M.dtype)[None]])
        points = np.matmul(M, points)
        points = points[:2].transpose(1, 0)

        return points.astype(np.int32)

    def face_mask_destroy(self, face_mask, points, model_neck_mask):# 822, 1177
        face_points, neck_points = points[:-2, ], points[-2:]
        model_neck_points = np.array([[822, 1000], [1177, 1000]])
        scale_center = (1000, 1090)

        # 缩小模板脖子mask宽度，使其小于脖子轮廓点的宽度
        model_neck_w = 1177 - 822
        cur_neck_w = neck_points[1][0] - neck_points[0][0]
        trg_neck_w = cur_neck_w - np.random.randint(15, 70)
        scale_neck = trg_neck_w / model_neck_w
        M_neck = cv2.getRotationMatrix2D(scale_center, 0, scale_neck)
        cur_model_neck_mask = cv2.warpAffine(model_neck_mask, M_neck, self.size)
        cur_model_neck_points = self.affine_point(model_neck_points, M_neck)
        cur_face_neck_points = self.affine_point(neck_points, M_neck)

        # 开始在左右两边进行破坏
        # 按范围进行分类:左右两边各以cur_face_neck_points分界，分别进行破坏
        '''原脖子区域外进行破坏'''
        h_l_destroy = np.random.randint(10, 90)
        h_r_destroy = np.random.randint(10, 90)
        l_inner = True if np.random.uniform() < 0.8 else False
        r_inner = True if np.random.uniform() < 0.8 else False
        face_mask_destroy = np.copy(face_mask)
        face_mask_binary = np.where(face_mask > 127, 255, 0).astype(np.uint8)
        for i in range(1):
            face_mask_destroy = self.draw_destroy_mask_v1(face_mask_destroy, face_mask_binary, h_l_destroy, cur_face_neck_points, b_left=True, inner=l_inner)
            face_mask_destroy = self.draw_destroy_mask_v1(face_mask_destroy, face_mask_binary, h_r_destroy, cur_face_neck_points, b_left=False, inner=r_inner)

        '''原脖子区域内进行破坏'''
        r_w_max = max(3, cur_model_neck_points[1][0] - model_neck_points[1][0])
        l_w_max = max(3, model_neck_points[0][0] - cur_model_neck_points[0][0])
        l_inner = True if np.random.uniform() < 0.9 else False
        r_inner = True if np.random.uniform() < 0.9 else False
        for i in range(1):
            face_mask_destroy = self.draw_destroy_mask_v2(face_mask_destroy, face_mask_binary, l_w_max, cur_model_neck_points, b_left=True, inner=l_inner)
            face_mask_destroy = self.draw_destroy_mask_v2(face_mask_destroy, face_mask_binary, r_w_max, cur_model_neck_points, b_left=False, inner=r_inner)


        cur_face_neck_mask = cv2.add(face_mask_destroy, cur_model_neck_mask)
        gt_face_neck_mask = cv2.add(face_mask, cur_face_neck_mask)
        return cur_face_neck_mask, gt_face_neck_mask


    '''
    @:param 
    @:param  max_h  最大坏的高度
    @:param  face_neck_points  骨骼点
    @:param  b_left  破坏区域位于左边还是右边
    @:param  inner  构造凹形还是凸形破坏区域
    '''
    def draw_destroy_mask_v1(self, face_mask_destroyed, face_mask, max_h, face_neck_points, b_left=True, inner=True):
        # 算出破坏区域mask的上下范围
        ymax = face_neck_points[0][1] if b_left else face_neck_points[1][1]

        # n = np.random.randint(1, max_h // 2)
        n = max_h // np.random.randint(8, 16)
        h_arr = random.sample(list(range(ymax - max_h, ymax)), k=n)
        face_rows = face_mask[h_arr]
        face_ys = np.sort(np.array(h_arr, np.int32))

        face_xs = np.argmax(face_rows, axis=1) if b_left else self.size[0] - np.argmax(face_rows[::-1], axis=1)
        if inner:
            start_point = np.array([[face_xs[0] - 30, face_ys[0]]]) if b_left else np.array([[face_xs[0] + 30, face_ys[0]]])
            end_point = np.array([[face_xs[-1] - 30, face_ys[-1]]]) if b_left else np.array([[face_xs[-1] + 30, face_ys[-1]]])
            x_shift = np.random.randint(0, 5, face_xs.size)
            face_xs = face_xs + x_shift if b_left else face_xs - x_shift
        else:
            start_point = np.array([[face_xs[0] + 30, face_ys[0]]]) if b_left else np.array([[face_xs[-1] - 30, face_ys[-1]]])
            end_point = np.array([[face_xs[-1] + 30, face_ys[-1]]]) if b_left else np.array([[face_xs[-1] - 30, face_ys[-1]]])
            x_shift = np.random.randint(0, 3, face_xs.size)
            face_xs = face_xs - x_shift if b_left else face_xs + x_shift

        destroy_face_points = np.concatenate([face_xs[:, None], face_ys[:, None]], axis=1)

        destroy_face_points = np.concatenate([start_point, destroy_face_points, end_point], axis=0)

        # 可视化点位
        # mask_pt_vis = np.zeros(self.size, dtype=np.uint8)
        # for pt in destroy_face_points:
        #     cv2.drawMarker(mask_pt_vis, pt, 255, cv2.MARKER_CROSS, 2, 1)

        mask = np.zeros(self.size, dtype=np.uint8)
        cv2.fillPoly(mask, [destroy_face_points], 255)
        # ks = np.random.choice(list(range(3, 22, 2))) if inner else 3
        # mask = cv2.GaussianBlur(mask, (ks, ks), 0)

        face_mask_destroyed =cv2.subtract(face_mask_destroyed, mask) if inner else cv2.add(face_mask_destroyed, mask)
        return face_mask_destroyed

    def draw_destroy_mask_v2(self, face_mask_destroyed, face_mask, max_w, model_neck_points, b_left=True, inner=True):
        # 算出破坏区域mask的上下范围
        x_ref = model_neck_points[0][0] + 7 if b_left else model_neck_points[1][0] - 7

        n = max_w
        x_arr = random.sample(list(range(x_ref - max_w, x_ref)), k=n) if b_left else random.sample(list(range(x_ref, x_ref + max_w)), k=n)
        face_cols = face_mask[:, x_arr]
        face_xs = np.sort(np.array(x_arr, np.int32))

        face_ys = self.size[1] - np.argmax(face_cols[::-1], axis=0)
        if inner:
            start_point = np.array([[face_xs[0], face_ys[0] + 30]])
            end_point = np.array([[face_xs[-1], face_ys[-1] + 30]])
            y_shift = np.random.randint(0, 5, face_xs.size) if np.random.uniform() < 0.8 else np.random.randint(7, 13, face_xs.size)
            face_ys = face_ys - y_shift
        else:
            start_point = np.array([[face_xs[0], face_ys[0] - 30]])
            end_point = np.array([[face_xs[-1], face_ys[-1] - 30]])
            y_shift = np.random.randint(0, 5, face_xs.size)
            face_ys = face_ys + y_shift

        destroy_face_points = np.concatenate([face_xs[:, None], face_ys[:, None]], axis=1)

        destroy_face_points = np.concatenate([start_point, destroy_face_points, end_point], axis=0)

        # 可视化点位
        mask_pt_vis = np.zeros(self.size, dtype=np.uint8)
        for pt in destroy_face_points:
            cv2.drawMarker(mask_pt_vis, pt, 255, cv2.MARKER_CROSS, 2, 1)

        mask = np.zeros(self.size, dtype=np.uint8)
        cv2.fillPoly(mask, [destroy_face_points], 255)
        # ks = np.random.choice(list(range(3, 22, 2))) if inner else 3
        # mask = cv2.GaussianBlur(mask, (ks, ks), 0)

        face_mask_destroyed = cv2.subtract(face_mask_destroyed, mask) if inner else cv2.add(face_mask_destroyed, mask)
        return face_mask_destroyed


if __name__ == '__main__':
    obj_face_destroy = FaceDestroy()
    src_root = '/data_ssd/ay/切脸数据/train_v2'
    points_dir = os.path.join(src_root, 'points')
    face_mask_dir = os.path.join(src_root, 'face_mask_cnt')
    model_neck_mask = cv2.imread("/root/group-trainee/ay/version1/main/model/sucai/woman/model_fake_neck_mask.png", -1)

    for fn in os.listdir(face_mask_dir):
        pre, post = os.path.splitext(fn)
        if '.' not in pre and post != '':
            img_path = os.path.join(face_mask_dir, fn)
            points_path = os.path.join(points_dir, pre + '.npy')
            face_mask = cv2.imread(img_path)[:, :, 0]
            # face_mask = np.where(face_mask == 5, 0, face_mask).astype(np.uint8)
            points = np.load(points_path)

            face_mask_destroyed = obj_face_destroy.face_mask_destroy(face_mask, points, model_neck_mask)
            pass




