import os
import cv2
import torch
import numpy as np
import shutil
import random
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from lollipop import FacePoint

class Utils:
    def __init__(self):
        pass

    def is_imgfile(self, img_fn, suffix_list=['.png', '.jpg', '.tif', '.bmp', '.tiff', '.jpeg', '.npy']):
        pre, post = os.path.splitext(img_fn)
        return '.' not in pre and post.lower() in suffix_list

    def alpha_merge(self, foreground, background, alpha):
        alpha = cv2.merge([alpha, alpha, alpha]) * np.float64(1 / 255)
        foreground = foreground.astype(np.float64)

        img_out = foreground * alpha + background * (1 - alpha)
        return img_out.clip(0, 255).astype(np.uint8)

    def check_image_info(self, img_dir):
        for fn in os.listdir(img_dir):
            fp = os.path.join(img_dir, fn)
            img = cv2.imread(fp, -1)
            if img is not None:
                print('图像大小：{0}'.format(img.shape))

    def run_one(self, image, prompt='请描述这张图片', chinese=True, url='http://172.30.2.151:11070/llava'):
        import requests
        import base64
        scale = 1024. / max(image.shape)
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        png_data = cv2.imencode('.png', image)[1]
        base_64_data = base64.b64encode(png_data).decode('ascii')
        dict_sample = {
            'parameter': {'my_name': 'yX/Ceic10KKc9Y36VnSkOaRl2xZz9UcQrNJOgFY/FqQ=', },
            'recognize_img': [{
                'media_data': base_64_data,
                'prompt': prompt,
                'chinese': chinese
            }],
        }
        return requests.post(url=url, json=dict_sample).json()

    # 对文件夹中的图片进行操作
    def process_imgs_in_fold(self, dir, suffix=['.png', '.jpg']):
        for fn in os.listdir(dir):
            pre, post = os.path.splitext(fn)
            if '.' not in pre and post.lower() in suffix:
                fp = os.path.join(dir, fn)
                img = cv2.imread(fp, -1)[860:1372, 758:1270, 3]

                cv2.imwrite(fp, img)

    # 文件夹中图片按数字序号重命名，并复制到另一个文件夹
    def copy_rename_from_n(self, src_dir, trg_dir, n):
        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)
        for fn in os.listdir(src_dir):
            if self.is_imgfile(fn):
                src_fp = os.path.join(src_dir, fn)
                trg_fn = str(n) + '.png'
                trg_fp = os.path.join(trg_dir, trg_fn)
                shutil.copyfile(src_fp, trg_fp)
                n += 1

    def copy_org_img_from_res(self, org_dir, res_dir, trg_dir, i='vthird'):
        g = np.full([2000, 2000], 153, np.uint8)
        b = np.full([2000, 2000], 85, np.uint8)
        r = np.full([2000, 2000], 91, np.uint8)
        green = np.dstack([b, g, r])
        pre_cloth_dir = os.path.join(org_dir, 'pre_cloth')
        for fn in tqdm(os.listdir(res_dir)):
            if self.is_imgfile(fn):
                pre = os.path.splitext(fn)[0]
                hair_fn = pre[:-4]+ i + '.png'
                ref_fn = hair_fn[:-4] + '-ref.png'
                org_fp = os.path.join(org_dir, fn)
                '''mask cloth'''
                ref = cv2.imread(org_fp, -1)
                pre_fp = os.path.join(pre_cloth_dir, fn[:-8] + '.png')
                pre_cloth = cv2.imread(pre_fp, -1)[:, :, 3]
                img_out = self.alpha_merge(green, ref[:, :, :3], pre_cloth)
                save_fp = os.path.join(trg_dir, ref_fn)
                cv2.imwrite(save_fp, img_out)

                org_fp = os.path.join(org_dir, pre[:-4]+'.png')
                trg_fp = os.path.join(trg_dir, hair_fn)
                shutil.copyfile(org_fp, trg_fp)

    def copy_org_img_from_res_1(self, org_dir, res_dir, trg_dir, extra_postfix=''):
        org_fns = os.listdir(org_dir)
        res_fns = os.listdir(res_dir)

        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        for fn in tqdm(res_fns):
            pre, post = os.path.splitext(fn)
            res_fn = pre[:-4] + '.png'
            if self.is_imgfile(fn) and res_fn in org_fns:
                org_fp = os.path.join(org_dir, res_fn)
                trg_fp = os.path.join(trg_dir, res_fn)
                shutil.copyfile(org_fp, trg_fp)
                shutil.copyfile(os.path.join(org_dir, fn), os.path.join(trg_dir, fn))

    def copy_org_img_from_res_2(self, org_dir, res_dir, trg_dir, extra_postfix=''):
        org_fns = os.listdir(org_dir)
        res_fns = os.listdir(res_dir)

        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        for fn in tqdm(res_fns):
            pre, post = os.path.splitext(fn)
            res_fn = pre[:-4] + '.png'
            if self.is_imgfile(fn) and res_fn in org_fns:
                org_fp = os.path.join(org_dir, res_fn)
                trg_fp = os.path.join(trg_dir, res_fn)
                shutil.copyfile(org_fp, trg_fp)
                shutil.copyfile(os.path.join(org_dir, fn), os.path.join(trg_dir, fn))

                res_fn = pre[:-4] + '.jpg'
                trg_fn = pre[:-4] + '-z.jpg'
                org_fp = os.path.join(os.path.dirname(org_dir), res_fn)
                trg_fp = os.path.join(trg_dir, trg_fn)
                shutil.copyfile(org_fp, trg_fp)

    # 从结果图拿原图
    def copy_org_img_from_res_3(self, org_dir, res_dir, trg_dir, extra_postfix=''):
        res_fns = os.listdir(res_dir)

        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        for fn in tqdm(res_fns):
            if self.is_imgfile(fn) and '-ref' not in fn:
                pre, post = os.path.splitext(fn)
                org_fn = pre.replace(extra_postfix, '') + '.png'
                org_fp = os.path.join(org_dir, org_fn)
                trg_fn = pre + '.png'
                trg_fp = os.path.join(trg_dir, trg_fn)
                if os.path.exists(org_fp):
                    shutil.copyfile(org_fp, trg_fp)


    def copy_org_img_from_res_4(self, org_dir, trg_dir, extra_postfix='', sub_dirs=['back_cloth', 'coor', 'person_mask', 'pre_cloth', 'ref', 'hair']):
        res_fns = os.listdir(trg_dir)

        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)
        for sub_dir in sub_dirs:
            fp = os.path.join(trg_dir, sub_dir)
            if not os.path.exists(fp):
                os.makedirs(fp)

        for fn in tqdm(res_fns):
            if self.is_imgfile(fn):
                pre, post = os.path.splitext(fn)
                for sub_dir_ in sub_dirs:
                    if sub_dir_=='coor':
                        org_sub_fn = os.path.join(org_dir, sub_dir_, pre.replace(extra_postfix, '') + '.npy')
                        trg_sub_fn = os.path.join(trg_dir, sub_dir_, pre + '.npy')
                    elif sub_dir_=='hair':
                        org_sub_fn = os.path.join(org_dir, pre.replace(extra_postfix, '') + '.png')
                        trg_sub_fn = os.path.join(trg_dir, sub_dir_, pre + '.png')
                    else:
                        org_sub_fn = os.path.join(org_dir, sub_dir_, pre.replace(extra_postfix, '') + '.png')
                        trg_sub_fn = os.path.join(trg_dir, sub_dir_, pre + '.png')

                    if os.path.exists(org_sub_fn):
                        shutil.copyfile(org_sub_fn, trg_sub_fn)

    def remove_first_char(self, root, sub_dirs=['pre_cloth', 'back_cloth', 'coor', 'org', 'org_cloth_mask', 'ref', '']):
        dirs = []
        for sub_dir in sub_dirs:
            dirs.append(os.path.join(root, sub_dir))

        for dir_ in dirs:
            for fn in tqdm(os.listdir(dir_)):
                if self.is_imgfile(fn) and os.path.splitext(fn)[1] != '':
                    fp = os.path.join(dir_, fn)
                    new_fp = os.path.join(dir_, fn[1:])
                    os.rename(fp, new_fp)


    def extract_hairs(self, hair_dir):
        sub_hair_dir = os.path.join(hair_dir, 'ref')
        if not os.path.exists(sub_hair_dir):
            os.makedirs(sub_hair_dir)
        sub_hair_dir = '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/定稿/ref/'

        for dirpath, _, fns in os.walk(hair_dir):
            for fn in fns:
                if self.is_imgfile(fn) and '-ref' in fn:
                    src_fp = os.path.join(dirpath, fn)
                    trg_fp = os.path.join(sub_hair_dir, fn)
                    shutil.copyfile(src_fp, trg_fp)


    # 不同mask融合得到新的结果
    def merge_mask(self, mask_lst, model_mask_path='/root/group-trainee/ay/version1/main/model/model_neck_smooth/org.png'):
        model_mask = cv2.imread(model_mask_path, -1)[860:1372, 758:1270, 3]
        n = len(mask_lst)
        selected_mask = random.choices(mask_lst, k=2)
        if random.choice([True, False])[0]:
            mask = cv2.absdiff(selected_mask[0], selected_mask[1])
        else:
            mask = (selected_mask[0] + selected_mask[1]).clip(0, 255).astype(np.uint8)
        mask = np.where(model_mask > 0, mask, 0)

        return mask

    def calc_img_nums(self, dir):
        n = 0
        for fn in os.listdir(dir):
            if self.is_imgfile(fn):
                n += 1
        print('image nums in this directory: {0}'.format(n))

    def resize_imgs(self, dir1, dir2, interpolate=cv2.INTER_CUBIC, trg_size=(512, 512), fx=1.0, fy=1.0):
        for fn in os.listdir(dir1):
            fp = os.path.join(dir1, fn)
            img = cv2.imread(fp, -1)
            img_resized = np.copy(img)
            if fx==1.0 and fy==1.0:
                img_resized = cv2.resize(img, trg_size, interpolation=interpolate)
            else:
                img_resized = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interpolate)

            save_fp = os.path.join(dir2, fn)
            cv2.imwrite(save_fp, img_resized)

    def get_input_of_badcase(self, input_dir, src_dir, trg_dir):
        for fn in os.listdir(src_dir):
            if self.is_imgfile(fn):
                input_fp = os.path.join(input_dir, fn)
                trg_fp = os.path.join(trg_dir, fn)
                shutil.copyfile(input_fp, trg_fp)

    def merge_dirs(self, dirs, trg_dir, sub_dirs=None, i=0, img_post='.png'):
        if sub_dirs == None:
            if not os.path.exists(trg_dir):
                os.makedirs(trg_dir)
            for dir in dirs:
                for fn in tqdm(os.listdir(dir), desc='Processing -> ' + dir):
                    if self.is_imgfile(fn):
                        src_fp = os.path.join(dir, fn)
                        # trg_fn = str(i) + img_post
                        trg_fp = os.path.join(trg_dir, fn)
                        shutil.copyfile(src_fp, trg_fp)
                        # i += 1
        else:
            for sub_dir in sub_dirs:
                dirs_merged = [os.path.join(dir_, sub_dir) for dir_ in dirs]
                trg_dir_merged = os.path.join(trg_dir, sub_dir)
                self.merge_dirs(dirs_merged, trg_dir_merged)

        return

    def curve_fit_third(self):
        # 假设控制点为 (x1, y1), (x2, y2), ..., (x7, y7)
        # 并且我们选择第三个控制点作为拐点
        control_points = np.array([
            [-3, 8.9],
            [-2, 8.2],
            [-1, 1.05],
            # 添加其他控制点
            [0, 0],  # 拐点
            [1, -0.98],
            [2, -7.86],
            [3, -9.15],

        ])

        # 提取x和y坐标
        X = control_points[:, 0]
        Y = control_points[:, 1]

        # 构造 Vandermonde 矩阵
        A = np.vstack([x ** 3 for x in X])
        A = np.column_stack((A, np.vstack([x ** 2 for x in X]), np.vstack([x for x in X]), np.ones(len(X))[:, None]))

        # 解线性方程组来找到系数 a, b, c, d
        coefficients = np.linalg.lstsq(A, Y, rcond=None)[0]

        # 打印系数
        a, b, c, d = coefficients
        print(f"三次多项式的系数为: a={a}, b={b}, c={c}, d={d}")

        # 现在我们有了系数，可以定义三次多项式函数
        def cubic_polynomial(x):
            return a * x ** 3 + b * x ** 2 + c * x + d

        # 计算拐点的x值，使得二阶导数为0
        # 二阶导数为 6ax + 2b
        # 6ax + 2b = 0
        # x = -b / (6a)
        拐点_x = -b / (6 * a)

        # 检查拐点是否在我们选择的控制点上
        拐点_y = cubic_polynomial(拐点_x)
        print(f"拐点: ({拐点_x}, {拐点_y})")

        # 使用多项式函数
        # 例如，在一系列x值上评估多项式
        x_values = np.linspace(min(X), max(X), 100)
        y_values = cubic_polynomial(x_values)

        # 如果需要，可以绘制曲线和控制点
        import matplotlib.pyplot as plt

        plt.plot(X, Y, 'o', label='Control Points')
        plt.plot(x_values, y_values, label='Fitted Cubic Polynomial')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

        return

    def remove_extra_files(self, src_dir, ref_dir):
        for fn in tqdm(os.listdir(src_dir)):
            if self.is_imgfile(fn):
                fp = os.path.join(ref_dir, fn)
                if not os.path.exists(fp):
                    os.remove(os.path.join(src_dir, fn))

    def img_tensor_transformer(self, img):
        import torchvision.transforms.functional as TF

        factor = np.random.uniform(1.05, 1.2)

        img_ = TF.adjust_saturation(img, factor)

        # img_ = transforms.ToPILImage()(img_tensor)

        a = np.hstack([img, img_])
        plt.imshow(a)
        plt.show()


    def approxPlolyDP(self):
        # 创建一个示例图像和轮廓
        image = np.zeros((500, 500), dtype=np.uint8)
        cv2.circle(image, (250, 250), 200, 255, -1)  # 画一个圆

        # 检测轮廓
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 对最大的轮廓进行多边形近似
        epsilon = 0.0002 * cv2.arcLength(contours[0], True)
        approxCurve = cv2.approxPolyDP(contours[0], epsilon, True)

        # 在图像上绘制原始轮廓和近似轮廓
        cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        cv2.drawContours(image, [approxCurve], -1, (255, 0, 0), 5)

        # 显示图像
        cv2.imshow('Contours and Approximated Polygon', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def refine_mask(self, src_dir, fn_dir, trg_dir):
        for fn in os.listdir(fn_dir):
            pre, _ = os.path.splitext(fn)
            if '.' not in pre and _ != '':
                face_skin_mask_dir = os.path.join(src_dir, 'face_skin_mask')
                other_skin_mask_dir = os.path.join(src_dir, 'other_skin_mask')
                real_neck_mask_dir = os.path.join(src_dir, 'real_neck_mask')

                face_skin_mask = cv2.imread(os.path.join(face_skin_mask_dir, fn), -1)
                other_skin_mask = cv2.imread(os.path.join(other_skin_mask_dir, fn), -1)
                real_neck_mask = cv2.imread(os.path.join(real_neck_mask_dir, fn), -1)

                face_skin_mask = cv2.subtract(face_skin_mask, other_skin_mask)
                real_neck_mask = cv2.add(real_neck_mask, other_skin_mask)

                face_neck_mask = cv2.add(face_skin_mask, real_neck_mask)
                face_neck_mask[real_neck_mask > 0] = 255
                real_neck_mask = cv2.subtract(face_neck_mask, face_skin_mask)
                other_skin_mask_ = np.where(other_skin_mask > 10, 255, 0).astype(np.uint8)
                real_neck_mask = cv2.subtract(real_neck_mask, other_skin_mask_)



                face_skin_mask_dir = os.path.join(trg_dir, 'face_skin_mask')
                other_skin_mask_dir = os.path.join(trg_dir, 'other_skin_mask')
                real_neck_mask_dir = os.path.join(trg_dir, 'real_neck_mask')

                face_skin_mask = cv2.imwrite(os.path.join(face_skin_mask_dir, fn), face_skin_mask)
                other_skin_mask = cv2.imwrite(os.path.join(other_skin_mask_dir, fn), other_skin_mask)
                neck_mask = cv2.imwrite(os.path.join(real_neck_mask_dir, fn), real_neck_mask)


    def remove_replications(self, fn_dir, trg_dir):
        for _, _, fns in os.walk(fn_dir):
            for fn in fns:
                if self.is_imgfile(fn):
                    trg_fp = os.path.join(trg_dir, fn)
                    if os.path.exists(trg_fp):
                        os.remove(trg_fp)


    def val_hair_data(self, root, save_dir, post_fix='vthird'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for fn in tqdm(os.listdir(root)):
            if self.is_imgfile(fn) and '-ref' not in fn: #and fn[-10:-4]==post_fix
                hair = cv2.imread(os.path.join(root, fn), -1)
                ref_fn = fn[:-4] + '-ref.png'
                ref = cv2.imread(os.path.join(root, ref_fn), -1)

                res = self.alpha_merge(hair[:, :, :3], ref[:, :, :3], hair[:, :, 3])
                res_path = os.path.join(save_dir, fn)
                cv2.imwrite(res_path, res)


    def val_hair_data_final(self, hair_dir, ref_dir, save_dir, post_fix='vthird'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        red = np.full((2000, 2000), 255, np.uint8)
        black = np.zeros_like(red, np.uint8)
        red = np.dstack([black, black, red])

        for fn in tqdm(os.listdir(hair_dir)):
            if self.is_imgfile(fn): # and fn[-10:-4]==post_fix
                hair = cv2.imread(os.path.join(hair_dir, fn), -1)
                ref = cv2.imread(os.path.join(ref_dir, fn), -1)
                if ref is None:
                    continue

                res = self.alpha_merge(hair[:, :, :3], red, hair[:, :, 3])
                res = self.alpha_merge(ref[:, :, None], res, ref)
                res_path = os.path.join(save_dir, fn)
                cv2.imwrite(res_path, res)

    def remove_val_data(self, hair_dir, ref_dir, save_dir, post_fix='vthird'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        red = np.full((2000, 2000), 255, np.uint8)
        black = np.zeros_like(red, np.uint8)
        red = np.dstack([black, black, red])

        for fn in tqdm(os.listdir(hair_dir)):
            if self.is_imgfile(fn): # and fn[-10:-4]==post_fix
                hair = cv2.imread(os.path.join(hair_dir, fn), -1)
                ref = cv2.imread(os.path.join(ref_dir, fn), -1)
                if ref is None:
                    continue

                res = self.alpha_merge(hair[:, :, :3], red, hair[:, :, 3])
                res = self.alpha_merge(ref[:, :, None], res, ref)
                res_path = os.path.join(save_dir, fn)
                cv2.imwrite(res_path, res)

    def divide_into_n_folder(self, root, sub_dirs=['0', '1', '2']):
        save_dirs = [os.path.join(root, sub_dir) for sub_dir in sub_dirs]
        for save_dir in save_dirs:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for fn in os.listdir(root):
            if self.is_imgfile(fn) and '_1' not in fn and '_2' not in fn:
                _, post = os.path.splitext(fn)
                shutil.copyfile(os.path.join(root, fn), os.path.join(save_dirs[0], fn))
                fn_1 = fn[:-4] + '_1' + post
                shutil.copyfile(os.path.join(root, fn_1), os.path.join(save_dirs[1], fn))
                fn_2 = fn[:-4] + '_2' + post
                shutil.copyfile(os.path.join(root, fn_2), os.path.join(save_dirs[2], fn))

    def create_gif_from_folders(self, root, duration=500):
        folder1, folder2, folder3, output_dir = [os.path.join(root, sub_dir) for sub_dir in ['0', '1', '2', '3']]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 获取文件夹中的所有图片文件
        files1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
        files2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
        files3 = sorted([os.path.join(folder3, f) for f in os.listdir(folder3) if f.endswith(('png', 'jpg', 'jpeg'))])

        # 确保每个文件夹中的图片数量相同
        if not (len(files1) == len(files2) == len(files3)):
            raise ValueError("Each folder must contain the same number of images")


        # 逐一打开图片，并添加到images列表中
        for f1, f2, f3 in tqdm(zip(files1, files2, files3)):
            images = []
            fn = os.path.basename(f1)
            gif_path = os.path.join(output_dir, os.path.splitext(fn)[0]+'.gif')

            img1 = Image.open(f1)
            img2 = Image.open(f2)
            img3 = Image.open(f3)

            # 将图像添加到images列表
            images.append(img1)
            images.append(img2)
            images.append(img3)

            # 保存为GIF
            imageio.mimsave(gif_path, images, format='GIF', duration=duration, loop=0)


    def create_gif(self, root, sub_dirs=['0', '1', '2'], duration=500):
        save_dirs = [os.path.join(root, sub_dir) for sub_dir in sub_dirs]
        for save_dir in save_dirs:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for fn in os.listdir(root):
            if self.is_imgfile(fn) and '_3' not in fn and '_4' not in fn:
                _, post = os.path.splitext(fn)
                shutil.copyfile(os.path.join(root, fn), os.path.join(save_dirs[0], fn))
                fn_1 = fn[:-4] + '_3' + post
                shutil.copyfile(os.path.join(root, fn_1), os.path.join(save_dirs[1], fn))
                fn_2 = fn[:-4] + '_4' + post
                shutil.copyfile(os.path.join(root, fn_2), os.path.join(save_dirs[2], fn))

        folder1, folder2, folder3, output_dir = [os.path.join(root, sub_dir) for sub_dir in ['0', '1', '2', '3']]

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # 获取文件夹中的所有图片文件
        files1 = sorted([os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith(('png', 'jpg', 'jpeg'))])
        files2 = sorted([os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith(('png', 'jpg', 'jpeg'))])
        files3 = sorted([os.path.join(folder3, f) for f in os.listdir(folder3) if f.endswith(('png', 'jpg', 'jpeg'))])

        # 确保每个文件夹中的图片数量相同
        if not (len(files1) == len(files2) == len(files3)):
            raise ValueError("Each folder must contain the same number of images")

        # 逐一打开图片，并添加到images列表中
        for f1, f2, f3 in tqdm(zip(files1, files2, files3)):
            images = []
            fn = os.path.basename(f1)
            gif_path = os.path.join(output_dir, os.path.splitext(fn)[0] + '.gif')

            img1 = Image.open(f1)
            img2 = Image.open(f2)
            img3 = Image.open(f3)

            # 将图像添加到images列表
            images.append(img1)
            images.append(img2)
            images.append(img3)

            # 保存为GIF
            imageio.mimsave(gif_path, images, format='GIF', duration=duration, loop=0)


    def merge_n_sub_dirs(self, roots, sub_dirs, trg_dir):
        for i, sub_dir in enumerate(sub_dirs):
            for root in roots:
                if os.path.exists(os.path.join(root, sub_dir)):
                    for fn in tqdm(os.listdir(os.path.join(root, sub_dir))):
                        if self.is_imgfile(fn):
                            src_fp = os.path.join(root, sub_dir, fn)
                            trg_fp = os.path.join(trg_dir, sub_dir, fn)
                            shutil.copyfile(src_fp, trg_fp)


    def save_face_points(self, model_path, save_path, trg_size=(512, 512)):
        img = cv2.imread(model_path)
        img = cv2.resize(img, trg_size, None, interpolation=cv2.INTER_CUBIC)
        face_model = FacePoint()
        face_points = face_model(img)[0]

        np.save(save_path, face_points)


    def delete_extra_images(self, src_dir, extra_dir):
        for fn in tqdm(os.listdir(extra_dir)):
            if self.is_imgfile(fn):
                for post in ['.jpg', '.jpeg', '.png']:
                    pre, _ = os.path.splitext(fn)
                    fn = pre[:-4] + post
                    src_fp = os.path.join(src_dir, fn)
                    if not os.path.exists(src_fp):
                        continue
                    os.remove(src_fp)


    def to_same_name(self, dir1, dir2):
        fns1, fns2 = [], []
        for fn in os.listdir(dir1):
            if self.is_imgfile(fn):
                fns1.append(fn)
        for fn in os.listdir(dir2):
            if self.is_imgfile(fn):
                fns2.append(fn)

        for fn1, fn2 in zip(fns1, fns2):
            pre = os.path.splitext(fn1)[0]
            fn = pre + '.png'
            fp1, fp2 = os.path.join(dir1, fn1), os.path.join(dir2, fn2)
            img1, img2 = cv2.imread(fp1, -1), cv2.imread(fp2, -1)
            cv2.imwrite(os.path.join(os.path.dirname(dir1), '换发型', 'p1', fn), img1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            cv2.imwrite(os.path.join(os.path.dirname(dir2), '换发型', 'p2', fn), img2, [cv2.IMWRITE_PNG_COMPRESSION, 0])


    def get_final_hair_data(self, root, trg_dir):
        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        trg_dir_1 = os.path.join(trg_dir, 'hair')
        trg_dir_2 = os.path.join(trg_dir, 'hair_tar')
        for sub_dir in [trg_dir_1, trg_dir_2]:
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        for dirpath, dirname, fns in os.walk(root):
            for fn in fns:
                pre = os.path.splitext(fn)[0]
                if self.is_imgfile(fn) and '-ref' not in fn:
                    if pre[-2:] == '_a':
                        shutil.copyfile(os.path.join(dirpath, fn), os.path.join(trg_dir_2, fn))
                    else:
                        shutil.copyfile(os.path.join(dirpath, fn), os.path.join(trg_dir_1, fn))

    def delete_extra_files(self, trg_dir, delete_dir):
        for fn in os.listdir(delete_dir):
            if self.is_imgfile(fn):
                fp = os.path.join(trg_dir, fn)
                if os.path.exists(fp):
                    os.remove(fp)


    def find_missing_files(self, base_folder, check_dir):
        """
        找出其他文件夹中缺少的基准文件夹中的文件名

        :param base_folder: 基准文件夹的路径
        :param other_folders: 其他文件夹的路径
        :return: 一个字典，键是其他文件夹的路径，值是缺少的文件名集合
        """
        # 获取基准文件夹中的所有文件名
        i = 0
        check_dir_pres = []
        for fn in os.listdir(check_dir):
            pre, post = os.path.splitext(fn)
            if '.' not in pre and post != '':
                check_dir_pres.append(pre)


        for fn in os.listdir(base_folder):
            pre, post = os.path.splitext(fn)
            if '.' not in pre and post != '' and pre not in check_dir_pres and len(pre) > 10:

                # trg_fn = pre[:-2] + '.jpg'
                # src_fp = os.path.join('/root/group-inspect-data/证件照换装/原始数据/20240618-DP2/', trg_fn)
                # trg_fp = os.path.join('/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/原图/1/', trg_fn)
                # if os.path.exists(src_fp):
                #     shutil.copyfile(src_fp, trg_fp)

                print(fn)
                i += 1

        print (f'一共 {i} 个文件')



    def rename_fn(self, src_dir, trg_sub_dir_name="renamed"):
        trg_dir = os.path.join(src_dir, trg_sub_dir_name)
        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)
        for fn in tqdm(os.listdir(src_dir)):
            if self.is_imgfile(fn):
                src_fp = os.path.join(src_dir, fn)
                trg_fp = os.path.join(trg_dir, fn)
                shutil.copyfile(src_fp, trg_fp)
                src_fp = os.path.join(trg_dir, fn)
                trg_fn = os.path.splitext(fn)[0][:-2] + '.png'
                trg_fp = os.path.join(trg_dir, trg_fn)
                os.rename(src_fp, trg_fp)


    def get_finalized_hair_data(self, src_dir, trg_dir, sub_dirs=['hair', 'hair_tar', 'ref']):
        for sub_dir in sub_dirs:
            dir_ = os.path.join(trg_dir, sub_dir)
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        for dir_name, _, fns in os.walk(src_dir):
            for fn in tqdm(fns):
                pre, post = os.path.splitext(fn)
                if '.' not in pre and post != '':
                    if pre[-4:] == '-ref':
                        trg_fn = pre[:-4]
                        hair_fn = trg_fn + '.png'
                        hair_tar_fn = trg_fn + '_a.png'
                        ref_fn = trg_fn + '-ref.png'
                        trg_fn = trg_fn + '.png'

                        shutil.copyfile(os.path.join(dir_name, hair_fn), os.path.join(trg_dir, 'hair', trg_fn))
                        shutil.copyfile(os.path.join(dir_name, hair_tar_fn), os.path.join(trg_dir, 'hair_tar', trg_fn))
                        shutil.copyfile(os.path.join(dir_name, ref_fn), os.path.join(trg_dir, 'ref', trg_fn))


    def move_files(self, root, src_sub_dirs = ['头发在前', '头发在前_b_hair_refine']):
        trg_dir = os.path.join(root, 'res', 'all')
        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        for sub_dir in src_sub_dirs:
            src_dir = os.path.join(root, sub_dir)
            for fn in os.listdir(src_dir):
                if self.is_imgfile(fn):
                    if '_3' in fn or '_4' in fn:
                        src_fn = os.path.join(src_dir, fn)
                        trg_fn = os.path.join(trg_dir, fn)
                        shutil.copyfile(src_fn, trg_fn)


    def copy_from_res(self, res_dir, src_dir, trg_dir):
        for fn in os.listdir(res_dir):
            if self.is_imgfile(fn) and '_3' not in fn:
                pre, post = os.path.splitext(fn)
                fn = pre + '_4' + post
                src_fn = os.path.join(src_dir, fn)
                trg_fn = os.path.join(trg_dir, fn)
                shutil.copyfile(src_fn, trg_fn)

    def copy_files_from_resdir(self, res_dir, src_dir, sub_dirs=['back_cloth', 'pre_cloth', 'coor', 'hair', 'org', 'org_cloth_mask', 'person_mask']):
        trg_root = os.path.join(res_dir, 'res')
        trg_dirs = [os.path.join(res_dir, sub_dir) for sub_dir in sub_dirs]
        for trg_dir in trg_dirs:
            if not os.path.exists(trg_dir):
                os.makedirs(trg_dir)

        for fn in tqdm(os.listdir(res_dir)):
            pre, post = os.path.splitext(fn)
            if '.' not in pre and post != '':

                for i in range(len(sub_dirs)):
                    if sub_dirs[i] == 'coor':
                        trg_fn = pre[:-4] + '.npy'
                    elif sub_dirs[i] == 'ref':
                        trg_fn = pre[:-4] + '-ref.png'
                    else :
                        trg_fn = pre[:-4] + '.png'

                    src_fp = os.path.join(src_dir, sub_dirs[i], trg_fn)
                    trg_fp = os.path.join(trg_dirs[i], trg_fn)

                    if os.path.exists(src_fp):
                        shutil.copyfile(src_fp, trg_fp)
                    else:
                        print(f"该文件不存在：{src_fp}")

    def add_masks_from_diff_dirs(self, mask_dirs, trg_dir):
        if not os.path.exists(trg_dir):
            os.makedirs(trg_dir)

        for fn in os.listdir(mask_dirs[0]):
            if self.is_imgfile(fn):
                mask0 = os.path.join(mask_dirs[0], fn)
                mask0 = cv2.imread(mask0, -1)
                mask_out = mask0
                for i in range(1, len(mask_dirs)):
                    mask_i = cv2.imread(os.path.join(mask_dirs[i], fn), -1)
                    mask_out = cv2.add(mask_out, mask_i)

            save_fn = os.path.join(trg_dir, fn)
            cv2.imwrite(save_fn, mask_out)



if __name__ == '__main__':
    util = Utils()
    img_path = '/data_ssd/ay/general/imgs/dog.png'

    # util.check_image_info('/root/group-trainee/ay/version1/main/model/model_neck_smooth/')

    # result = util.run_one(cv2.imread(img_path))
    # for k, v in result.items():
    #     print(k, v)

    # util.process_imgs_in_fold('/root/group-trainee/ay/version1/dataset/a_online512/test_randomask/')

    # util.copy_rename_from_n('/root/group-trainee/ay/hair/0808脸部碎发制作/',
    #                    '/root/group-trainee/ay/hair/samples2/', 13)

    # util.resize_imgs('/root/group-trainee/ay/hair/hair4000/', '/root/group-trainee/ay/hair/hair512/', trg_size=(512, 512))

    # mask_lst = []
    # mask_dir = '/root/group-trainee/ay/version1/dataset/a_online512/test_randomask/'
    # trg_dir = '/root/group-trainee/ay/version1/dataset/a_online512/test_randomask/fake_neck_mask/'
    # for fn in mask_dir:
    #     fp = os.path.join(mask_dir, fn)
    #     mask = cv2.imread(fp, -1)
    #     mask_lst.append(mask)

    # util.calc_img_nums('/root/group-inspect-data/证件照换装/原始数据/2完美图/未分类480p/')

    # util.get_input_of_badcase('/root/group-trainee/ay/version1/dataset/test_normal/880张证件照原片样片/1/',
    #                           '/root/group-inspect-data/证件照换装/中间结果/2024-05-07/880张证件照原片样片/1/有问题/13比例/',
    #                           '/root/group-trainee/ay/version1/dataset/test_normal/hard_case/head_shoulder_width/imgs/')

    # util.merge_dirs(['/root/group-trainee/ay/version1/dataset/a_online512/train/all2/stage_2/',
    #                  '/root/group-trainee/ay/version1/dataset/a_online512/train/extra00/stage_2/'],
    #                 '/root/group-trainee/ay/version1/dataset/a_online512/train/complete/stage_2',
    #                 sub_dirs=['src_img', 'gt_img', 'fake_neck_mask'])

    # util.merge_dirs(['/root/group-trainee/ay/version1/dataset/a_online512/train/all_no_shadow_neck_wo_otherskin',
    #                  '/root/group-trainee/ay/version1/dataset/a_online512/train/extra_no_shadow_neck_wo_otherskin/'],
    #                 '/root/group-trainee/ay/version1/dataset/a_online512/train/complete_no_shadow_neck_wo_otherskin',
    #                 sub_dirs=['GT', 'real_neck_mask', 'face_skin_mask', 'other_skin_mask'])

    # util.merge_dirs(['/root/group-trainee/ay/version1/dataset/dataset/extra',
    #                  '/root/group-trainee/ay/version1/dataset/dataset/both'],
    #                 '/root/group-trainee/ay/version1/dataset/dataset/complete/')


    # util.curve_fit_third()
    #
    # img = Image.open('/root/group-trainee/ay/tmp/results/1.png')
    # for _ in range(100):
    #     util.img_tensor_transformer(img)


    # util.approxPlolyDP()


    # util.copy_org_img_from_res('/root/group-trainee/ay/version1/dataset/test_normal/hair_all/',
    #                            '/root/group-trainee/ay/version1/dataset/a_hair/train/GT/',
    #                            '/root/group-trainee/ay/version1/dataset/test_normal/hair')

    # util.copy_org_img_from_res_1('/root/group-inspect-data/证件照换装/中间结果/2024-08-07/补发/',
    #                            '/root/group-inspect-data/证件照换装/中间结果/2024-07-31/hair/shoulder/0',
    #                            '/root/group-inspect-data/证件照换装/中间结果/2024-07-31/hair/shoulder/2',
    #                              extra_postfix='_2')


    # util.copy_org_img_from_res_2('/root/group-inspect-data/证件照换装/原始数据/4原图/240809_短发原图268p/结果图',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_短发原图268p/结果图/ref/p/',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_短发原图268p/结果图/ref/p/140p',
    #                            extra_postfix='')
    # util.val_hair_data('/root/group-inspect-data/证件照换装/原始数据/4原图/240809_短发原图268p/结果图/ref/p/140p',
    #                    '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_短发原图268p/结果图/ref/p/140p/res', post_fix='')

    # util.copy_org_img_from_res_2('/root/group-inspect-data/证件照换装/原始数据/4原图/240812_双马尾原图150p/结果图',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240812_双马尾原图150p/结果图/ref/y/',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240812_双马尾原图150p/结果图/ref/y/13p',
    #                            extra_postfix='')
    # util.val_hair_data('/root/group-inspect-data/证件照换装/原始数据/4原图/240812_双马尾原图150p/结果图/ref/y/13p',
    #                    '/root/group-inspect-data/证件照换装/原始数据/4原图/240812_双马尾原图150p/结果图/ref/y/13p/res', post_fix='')

    # util.copy_org_img_from_res_3('/root/group-inspect-data/证件照换装/中间结果/hair_wo_completion/hair_org_data/selected/DP_618_2/',
    #                              '/root/group-inspect-data/证件照换装/原始数据/头发送修4.0_正式方案/55p/',
    #                              '/root/group-inspect-data/证件照换装/原始数据/头发送修4.0_正式方案/55p/原图/', extra_postfix='vfifth')

    # util.copy_org_img_from_res_3('/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/org',
    #                              '/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/need/染发88p',
    #                              '/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/need/染发88p/原图', extra_postfix='')

    # util.remove_first_char('/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/')

    # util.copy_org_img_from_res_4('/root/group-inspect-data/证件照换装/中间结果/头发送修/data_ssd_18/DP_618_1_remake_nohair_v1/',
    #                              '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/49p/原图/'
    #                              , extra_postfix='vfourth')

    # util.extract_hairs('/root/group-inspect-data/证件照换装/原始数据/头发送修4.0_正式方案/短发140p/')

    # util.copy_org_img_from_res_1('/root/group-inspect-data/证件照换装/原始数据/4原图/240809_双马尾原图125p/结果图',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_双马尾原图125p/结果图/ref/n',
    #                            '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_双马尾原图125p/结果图/ref/n/双马尾要补32p',
    #                            extra_postfix='')
    # util.val_hair_data('/root/group-inspect-data/证件照换装/原始数据/4原图/240809_双马尾原图125p/结果图/ref/n/双马尾要补32p',
    #                    '/root/group-inspect-data/证件照换装/原始数据/4原图/240809_双马尾原图125p/结果图/ref/n/双马尾要补32p/res', post_fix='')

    # util.get_final_hair_data('/root/group-inspect-data/证件照换装/原始数据/头发送修4.0_正式方案/',
    #                          '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/定稿')
    # util.delete_extra_files('/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/定稿',
    #                         '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/试验')
    # util.extract_hairs('/root/group-inspect-data/证件照换装/原始数据/3已定稿/补发修图/')

    # util.refine_mask('/root/group-trainee/ay/version1/dataset/a_online512/train/complete_perfect',
    #                  '/root/group-trainee/ay/version1/dataset/a_online512/train/extra_no_shadow_neck_wo_otherskin/GT/',
    #                  '/root/group-trainee/ay/version1/dataset/a_online512/train/extra_no_shadow_neck_wo_otherskin/')

    # util.remove_extra_files('/root/group-trainee/ay/version1/dataset/a_online512/train/all_no_shadow_neck_wo_otherskin/GT/',
    #                         '/root/group-trainee/ay/version1/dataset/a_online512/train/all_no_shadow_neck_wo_otherskin_v1/GT/')


    # util.copy_org_img_from_res_1('/root/group-inspect-data/证件照换装/原始数据/4原图/0802_短发原图67p/',
    #                            '/data_ssd/ay/hair_to_ps/0802_短发原图67p/res/ref/need/',
    #                            '/data_ssd/ay/hair_to_ps/0802_短发原图67p/res/org',
    #                            extra_postfix='-ref')

    # util.copy_org_img_from_res('/root/group-inspect-data/证件照换装/中间结果/头发送修/DP_618_1/ref/no_hair/送修2.0/',
    #                            '/root/group-inspect-data/证件照换装/中间结果/头发送修/DP_618_1/ref/no_hair/大区域/',
    #                            '/root/group-inspect-data/证件照换装/中间结果/头发送修/DP_618_1/ref/no_hair/送修2.0/大区域/',
    #                            post_fix='-ref')

    # util.copy_org_img_from_res('/data_ssd/ay/hair_to_ps/DP_618_1_remake_nohair_v1/',
    #                            '/data_ssd/ay/hair_to_ps/DP_618_1_remake_nohair_v1/ref/remain_300p/opacity/',
    #                            '/data_ssd/ay/hair_to_ps/DP_618_1_remake_nohair_v1/ref/remain_300p/opacity/50p',
    #                            i='vfourth')

    # util.copy_org_img_from_res('/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图',
    #                            '/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/no_comp',
    #                            '/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/no_comp/染发不补56p',
    #                            i='')
    # util.val_hair_data('/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/no_comp/染发不补56p',
    #                    '/data_ssd/ay/ID_DATA/肩宽点/imgs/结果图/ref/no_comp/染发不补56p/res', post_fix='')


    # util.delete_extra_images('/data_ssd/ay/hair_to_ps/0802_短发原图67p/', '/data_ssd/ay/hair_to_ps/0802_短发原图67p/res/ref/need/')
    # util.to_same_name('/data_ssd/ay/hair_to_ps/短发送修(换发型)_mark/p1', '/data_ssd/ay/hair_to_ps/短发送修(换发型)_mark/p2')

    # util.merge_n_sub_dirs(['/root/group-trainee/ay/version1/dataset/a_hair/trail/', '/root/group-trainee/ay/version1/dataset/a_hair/add_v1/'],
    #                    # ['hair', 'hair_tar', 'pre_cloth', 'back_cloth', 'coor', 'ref'],
    #                       ['coor'],
    #                    '/root/group-trainee/ay/version1/dataset/a_hair/normal_v1/')


    # util.divide_into_n_folder('/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-01/240801_短发测试图62p/', sub_dirs=['0', '1', '2'])

    # util.create_gif_from_folders('/root/group-inspect-data/证件照换装/中间结果/2024-07-31/hair/shoulder', duration=500)

    # util.remove_replications('/root/group-inspect-data/证件照换装/测试数据/1500女证件照高清/', '/root/group-inspect-data/证件照换装/中间结果/hair_wo_completion/hair_org_data/20240618-DP (copy 1)/')

    # util.save_face_points('/root/group-trainee/ay/version1/main/model_person/man.png',
    #                       '/root/group-trainee/ay/version1/main/model/512/man256.npy', trg_size=(256, 256))

    # util.rename_fn('/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/0802_短发原图67p/res/org/')
    # util.get_finalized_hair_data('/root/group-inspect-data/证件照换装/原始数据/3已定稿/补发修图/',
    #                     '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/a_所有定稿数据/', sub_dirs=['hair', 'hair_tar', 'ref'])
    # util.find_missing_files('/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/hair/',
    #                         '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/pre_cloth/')


    # for dir_path, _ , fns in os.walk('/root/group-inspect-data/证件照换装/中间结果/头发送修/data_ssd_18/DP_618_2_v1/ref/need/15p/'):
    #     if dir_path != '/root/group-inspect-data/证件照换装/中间结果/头发送修/data_ssd_18/DP_618_2_v1/ref/need/15p/':
    #         for fn in fns:
    #             pre ,post = os.path.splitext(fn)
    #             if '.' not in pre and post != '':
    #                 trg_fn = pre + 'v2' + post
    #                 if pre + 'v2-ref.png' in os.listdir('/root/group-inspect-data/证件照换装/中间结果/头发送修/data_ssd_18/DP_618_2_v1/ref/need/15p/'):
    #                     os.rename(os.path.join(dir_path, fn), os.path.join(dir_path, trg_fn))


    # util.create_gif('/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-26/res/all/')
    # util.move_files('/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-26/', src_sub_dirs = ['头发在前', '头发在前_b_hair_refine'])

    # util.copy_from_res('/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-26/res/all/',
    #                    '/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-26/res/3/',
    #                    '/root/group-inspect-data/证件照换装/中间结果/a_效果版本迭代结果/补发/2024-08-26/res/all')

    # util.copy_files_from_resdir('/data_ssd/ay/hair_to_ps/20240816DP/结果/ref/完美图/', '/data_ssd/ay/hair_to_ps/20240816DP/结果_头发在后/',
    #                        sub_dirs=['back_cloth', 'pre_cloth', 'coor', 'hair', 'org', 'org_cloth_mask', 'ref'])
    #
    # util.add_masks_from_diff_dirs(['/root/group-trainee/ay/version1/dataset/a_online512/test_Stage2_no_shadow_white/fake_neck_mask',
    #                                '/root/group-trainee/ay/version1/dataset/a_online512/test_Stage2_no_shadow_white/real_neck_mask'],
    #                               '/root/group-trainee/ay/version1/dataset/a_online512/test_Stage2_no_shadow_white/model_neck_mask')


    util.val_hair_data_final('/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/hair_tar/',
                             '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/person_mask/',
                             '/data_ssd/ay/hair_to_ps/获取衣服mask（所有头发送修图送修）/HairCompletion_train/normal_v4/a_val')