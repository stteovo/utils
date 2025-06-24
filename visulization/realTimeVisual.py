import sys
import os
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class ImageSwitcher(QMainWindow):
    def __init__(self, process_func, default_dir='/data_ssd/TestData/augmentation/'):
        super().__init__()
        self.process_func = process_func
        self.default_dir = default_dir

        # 初始化窗口大小
        screen_geometry = QApplication.primaryScreen().geometry()
        self.setGeometry(0, 0, screen_geometry.width() * 2 // 3, screen_geometry.height() * 2 // 3)

        # 图像显示标签
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.label)

        # 背景颜色设置，默认为黑色
        self.bg_color = 'black'
        self.set_background_color()

        # 图片路径和当前索引
        self.image_folder = self.select_folder()
        self.image_list = self.load_images(self.image_folder)
        self.current_index = 0

        # 显示处理后的图像标志
        self.show_processed = False
        self.previous_image = None
        self.space_pressed = False  # 空格键按下状态标志

        # 显示第一个图像
        self.show_image()

    def select_folder(self):
        """选择图片文件夹"""
        folder = QFileDialog.getExistingDirectory(self, "选择图像文件夹", self.default_dir)
        return folder
        return None

    def load_images(self, folder):
        """加载图像文件夹中的图片"""
        valid_extensions = ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']
        return [QImage(os.path.join(folder, f)) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_extensions]

    def show_image(self):
        """显示当前图像（原始或处理后）"""
        if self.image_list:
            qt_image = self.image_list[self.current_index]
            # 获取图像的宽度和高度，计算最长边，并将图像缩放到最长边不超过1024
            width = qt_image.width()
            height = qt_image.height()
            max_size = 1024
            if max(width, height) > max_size:
                scale_factor = max_size / max(width, height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                qt_image = qt_image.scaled(new_width, new_height, Qt.KeepAspectRatio)

            if self.show_processed:
                cv_image = self.qimage_to_cv(qt_image)
                processed_image = self.apply_image_processing(cv_image)
                qt_image = self.cv_to_qimage(processed_image)

            pixmap = QPixmap.fromImage(qt_image)
            self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def keyPressEvent(self, event):
        """处理键盘按键事件"""
        if event.key() == Qt.Key_Space and not self.space_pressed:
            # 切换原始图像和处理后的图像
            self.space_pressed = True  # 设置空格键已被按下
            self.show_processed = True
            self.show_image()

        elif event.key() == Qt.Key_A:
            # 上一张图片
            self.current_index = (self.current_index - 1) % len(self.image_list)
            self.show_processed = False
            self.show_image()

        elif event.key() == Qt.Key_D:
            # 下一张图片
            self.current_index = (self.current_index + 1) % len(self.image_list)
            self.show_processed = False
            self.show_image()

        elif event.key() == Qt.Key_B:
            # 切换背景颜色
            self.bg_color = 'white' if self.bg_color == 'black' else 'black'
            self.set_background_color()

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.space_pressed = False  # 空格键未被按下
            self.show_processed = False
            self.show_image()

    def set_background_color(self):
        """设置背景颜色"""
        self.setStyleSheet(f"background-color: {self.bg_color};")

    def mousePressEvent(self, event):
        """处理鼠标按下事件"""
        if event.button() == Qt.LeftButton:
            # 点击鼠标左键，切换图像
            self.show_processed = not self.show_processed
            self.show_image()

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        if event.button() == Qt.LeftButton:
            # 释放鼠标左键，返回上一张图片
            self.show_processed = not self.show_processed
            self.show_image()

    def resizeEvent(self, event):
        """处理窗口大小变化事件"""
        self.show_image()

    def qimage_to_cv(self, qimg):
        """将 QImage 转换为 cv::Mat 格式"""
        qimg = qimg.convertToFormat(QImage.Format_ARGB32)  # 先转换为 ARGB32，统一处理
        width = qimg.width()
        height = qimg.height()
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        arr = np.array(ptr).reshape(height, width, 4)  # ARGB 是四通道

        # 去除 alpha 通道（如果不需要透明度），将图像转换为三通道 BGR 格式
        cv_image = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)

        # 判断是否为单通道灰度图
        if qimg.format() == QImage.Format_Grayscale8:
            cv_image = arr[:, :, 0]  # 只保留灰度值

        return cv_image

    def cv_to_qimage(self, cv_img):
        """将 cv::Mat 转换为 QImage 格式"""
        height, width = cv_img.shape[:2]

        if len(cv_img.shape) == 2:
            # 单通道灰度图
            qimg = QImage(cv_img.data, width, height, width, QImage.Format_Grayscale8)
        elif cv_img.shape[2] == 3:
            # 三通道 RGB 图像 (BGR 格式，需要转换)
            qimg = QImage(cv_img.data, width, height, cv_img.strides[0], QImage.Format_RGB888)
            qimg = qimg.rgbSwapped()  # cv::Mat 是 BGR 排列，需要转换为 RGB
        elif cv_img.shape[2] == 4:
            # 四通道带 Alpha 的图像
            qimg = QImage(cv_img.data, width, height, cv_img.strides[0], QImage.Format_ARGB32)
        else:
            raise ValueError("不支持的图像格式")

        return qimg
    def apply_image_processing(self, img, *args, **kwargs):
        """对图像进行处理，具体实现可由用户自定义"""
        # 在这里进行任意图像处理操作
        # 例如：将图像转换为灰度图
        processed_img = self.process_func(img, *args, **kwargs)
        processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

        return processed_img


def process_func(img):
    return cv2.GaussianBlur(img, (31,31), 0)


def visualize(process_func, default_dir='/data_ssd/TestData/augmentation/'):
    app = QApplication(sys.argv)
    window = ImageSwitcher(process_func, default_dir=default_dir)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    visualize(process_func, default_dir='/data_ssd/TestData/augmentation/')
