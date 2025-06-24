import os
import sys
from tkinter import Tk, Label, Button, filedialog, Canvas, PhotoImage, messagebox
from PIL import Image, ImageTk
import shutil

class ImageSorter:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Sorter")
        self.master.geometry("800x800")

        self.current_folder = ""
        self.image_paths = []
        self.current_image_index = 0
        self.image_label = None
        # self.sorted_folders = {"Folder1":'/root/group-trainee/ay/version1/dataset/a_online512/train/all_no_shadow/no_shadow_exactly/',
        #                        "Folder2": '/root/group-trainee/ay/version1/dataset/a_online512/train/all_no_shadow/shadow/'}

        self.sorted_folders = {"Folder1":'/data_ssd/ay/ID_DATA/肩宽点/imgs/阴影/脖子两边/',
                               "Folder2":'/data_ssd/ay/ID_DATA/肩宽点/imgs/阴影/衣领处/'}

        self.load_button = Button(master, text="Load Images", command=self.load_images)
        self.load_button.pack(fill="x")

        self.canvas = Canvas(master)
        self.canvas.pack(fill="both", expand=True)

        self.image_label = Label(master)
        self.image_label.pack()

        self.master.bind('<Right>', self.move_right)
        self.master.bind('<Left>', self.move_left)
        self.master.bind('<Up>', self.move_up)
        self.master.bind('<Down>', self.move_down)

    def load_images(self):
        self.current_folder = filedialog.askdirectory()
        if not self.current_folder:
            return
        self.image_paths = [os.path.join(self.current_folder, img) for img in os.listdir(self.current_folder) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.update_image()

    # def update_image(self):
    #     if not self.image_paths:
    #         return
    #     image_path = self.image_paths[self.current_image_index]
    #     image = Image.open(image_path)
    #     photo = ImageTk.PhotoImage(image)
    #     self.image_label.config(image=photo)
    #     self.image_label.image = photo  # Keep a reference!
    #     messagebox.showinfo("Image Path", image_path)

    def update_image(self):
        if not self.image_paths or self.current_image_index >= len(self.image_paths):
            return
        image_path = self.image_paths[self.current_image_index]
        try:
            image = Image.open(image_path)
            # 确保图片不超过2000x2000的分辨率，如果图片太大则缩小到这个尺寸
            max_display_size = (1000, 1000)
            size = image.size
            if size[0] > max_display_size[0] or size[1] > max_display_size[1]:
                # 计算缩放比例
                scale_x = max_display_size[0] / size[0]
                scale_y = max_display_size[1] / size[1]
                scale = min(scale_x, scale_y)
                # 应用缩放
                new_size = (int(scale * size[0]), int(scale * size[1]))
                image = image.resize(new_size, Image.Resampling.BICUBIC)

            tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=tk_image)
            self.image_label.image = tk_image  # Keep a reference to prevent GC
            self.canvas.config(scrollregion=self.canvas.bbox("all"))
            # messagebox.showinfo("Image Path", image_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def move_right(self, event):
        self.move_to_folder("Folder2")

    def move_left(self, event):
        self.move_to_folder("Folder1")

    def move_to_folder(self, folder_name):
        if self.current_image_index < len(self.image_paths):
            src_path = os.path.join(self.current_folder, self.image_paths[self.current_image_index])
            dst_path = self.sorted_folders.get(folder_name)
            if not dst_path:
                self.status_bar.config(text="Invalid folder name")
                return
            os.makedirs(dst_path, exist_ok=True)
            dst_file = os.path.join(dst_path, os.path.basename(self.image_paths[self.current_image_index]))
            shutil.move(src_path, dst_file)
            self.image_paths.remove(self.image_paths[self.current_image_index])
            # self.current_image_index -= 1  # Adjust index since we removed an item
            self.update_image()
            self.status_bar.config(text="Image saved successfully")

    def move_up(self, event):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.update_image()

    def move_down(self, event):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.update_image()

    def move_image_to_folder(self, index, folder_name):
        if not self.image_paths:
            return
        current_image_path = self.image_paths[index]
        folder_path = os.path.join(self.current_folder, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        basename = os.path.basename(current_image_path)
        shutil.copy(current_image_path, os.path.join(folder_path, basename))

if __name__ == "__main__":
    root = Tk()
    app = ImageSorter(root)
    root.mainloop()