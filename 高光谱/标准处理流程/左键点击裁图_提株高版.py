from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


class ImageCropper:
    def __init__(self, root):
        self.root = root
        self.root.title("图像裁剪工具")

        self.image_paths = []  # 存储加载的图像路径
        self.depth_image_paths = []  # 存储加载的深度图像路径
        self.current_image_index = 0  # 当前显示图像的索引
        self.name_flag = 0
        self.save_flag = 0
        self.plant_height_1st = 0

        self.crop_start = None
        self.crop_box = None
        self.crop_box_coordinates = None
        self.save_image = None
        self.folder_path1 = None
        self.folder_path2 = None
        self.file_path = None
        self.pre_filename = None

        self.canvas_width = 1280
        self.canvas_height = 720
        self.crop_width = 1000
        self.crop_height = 200
        self.cam_fix_height = 1800
        self.canvas = Canvas(root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack(expand=YES, fill=BOTH)
        # 创建背景
        # 加载图片
        self.image_path = r"H:\FX17-xhy\erz\hps\sd1\band_4_div_band_3_OTSU.png"
        self.background_image = PhotoImage(file=self.image_path)
        # 将图片作为背景放置在Canvas上
        self.canvas.create_image(0, 0, anchor="nw", image=self.background_image)
        # 使用左键框选
        self.canvas.bind("<Button-1>", self.start_crop)
        # self.canvas.bind("<B1-Motion>", self.crop_drag)
        # self.canvas.bind("<ButtonRelease-1>", self.end_crop)
        # 使用右键保存
        # self.canvas.bind("<Button-3>", self.crop_and_save)
        # 键盘翻页
        root.bind("<KeyPress-A>", self.show_prev_image_keypress)
        root.bind("<KeyPress-D>", self.show_prev_image_keypress)
        root.bind("<KeyPress-a>", self.show_prev_image_keypress)
        root.bind("<KeyPress-d>", self.show_prev_image_keypress)
        # # 鼠标翻页
        # root.bind("<MouseWheel>", self.on_mouse_wheel)

        self.status_label = Label(root, text="")
        self.status_label.pack()

        self.load_button = Button(root, text="选择color文件夹", command=self.load_images_from_folder, font=20, width=20)
        self.load_button.pack()
        self.load_button1 = Button(root, text="选择depth文件夹", command=self.load_depth_images_from_folder, font=20, width=20)
        self.load_button1.pack()
        self.load_button2 = Button(root, text="选择txt文档", command=self.load_txt_file, font=20, width=20)
        self.load_button2.pack()
        # 将三个按钮水平排列
        self.load_button.pack(side=LEFT, padx=5, pady=5)
        self.load_button1.pack(side=LEFT, padx=5, pady=5)
        self.load_button2.pack(side=LEFT, padx=5, pady=5)

        self.prev_button = Button(root, text="上一张", command=self.show_prev_image)
        self.prev_button.pack(side=LEFT, padx=10)

        self.next_button = Button(root, text="下一张", command=self.show_next_image)
        self.next_button.pack(side=RIGHT, padx=10)

        self.save_folder = None
        # self.save_button = Button(root, text="选择保存文件夹", command=self.choose_save_folder)
        # self.save_button.pack()

        # self.crop_button = Button(root, text="裁剪并保存", command=self.crop_and_save)
        # self.crop_button.pack()
        #
        # self.save_as_png_button = Button(root, text="保存为PNG", command=self.save_as_png)
        # self.save_as_png_button.pack(side=LEFT, padx=10)
        #
        # self.save_as_jpg_button = Button(root, text="保存为JPG", command=self.save_as_jpg)
        # self.save_as_jpg_button.pack(side=RIGHT, padx=10)

    def on_mouse_wheel(self, event):
        if event.delta > 0:
            # 向上滚动鼠标滚轮，显示上一页图像
            self.show_prev_image()
        else:
            # 向下滚动鼠标滚轮，显示下一页图像
            self.show_next_image()

    def show_prev_image_keypress(self, event):
        # 检查按下的键是否为 'a' 键
        if event.char == 'a' or event.char == 'A':
            self.show_prev_image()
        if event.char == 'd' or event.char == 'D':
            self.show_next_image()

    def save_as_png(self):
        self.save_image = "PNG"

    def save_as_jpg(self):
        self.save_image = "JPG"

    # 自定义排序函数，按文件名中的数字大小进行排序
    def sort_by_number(self, file_name):
        # 使用正则表达式从文件名中提取所有数字和小数点，并按数字大小进行排序
        numbers = re.findall(r'\d+\.\d+|\d+', file_name)
        return [float(num) for num in numbers]

    def load_images_from_folder(self):
        # 注意文件夹路径不能有中文括号等
        self.folder_path1 = filedialog.askdirectory(title="选择color文件夹")
        if self.folder_path1:
            self.image_paths = [os.path.join(self.folder_path1, file) for file in os.listdir(self.folder_path1) if
                                file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.image_paths = sorted(self.image_paths, key=self.sort_by_number)
            print(self.image_paths)
            if self.image_paths:
                self.current_image_index = 0
                self.show_current_image()

    def load_depth_images_from_folder(self):
        self.folder_path2 = filedialog.askdirectory(title="选择depth文件夹")
        if self.folder_path2:
            self.depth_image_paths = [os.path.join(self.folder_path2, file) for file in os.listdir(self.folder_path2) if
                                      file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.depth_image_paths = sorted(self.depth_image_paths, key=self.sort_by_number)
        # return self.depth_image_paths

    def load_txt_file(self):
        # 打开文件对话框选择txt文档
        self.file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])

    def show_prev_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index - 1) % len(self.image_paths)
            self.show_current_image()

    def show_next_image(self):
        if self.image_paths:
            self.current_image_index = (self.current_image_index + 1) % len(self.image_paths)
            self.show_current_image()

    # def choose_save_folder(self):
    #     self.save_folder = filedialog.askdirectory(title="选择保存文件夹")

    def show_current_image(self):
        if self.image_paths:
            img = Image.open(self.image_paths[self.current_image_index])
            img.thumbnail((img.width // 3 * 2, img.height // 3 * 2))  # 缩小2/3
            self.photo = ImageTk.PhotoImage(img)
            self.canvas.config(width=self.canvas_width, height=self.canvas_height)
            self.canvas.create_image(0, 0, anchor=NW, image=self.photo)
            # 获取图像的中心位置
            center_x = self.canvas_width // 2
            center_y = self.canvas_height // 2
            # 在图像中心绘制一条横线
            self.canvas.create_line(0, center_y, self.canvas_width, center_y, fill='red', dash=(3, 3), width=4)
            self.status_label.config(text=f"图像 {self.current_image_index + 1}/{len(self.image_paths)}")

    def start_crop(self, event):
        self.crop_start = (event.x, event.y)
        if self.crop_start:
            # current_x, current_y = event.x, event.y
            point_lup = ((self.crop_start[0]), (self.crop_start[1]))
            point_rdn = ((self.crop_start[0]), (self.crop_start[1]))
            self.crop_box_coordinates = (
                point_lup[0]*3//2-self.crop_width/2,
                point_lup[1]*3//2-self.crop_height/2,
                point_rdn[0]*3//2+self.crop_width/2,
                point_rdn[1]*3//2+self.crop_height/2
            )
            self.crop_box_coordinates = [int(num) for num in self.crop_box_coordinates]
            self.status_label.config(text="已选择裁剪区域：{}".format(self.crop_box_coordinates))
            # print(self.crop_box_coordinates)
            #显示红色边框
            self.canvas.delete("crop_box")
            self.crop_box = self.canvas.create_rectangle(
                point_lup[0]-self.crop_width/2*2/3, point_lup[1]-self.crop_height/2*2/3,
                point_rdn[0]+self.crop_width/2*2/3, point_rdn[1]+self.crop_height/2*2/3,
                outline="BLACK", tags="crop_box", width=3
            )
            # 打印框内的灰度值
            image_index = self.current_image_index + 1
            # print("e", self.depth_image_paths[image_index])
            image = cv2.imread(self.depth_image_paths[image_index], cv2.IMREAD_UNCHANGED)
            # print(file_names[image_index])
            # 在这里可以进行你想要的处理，比如打印灰度值等操作
            plant_height = self.print_grayscale_values(image, self.crop_box_coordinates[0], self.crop_box_coordinates[1], self.crop_box_coordinates[2], self.crop_box_coordinates[3])
            # # 将点击两次的值做平均
            # if self.save_flag == 1:
            #     plant_height_fin = (self.plant_height_1st + plant_height)/2
            #     with open(self.file_path, 'a') as file:
            #         file.write(f"{plant_height_fin}\n")
            #         self.save_flag = 0
            #         self.plant_height_1st = 0
            # else:
            #     self.save_flag += 1
            #     self.plant_height_1st = plant_height
            # print(plant_height)
            with open(self.file_path, 'a') as file:
                 file.write(f"{plant_height}\n")



    def crop_and_save(self, event=None):

        self.save_folder = os.path.join(os.path.dirname(self.folder_path1), "cropped_" + os.path.basename(self.folder_path1))

        if self.image_paths and self.crop_box_coordinates and self.save_folder:
            try:

                if not os.path.exists(self.save_folder):
                    os.makedirs(self.save_folder)

                img = Image.open(self.image_paths[self.current_image_index])
                cropped_img = img.crop(self.crop_box_coordinates)
                file_name = os.path.basename(self.image_paths[self.current_image_index])
                if self.save_image == 'PNG':
                    file_name = file_name.split('.')[0] + '.PNG'
                elif self.save_image == 'JPG':
                    file_name = file_name.split('.')[0] + '.JPG'
                output_path = os.path.join(self.save_folder, "cropped_" + file_name)
                i = 1
                while os.path.exists(output_path):
                    output_path = os.path.join(self.save_folder, "cropped_" + str(i) + '_' + file_name)
                    i += 1
                cropped_img.save(output_path)
                self.status_label.config(text="裁剪并保存成功，图像已保存到：" + output_path)
            except Exception as e:
                self.status_label.config(text="发生错误：" + str(e))
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_lineno = exc_traceback.tb_lineno
                print(f"The exception occurred on line {tb_lineno}.")
                print(f"An exception of type {type(e).__name__} occurred with message: {str(e)}")

    def print_grayscale_values(self, image, x1, y1, x2, y2):

        height, width = image.shape
        pixel_values = []
        # print("x1y1x2y2",x1, y1, x2, y2)
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if x >= 0 and x < width and y >= 0 and y < height:
                    pixel_value = image[y, x]
                    pixel_values.append(pixel_value)
                    # print(f"像素 ({x}, {y}) 的灰度值为：{pixel_value}")
        depth = self.cluster_filter(pixel_values)
        # print("depth", depth)
        plant_height = int((self.cam_fix_height - depth)/10)
        # print("该区域平均深度为为：", plant_height)
        return plant_height




    def cluster_filter(self, array):
        array = np.array(array)
        array = array.astype(float)
        # # 初级聚类方法，使用kmeans聚类，出问题的原因是框大了，把两边的沟也框进去了，但也不太稳定
        # # 将数组聚类
        # centroids, _ = kmeans(array, 3)  # 聚成3类
        # groups, _ = vq(array, centroids)
        # # 计算每个聚类的平均值
        # cluster_means = [np.mean(array[groups == i]) for i in range(len(centroids))]
        # 高级聚类方法，聚类效果提升
        # 使用 KMeans 聚类
        array_2d = array.reshape(-1, 1)
        kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200)  # 聚成3类
        kmeans.fit(array_2d)
        centroids = kmeans.cluster_centers_
        groups = kmeans.labels_
        # print(len(centroids))
        # 计算每个聚类的平均值
        cluster_means = [np.mean(array[groups == i]) for i in range(len(centroids))]

        # 找到平均值最大和最小的两类
        max_mean_index = np.argmax(cluster_means)
        min_mean_index = np.argmin(cluster_means)

        # 去掉平均值最大和最小的两类
        filtered_array = array[np.logical_and(groups != max_mean_index, groups != min_mean_index)]

        # 计算剩下聚类的平均值
        average_value = np.mean(filtered_array)

        # print("原始数组:", array)
        # print("聚类平均值:", cluster_means)
        # print("去除最大和最小平均值后的数组:", filtered_array)
        # print("剩下聚类的平均值:", average_value)
        return average_value



# 创建主窗口
root = Tk()

# 创建ImageCropper实例
cropper = ImageCropper(root)

# 启动主循环
root.mainloop()
