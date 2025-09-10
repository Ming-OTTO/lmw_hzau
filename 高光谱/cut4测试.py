import tkinter as tk
from tkinter import filedialog, ttk, messagebox, simpledialog, scrolledtext
from tkinter import PhotoImage
import os
import numpy as np
import cv2
import threading
from PIL import Image, ImageTk, ImageDraw, ImageFont


class HyperspectralImageCutter:
    def __init__(self, root):
        self.root = root
        self.root.title("高光谱图像处理工具")
        self.root.geometry("1200x800")

        # 初始化变量
        self.rgb_image_path = ""
        self.hsi_data_path = ""
        self.output_path = ""
        self.original_rgb_image = None
        self.display_rgb_image = None
        self.roi = None
        self.cropped_image = None
        self.scale_factor = 1.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.selecting_roi = False
        self.roi_start = (0, 0)
        self.roi_end = (0, 0)

        # 创建UI
        self.create_ui()

        # 确保中文显示
        self.set_chinese_font()

    def set_chinese_font(self):
        try:
            # 尝试加载中文字体
            self.font = ImageFont.truetype("simhei.ttf", 12)
        except:
            # 如果失败，尝试系统字体
            try:
                self.font = ImageFont.truetype("msyh.ttc", 12)
            except:
                # 最后尝试默认字体
                self.font = ImageFont.load_default()

    def create_ui(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 标题
        title_label = ttk.Label(main_frame, text="高光谱图像处理工具", font=("", 16, "bold"))
        title_label.pack(pady=(0, 10))

        # 输入输出部分
        io_frame = ttk.LabelFrame(main_frame, text="输入/输出设置")
        io_frame.pack(fill=tk.X, padx=5, pady=5)

        # RGB图像路径
        rgb_frame = ttk.Frame(io_frame)
        rgb_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(rgb_frame, text="RGB图像路径:").pack(side=tk.LEFT, padx=(0, 5))
        self.rgb_path_entry = ttk.Entry(rgb_frame, width=70)
        self.rgb_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(rgb_frame, text="浏览...", command=self.browse_rgb_image).pack(side=tk.LEFT)

        # 高光谱数据路径
        hsi_frame = ttk.Frame(io_frame)
        hsi_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(hsi_frame, text="高光谱数据路径:").pack(side=tk.LEFT, padx=(0, 5))
        self.hsi_path_entry = ttk.Entry(hsi_frame, width=70)
        self.hsi_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(hsi_frame, text="浏览...", command=self.browse_hsi_data).pack(side=tk.LEFT)

        # 输出路径
        output_frame = ttk.Frame(io_frame)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(output_frame, text="输出路径:").pack(side=tk.LEFT, padx=(0, 5))
        self.output_path_entry = ttk.Entry(output_frame, width=70)
        self.output_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(output_frame, text="浏览...", command=self.browse_output_path).pack(side=tk.LEFT)

        # 预览和操作部分
        preview_frame = ttk.Frame(main_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左面板 - 原始图像
        left_frame = ttk.LabelFrame(preview_frame, text="原始图像预览")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 图像导航控制
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(nav_frame, text="放大 (+)", command=lambda: self.zoom_image(1.2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="缩小 (-)", command=lambda: self.zoom_image(0.8)).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="重置缩放", command=self.reset_zoom).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="选择区域", command=self.select_roi).pack(side=tk.LEFT, padx=2)
        self.roi_label = ttk.Label(nav_frame, text="区域: 未选择")
        self.roi_label.pack(side=tk.LEFT, padx=10)

        # 图像显示区域
        self.image_frame = ttk.Frame(left_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建Canvas用于显示图像
        self.canvas = tk.Canvas(self.image_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # 添加滚动条
        self.v_scroll = ttk.Scrollbar(self.image_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self.image_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)

        # 放置滚动条
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)

        # 事件绑定
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux (up)
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux (down)

        # 右面板 - 预览和日志
        right_frame = ttk.Frame(preview_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)

        # 切割结果预览
        crop_frame = ttk.LabelFrame(right_frame, text="切割结果预览")
        crop_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.crop_canvas = tk.Canvas(crop_frame, bg="gray", height=300)
        self.crop_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 操作按钮
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(button_frame, text="预览图像", command=self.preview_images).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="应用切割", command=self.apply_cut).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(button_frame, text="保存结果", command=self.save_results).pack(side=tk.LEFT, padx=5, pady=5)

        # 日志区域
        log_frame = ttk.LabelFrame(right_frame, text="处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)

    def set_status(self, message):
        self.status_var.set(message)
        self.log(message)

    def browse_rgb_image(self):
        file_path = filedialog.askopenfilename(
            title="选择RGB图像",
            filetypes=[("图像文件", "*.png *.jpg *.jpeg *.tif *.tiff"), ("所有文件", "*.*")]
        )
        if file_path:
            self.rgb_path_entry.delete(0, tk.END)
            self.rgb_path_entry.insert(0, file_path)
            self.rgb_image_path = file_path

    def browse_hsi_data(self):
        file_path = filedialog.askopenfilename(
            title="选择高光谱数据",
            filetypes=[("数据文件", "*.hdr *.raw *.npy *.mat"), ("所有文件", "*.*")]
        )
        if file_path:
            self.hsi_path_entry.delete(0, tk.END)
            self.hsi_path_entry.insert(0, file_path)
            self.hsi_data_path = file_path

    def browse_output_path(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_path_entry.delete(0, tk.END)
            self.output_path_entry.insert(0, dir_path)
            self.output_path = dir_path

    def preview_images(self):
        if not self.rgb_image_path:
            self.set_status("错误: 请先选择RGB图像路径")
            return

        try:
            self.set_status("正在加载RGB图像...")
            # 启动一个新线程加载图像，避免界面冻结
            threading.Thread(target=self.load_rgb_image, daemon=True).start()
        except Exception as e:
            self.set_status(f"图像加载错误: {str(e)}")

    def load_rgb_image(self):
        try:
            # 使用PIL加载大图，只加载元数据
            with Image.open(self.rgb_image_path) as img:
                width, height = img.size

            self.log(f"图像尺寸: {width}×{height}")

            # 对于超大图像，只加载一部分
            if height > 30000 or width > 8000:
                self.set_status("图像非常大，正在加载缩略图...")
                # 计算缩放比例
                scale_factor = min(1.0, 2000 / height, 6000 / width)

                # 读取缩放后的图像
                self.original_rgb_image = cv2.imread(self.rgb_image_path)
                if self.original_rgb_image is not None:
                    self.display_rgb_image = cv2.resize(
                        self.original_rgb_image,
                        (int(width * scale_factor), int(height * scale_factor)),
                        interpolation=cv2.INTER_AREA
                    )
                    self.scale_factor = scale_factor
                else:
                    raise Exception("无法加载图像，请检查文件路径")
            else:
                # 正常尺寸图像
                self.original_rgb_image = cv2.imread(self.rgb_image_path)
                if self.original_rgb_image is not None:
                    self.display_rgb_image = self.original_rgb_image.copy()
                    self.scale_factor = 1.0
                else:
                    raise Exception("无法加载图像，请检查文件路径")

            # 更新界面显示
            self.root.after(0, self.display_loaded_image)
        except Exception as e:
            # 修复lambda作用域问题
            error_msg = str(e)
            self.root.after(0, lambda: self.set_status(f"图像加载错误: {error_msg}"))

    def display_loaded_image(self):
        if self.display_rgb_image is None:
            return

        try:
            # 转换为PIL格式用于显示
            pil_img = self.convert_cv_to_pil(self.display_rgb_image)

            # 更新Canvas尺寸
            self.update_canvas_size(pil_img.width, pil_img.height)

            # 创建Canvas图像
            self.canvas_image = ImageTk.PhotoImage(pil_img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)

            # 配置滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

            self.set_status(f"图像已加载: {self.display_rgb_image.shape[1]}×{self.display_rgb_image.shape[0]}像素")
        except Exception as e:
            error_msg = str(e)
            self.set_status(f"图像显示错误: {error_msg}")

    def convert_cv_to_pil(self, cv_img):
        # OpenCV使用BGR，转换为RGB
        if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:  # 彩色图像
            cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_img_rgb)
        elif len(cv_img.shape) == 2:  # 灰度图像
            pil_img = Image.fromarray(cv_img)
        else:
            # 尝试处理其他格式
            pil_img = Image.fromarray(cv_img.astype('uint8'))

        return pil_img

    def update_canvas_size(self, width, height):
        self.canvas.config(width=min(width, 800), height=min(height, 600))

    def on_canvas_click(self, event):
        if self.selecting_roi:
            # 记录ROI起点（画布坐标）
            self.roi_start = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            self.roi_end = self.roi_start
            # 清除之前的ROI矩形
            self.canvas.delete("roi_rect")
            # 创建一个新的ROI矩形
            self.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1],
                self.roi_end[0], self.roi_end[1],
                outline="green", width=2, tags="roi_rect"
            )

    def on_canvas_drag(self, event):
        if self.selecting_roi and self.roi_start:
            # 更新ROI终点（画布坐标）
            self.roi_end = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
            # 更新ROI矩形
            self.canvas.delete("roi_rect")
            # 确保矩形坐标正确（左上角到右下角）
            x1 = min(self.roi_start[0], self.roi_end[0])
            y1 = min(self.roi_start[1], self.roi_end[1])
            x2 = max(self.roi_start[0], self.roi_end[0])
            y2 = max(self.roi_start[1], self.roi_end[1])
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline="green", width=2, tags="roi_rect"
            )

    def on_canvas_release(self, event):
        if self.selecting_roi and self.roi_start and self.roi_end:
            # 计算实际图像坐标 (考虑缩放因子)
            orig_x = min(self.roi_start[0], self.roi_end[0]) / self.scale_factor
            orig_y = min(self.roi_start[1], self.roi_end[1]) / self.scale_factor
            orig_w = abs(self.roi_end[0] - self.roi_start[0]) / self.scale_factor
            orig_h = abs(self.roi_end[1] - self.roi_start[1]) / self.scale_factor

            # 保存ROI（原始图像上的坐标）
            self.roi = (int(orig_x), int(orig_y), max(1, int(orig_w)), max(1, int(orig_h)))
            self.roi_label.config(text=f"区域: X={int(orig_x)}, Y={int(orig_y)}, 宽={int(orig_w)}, 高={int(orig_h)}")
            self.set_status(f"区域选择完成: X={int(orig_x)}, Y={int(orig_y)}, 宽={int(orig_w)}, 高={int(orig_h)}")

            # 结束选择模式
            self.selecting_roi = False

            # 在图像上绘制ROI矩形（永久显示）
            if self.display_rgb_image is not None:
                try:
                    draw_rgb_image = self.display_rgb_image.copy()
                    # 计算缩放后的坐标
                    draw_x = int(self.roi[0] * self.scale_factor)
                    draw_y = int(self.roi[1] * self.scale_factor)
                    draw_w = int(self.roi[2] * self.scale_factor)
                    draw_h = int(self.roi[3] * self.scale_factor)

                    # 绘制矩形
                    cv2.rectangle(draw_rgb_image,
                                  (draw_x, draw_y),
                                  (draw_x + draw_w, draw_y + draw_h),
                                  (0, 255, 0), 2)

                    # 转换为PIL格式用于显示
                    pil_img = self.convert_cv_to_pil(draw_rgb_image)

                    # 创建Canvas图像
                    self.canvas_image = ImageTk.PhotoImage(pil_img)
                    self.canvas.delete("all")
                    self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)
                except Exception as e:
                    error_msg = str(e)
                    self.set_status(f"绘制ROI错误: {error_msg}")

    def on_mousewheel(self, event):
        # 处理Windows和Linux的滚轮事件
        if event.num == 5 or (hasattr(event, "delta") and event.delta < 0):
            self.zoom_image(0.8)
        elif event.num == 4 or (hasattr(event, "delta") and event.delta > 0):
            self.zoom_image(1.2)

    def zoom_image(self, factor):
        if self.display_rgb_image is None:
            return

        self.scale_factor *= factor

        # 限制缩放范围
        self.scale_factor = max(0.1, min(self.scale_factor, 5.0))

        # 计算新尺寸
        height, width = self.display_rgb_image.shape[:2]
        new_width = int(width * self.scale_factor)
        new_height = int(height * self.scale_factor)

        # 缩放图像
        scaled_img = cv2.resize(self.display_rgb_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # 转换为PIL格式用于显示
        pil_img = self.convert_cv_to_pil(scaled_img)

        # 更新Canvas尺寸
        self.update_canvas_size(new_width, new_height)

        # 创建Canvas图像
        self.canvas_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)

        # 配置滚动区域
        self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

        self.set_status(f"缩放级别: {self.scale_factor:.2f}x")

    def reset_zoom(self):
        self.scale_factor = 1.0
        if self.display_rgb_image is not None:
            # 恢复原始显示大小
            pil_img = self.convert_cv_to_pil(self.display_rgb_image)
            self.update_canvas_size(pil_img.width, pil_img.height)

            # 创建Canvas图像
            self.canvas_image = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_image)

            # 配置滚动区域
            self.canvas.configure(scrollregion=self.canvas.bbox(tk.ALL))

            self.set_status("缩放已重置")

    def select_roi(self):
        if self.display_rgb_image is None:
            self.set_status("错误: 请先加载图像")
            return

        # 开始选择ROI模式
        self.selecting_roi = True
        self.set_status("请点击并拖动鼠标来选择区域")
        # 设置光标为十字
        self.canvas.config(cursor="crosshair")

    def apply_cut(self):
        if self.roi is None:
            self.set_status("错误: 请先选择切割区域")
            return

        x, y, w, h = self.roi

        # 确保区域在图像范围内
        if self.original_rgb_image is None:
            self.set_status("错误: 没有加载图像")
            return

        height, width = self.original_rgb_image.shape[:2]
        if x < 0 or y < 0 or x + w > width or y + h > height:
            self.set_status("错误: 区域超出图像范围")
            return

        # 模拟切割操作
        self.cropped_image = self.original_rgb_image[y:y + h, x:x + w]

        # 显示切割结果
        if self.cropped_image is not None and self.cropped_image.size > 0:
            try:
                # 转换为PIL格式
                cropped_pil = self.convert_cv_to_pil(self.cropped_image)

                # 创建Canvas图像
                self.crop_image_tk = ImageTk.PhotoImage(cropped_pil)
                self.crop_canvas.delete("all")
                self.crop_canvas.create_image(0, 0, anchor=tk.NW, image=self.crop_image_tk)

                # 更新Canvas尺寸
                self.crop_canvas.config(width=cropped_pil.width, height=cropped_pil.height)

                self.set_status(f"切割完成: {cropped_pil.width}×{cropped_pil.height}像素")
            except Exception as e:
                error_msg = str(e)
                self.set_status(f"显示切割结果错误: {error_msg}")
        else:
            self.set_status("错误: 切割区域无效")

    def save_results(self):
        if self.cropped_image is None:
            self.set_status("错误: 没有可保存的结果")
            return

        if not self.output_path:
            self.set_status("错误: 请先选择输出路径")
            return

        try:
            # 模拟保存高光谱数据
            output_file = os.path.join(self.output_path, "cropped_result.png")

            # 实际应用中应该保存高光谱数据，这里保存RGB图像作为示例
            cv2.imwrite(output_file, self.cropped_image)

            self.set_status(f"结果已保存到: {output_file}")
            messagebox.showinfo("保存成功", f"切割结果已保存到:\n{output_file}")
        except Exception as e:
            error_msg = str(e)
            self.set_status(f"保存错误: {error_msg}")


def main():
    root = tk.Tk()
    app = HyperspectralImageCutter(root)
    root.mainloop()


if __name__ == "__main__":
    main()