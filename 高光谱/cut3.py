
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import spectral as spy
import threading
import re

# 设置matplotlib支持中文显示
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class HyperspectralCutterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("高光谱图像切割工具")
        self.root.geometry("1200x800")

        # 初始化变量
        self.rgb_image_path = ""
        self.hyperspectral_path = ""
        self.output_path = ""
        self.rgb_image = None
        self.display_photo = None
        self.hyperspectral_data = None
        self.hyperspectral_format = None
        self.original_hdr_content = None
        self.dat_info = {}
        self.selection = None
        self.scale = 1.0
        self.is_selecting = False
        self.select_start_x, self.select_start_y = 0, 0

        # 创建界面
        self.create_widgets()

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # RGB图像路径选择
        ttk.Label(control_frame, text="RGB图像路径:").pack(anchor=tk.W, pady=(0, 5))
        rgb_frame = ttk.Frame(control_frame)
        rgb_frame.pack(fill=tk.X, pady=(0, 10))
        self.rgb_path_var = tk.StringVar()
        ttk.Entry(rgb_frame, textvariable=self.rgb_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                            padx=(0, 5))
        ttk.Button(rgb_frame, text="浏览...", command=self.select_rgb_image).pack(side=tk.LEFT)

        # 高光谱数据路径选择
        ttk.Label(control_frame, text="高光谱数据路径:").pack(anchor=tk.W, pady=(0, 5))
        hyper_frame = ttk.Frame(control_frame)
        hyper_frame.pack(fill=tk.X, pady=(0, 10))
        self.hyper_path_var = tk.StringVar()
        ttk.Entry(hyper_frame, textvariable=self.hyper_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                padx=(0, 5))
        ttk.Button(hyper_frame, text="浏览...", command=self.select_hyperspectral_data).pack(side=tk.LEFT)

        # DAT文件参数设置
        self.dat_params_frame = ttk.LabelFrame(control_frame, text="DAT文件参数", padding="5")

        ttk.Label(self.dat_params_frame, text="宽度 (像素):").pack(anchor=tk.W)
        self.dat_width_var = tk.StringVar()
        ttk.Entry(self.dat_params_frame, textvariable=self.dat_width_var, width=15).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.dat_params_frame, text="高度 (像素):").pack(anchor=tk.W)
        self.dat_height_var = tk.StringVar()
        ttk.Entry(self.dat_params_frame, textvariable=self.dat_height_var, width=15).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.dat_params_frame, text="波段数:").pack(anchor=tk.W)
        self.dat_bands_var = tk.StringVar()
        ttk.Entry(self.dat_params_frame, textvariable=self.dat_bands_var, width=15).pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.dat_params_frame, text="数据类型:").pack(anchor=tk.W)
        self.dat_dtype_var = tk.StringVar(value='float32')
        dtype_combo = ttk.Combobox(self.dat_params_frame, textvariable=self.dat_dtype_var,
                                   values=['uint8', 'int16', 'uint16', 'int32', 'uint32', 'float32', 'float64'],
                                   state='readonly')
        dtype_combo.pack(fill=tk.X, pady=(0, 5))

        ttk.Label(self.dat_params_frame, text="数据排列:").pack(anchor=tk.W)
        self.dat_interleave_var = tk.StringVar(value='bil')
        interleave_combo = ttk.Combobox(self.dat_params_frame, textvariable=self.dat_interleave_var,
                                        values=['bsq', 'bip', 'bil'], state='readonly')
        interleave_combo.pack(fill=tk.X)

        # 输出路径选择
        ttk.Label(control_frame, text="输出路径:").pack(anchor=tk.W, pady=(0, 5))
        output_frame = ttk.Frame(control_frame)
        output_frame.pack(fill=tk.X, pady=(0, 10))
        self.output_path_var = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True,
                                                                                  padx=(0, 5))
        ttk.Button(output_frame, text="浏览...", command=self.select_output_path).pack(side=tk.LEFT)

        # 操作按钮
        ttk.Button(control_frame, text="加载RGB图像", command=self.load_rgb_image).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(control_frame, text="加载高光谱数据", command=self.load_hyperspectral_data).pack(fill=tk.X,
                                                                                                    pady=(0, 5))

        # 缩放控制
        zoom_frame = ttk.LabelFrame(control_frame, text="缩放控制", padding="5")
        zoom_frame.pack(fill=tk.X, pady=(10, 0))

        zoom_buttons_frame = ttk.Frame(zoom_frame)
        zoom_buttons_frame.pack(fill=tk.X)

        ttk.Button(zoom_buttons_frame, text="放大", command=lambda: self.zoom(1.2)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(zoom_buttons_frame, text="缩小", command=lambda: self.zoom(0.8)).pack(side=tk.LEFT)
        ttk.Button(zoom_buttons_frame, text="适应窗口", command=self.fit_to_window).pack(side=tk.LEFT, padx=(5, 0))

        # 状态信息
        status_frame = ttk.LabelFrame(control_frame, text="状态", padding="5")
        status_frame.pack(fill=tk.X, pady=(10, 0))

        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=tk.W)

        # 框选信息
        selection_frame = ttk.LabelFrame(control_frame, text="框选区域", padding="5")
        selection_frame.pack(fill=tk.X, pady=(10, 0))

        self.selection_var = tk.StringVar(value="未选择")
        ttk.Label(selection_frame, textvariable=self.selection_var).pack(anchor=tk.W)

        # 处理按钮
        ttk.Button(control_frame, text="切割高光谱数据", command=self.cut_hyperspectral_data).pack(fill=tk.X,
                                                                                                   pady=(20, 0))

        # 右侧显示区域
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 创建选项卡控件
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # RGB预览选项卡
        self.rgb_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.rgb_tab, text="RGB预览")

        # 创建画布和滚动条
        self.rgb_canvas_frame = ttk.Frame(self.rgb_tab)
        self.rgb_canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.h_scrollbar = ttk.Scrollbar(self.rgb_canvas_frame, orient=tk.HORIZONTAL)
        self.h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.v_scrollbar = ttk.Scrollbar(self.rgb_canvas_frame, orient=tk.VERTICAL)
        self.v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.rgb_canvas = tk.Canvas(self.rgb_canvas_frame, bg="black",
                                    xscrollcommand=self.h_scrollbar.set,
                                    yscrollcommand=self.v_scrollbar.set)
        self.rgb_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.h_scrollbar.config(command=self.rgb_canvas.xview)
        self.v_scrollbar.config(command=self.rgb_canvas.yview)

        # 结果预览选项卡
        self.result_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.result_tab, text="结果预览")

        self.result_canvas = tk.Canvas(self.result_tab, bg="black")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)

        # 状态栏
        self.status_bar = ttk.Label(self.root, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 绑定鼠标事件
        self.rgb_canvas.bind("<Button-1>", self.on_canvas_click)
        self.rgb_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.rgb_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.rgb_canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.rgb_canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux 滚轮上滚
        self.rgb_canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux 滚轮下滚

        # 绑定画布大小变化事件
        self.rgb_canvas.bind("<Configure>", self.on_canvas_configure)

    def select_rgb_image(self):
        file_path = filedialog.askopenfilename(
            title="选择RGB图像",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")]
        )
        if file_path:
            self.rgb_image_path = file_path
            self.rgb_path_var.set(file_path)

    def select_hyperspectral_data(self):
        file_path = filedialog.askopenfilename(
            title="选择高光谱数据",
            filetypes=[
                ("ENVI格式", "*.hdr"),
                ("DAT文件", "*.dat"),
                ("所有支持格式", "*.hdr;*.dat;*.img;*.raw")
            ]
        )
        if file_path:
            self.hyperspectral_path = file_path
            self.hyper_path_var.set(file_path)

            if file_path.lower().endswith('.hdr'):
                self.dat_params_frame.pack_forget()
                self.hyperspectral_format = 'envi'
            elif file_path.lower().endswith('.dat'):
                self.dat_params_frame.pack(fill=tk.X, pady=(10, 0))
                self.hyperspectral_format = 'dat'
            else:
                messagebox.showwarning("警告", "不支持的文件格式，请选择HDR或DAT文件")
                self.hyperspectral_path = ""
                self.hyper_path_var.set("")

    def select_output_path(self):
        dir_path = filedialog.askdirectory(title="选择输出目录")
        if dir_path:
            self.output_path = dir_path
            self.output_path_var.set(dir_path)

    def load_rgb_image(self):
        if not self.rgb_image_path:
            messagebox.showwarning("警告", "请先选择RGB图像")
            return

        try:
            self.status_var.set("正在加载RGB图像...")
            self.root.update()

            self.rgb_image = Image.open(self.rgb_image_path)
            self.scale = 1.0
            self.update_display_image()

            self.status_var.set(f"RGB图像加载成功: {self.rgb_image.size[0]}x{self.rgb_image.size[1]}像素")
            self.status_bar.config(text=f"RGB图像加载成功: {self.rgb_image.size[0]}x{self.rgb_image.size[1]}像素")
        except Exception as e:
            self.status_var.set(f"加载RGB图像失败: {str(e)}")
            messagebox.showerror("错误", f"加载RGB图像失败: {str(e)}")

    def load_hyperspectral_data(self):
        if not self.hyperspectral_path:
            messagebox.showwarning("警告", "请先选择高光谱数据")
            return

        try:
            self.status_var.set("正在加载高光谱数据...")
            self.root.update()

            threading.Thread(target=self._load_hyperspectral_data_thread, daemon=True).start()
        except Exception as e:
            self.status_var.set(f"加载高光谱数据失败: {str(e)}")
            messagebox.showerror("错误", f"加载高光谱数据失败: {str(e)}")

    def _load_hyperspectral_data_thread(self):
        try:
            if self.hyperspectral_format == 'envi':
                # 读取并保存原始HDR内容
                with open(self.hyperspectral_path, 'r') as f:
                    self.original_hdr_content = f.read()

                # 加载数据
                self.hyperspectral_data = spy.open_image(self.hyperspectral_path)
            elif self.hyperspectral_format == 'dat':
                # 加载DAT文件
                try:
                    width = int(self.dat_width_var.get())
                    height = int(self.dat_height_var.get())
                    bands = int(self.dat_bands_var.get())
                    dtype = self.dat_dtype_var.get()
                    interleave = self.dat_interleave_var.get()
                except ValueError:
                    self.root.after(0, lambda: messagebox.showerror("错误", "请输入有效的DAT文件参数"))
                    return

                self.dat_info = {
                    'width': width,
                    'height': height,
                    'bands': bands,
                    'dtype': dtype,
                    'interleave': interleave
                }

                # 创建虚拟头文件信息
                envi_header = {
                    'file type': 'ENVI Standard',
                    'samples': width,
                    'lines': height,
                    'bands': bands,
                    'data type': self._get_envi_data_type(dtype),
                    'interleave': interleave,
                    'byte order': 0
                }

                # 打开DAT文件
                self.hyperspectral_data = spy.envi.open_memmap(
                    None,
                    envi_header,
                    image=self.hyperspectral_path
                )
            else:
                self.root.after(0, lambda: messagebox.showerror("错误", "不支持的高光谱数据格式"))
                return

            # 更新状态
            self.root.after(0, lambda: self.status_var.set(f"高光谱数据加载成功: {self.hyperspectral_data.shape}"))
            self.root.after(0,
                            lambda: self.status_bar.config(text=f"高光谱数据加载成功: {self.hyperspectral_data.shape}"))

            # 验证尺寸匹配
            if self.rgb_image:
                rgb_width, rgb_height = self.rgb_image.size
                hyper_height, hyper_width, _ = self.hyperspectral_data.shape

                if rgb_width != hyper_width or rgb_height != hyper_height:
                    self.root.after(0, lambda: messagebox.showwarning("警告",
                                                                      f"RGB图像尺寸({rgb_width}x{rgb_height})与高光谱数据尺寸({hyper_width}x{hyper_height})不匹配，"
                                                                      "框选区域可能无法正确对应"))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"加载高光谱数据失败: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"加载高光谱数据失败: {str(e)}"))

    def _get_envi_data_type(self, dtype):
        type_map = {
            'uint8': 1,
            'int16': 2,
            'uint16': 12,
            'int32': 3,
            'uint32': 13,
            'float32': 4,
            'float64': 5
        }
        return type_map.get(dtype, 4)

    def update_display_image(self):
        if self.rgb_image is None:
            return

        # 应用缩放
        new_width = int(self.rgb_image.size[0] * self.scale)
        new_height = int(self.rgb_image.size[1] * self.scale)
        display_image = self.rgb_image.resize((new_width, new_height), Image.LANCZOS)

        # 绘制框选区域
        if self.selection:
            x1, y1, x2, y2 = self.selection
            x1_scaled = x1 * self.scale
            y1_scaled = y1 * self.scale
            x2_scaled = x2 * self.scale
            y2_scaled = y2 * self.scale

            draw = ImageDraw.Draw(display_image)
            draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline="red", width=2)

        # 更新显示
        self.display_photo = ImageTk.PhotoImage(display_image)
        self.rgb_canvas.delete("all")
        self.image_on_canvas = self.rgb_canvas.create_image(0, 0, anchor=tk.NW, image=self.display_photo)

        # 设置画布滚动区域
        self.rgb_canvas.config(scrollregion=self.rgb_canvas.bbox("all"))

        # 更新选择区域信息
        if self.selection:
            x1, y1, x2, y2 = self.selection
            self.selection_var.set(f"X: {min(x1, x2)}-{max(x1, x2)}\nY: {min(y1, y2)}-{max(y1, y2)}\n"
                                   f"宽度: {abs(x2 - x1)}\n高度: {abs(y2 - y1)}")

    def on_canvas_configure(self, event):
        if hasattr(self, 'display_photo') and self.display_photo and self.scale < 1.0:
            canvas_width = event.width
            canvas_height = event.height
            image_width = self.rgb_image.size[0] * self.scale
            image_height = self.rgb_image.size[1] * self.scale

            if image_width < canvas_width and image_height < canvas_height:
                x_pos = (canvas_width - image_width) // 2
                y_pos = (canvas_height - image_height) // 2
                self.rgb_canvas.coords(self.image_on_canvas, x_pos, y_pos)

    def zoom(self, factor):
        if self.rgb_image is None:
            return

        # 保存当前视图位置
        xview = self.rgb_canvas.xview()
        yview = self.rgb_canvas.yview()

        # 保存当前鼠标位置
        x, y = self.rgb_canvas.winfo_pointerx() - self.rgb_canvas.winfo_rootx(), \
               self.rgb_canvas.winfo_pointery() - self.rgb_canvas.winfo_rooty()

        # 应用缩放因子
        self.scale *= factor

        # 更新显示
        self.update_display_image()

        # 恢复视图位置
        new_x = (x / self.scale) * self.scale
        new_y = (y / self.scale) * self.scale

        x_fraction = new_x / (self.rgb_image.size[0] * self.scale)
        y_fraction = new_y / (self.rgb_image.size[1] * self.scale)

        self.rgb_canvas.xview_moveto(x_fraction)
        self.rgb_canvas.yview_moveto(y_fraction)

    def fit_to_window(self):
        if self.rgb_image is None:
            return

        canvas_width = self.rgb_canvas.winfo_width()
        canvas_height = self.rgb_canvas.winfo_height()

        if canvas_width <= 0 or canvas_height <= 0:
            return

        image_width, image_height = self.rgb_image.size

        # 计算适应窗口的缩放比例
        scale_x = canvas_width / image_width
        scale_y = canvas_height / image_height
        self.scale = min(scale_x, scale_y) * 0.95  # 留出一点边距

        # 更新显示
        self.update_display_image()

    def on_canvas_click(self, event):
        if self.rgb_image is None:
            return

        self.is_selecting = True

        # 获取画布上的实际坐标
        x = self.rgb_canvas.canvasx(event.x)
        y = self.rgb_canvas.canvasy(event.y)

        # 转换为原始图像坐标
        self.select_start_x = x / self.scale
        self.select_start_y = y / self.scale

    def on_canvas_drag(self, event):
        if not hasattr(self, 'is_selecting') or not self.is_selecting:
            return

        # 获取画布上的实际坐标
        x = self.rgb_canvas.canvasx(event.x)
        y = self.rgb_canvas.canvasy(event.y)

        # 转换为原始图像坐标
        current_x = x / self.scale
        current_y = y / self.scale

        # 创建选择区域
        x1 = min(self.select_start_x, current_x)
        y1 = min(self.select_start_y, current_y)
        x2 = max(self.select_start_x, current_x)
        y2 = max(self.select_start_y, current_y)

        self.selection = (x1, y1, x2, y2)
        self.update_display_image()

    def on_canvas_release(self, event):
        if hasattr(self, 'is_selecting'):
            self.is_selecting = False

        # 确保选择区域有效
        if self.selection:
            x1, y1, x2, y2 = self.selection
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                self.selection = None
                self.update_display_image()

    def on_mouse_wheel(self, event):
        if self.rgb_image is None:
            return

        # 处理不同平台的滚轮事件
        if event.num == 5 or event.delta < 0:  # 向下滚动
            self.zoom(0.9)
        else:  # 向上滚动
            self.zoom(1.1)

    def cut_hyperspectral_data(self):
        if self.hyperspectral_data is None:
            messagebox.showwarning("警告", "请先加载高光谱数据")
            return

        if self.selection is None:
            messagebox.showwarning("警告", "请先在RGB图像上框选区域")
            return

        if not self.output_path:
            messagebox.showwarning("警告", "请先选择输出路径")
            return

        try:
            self.status_var.set("正在切割高光谱数据...")
            self.root.update()

            # 获取选择区域
            x1, y1, x2, y2 = self.selection

            # 确保坐标顺序正确
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)

            # 转换为整数索引
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)

            # 切割高光谱数据
            threading.Thread(target=self._cut_hyperspectral_data_thread,
                             args=(x_min, y_min, x_max, y_max), daemon=True).start()
        except Exception as e:
            self.status_var.set(f"切割高光谱数据失败: {str(e)}")
            messagebox.showerror("错误", f"切割高光谱数据失败: {str(e)}")

    def _modify_hdr_content(self, original_content, new_samples, new_lines, x_start, y_start):
        """修改头文件时，保留所有原始元数据（如 Wavelength、fwhm 等）"""
        # 仅更新 samples/lines/x start/y start，其他字段原样保留
        lines = original_content.split('\n')
        modified = []
        for line in lines:
            if line.lower().startswith('samples ='):
                modified.append(f"samples = {new_samples}")
            elif line.lower().startswith('lines ='):
                modified.append(f"lines = {new_lines}")
            elif line.lower().startswith('x start ='):
                modified.append(f"x start = {x_start}")
            elif line.lower().startswith('y start ='):
                modified.append(f"y start = {y_start}")
            else:
                modified.append(line)
        return '\n'.join(modified)

    def _cut_hyperspectral_data_thread(self, x_min, y_min, x_max, y_max):
        try:
            # 切割高光谱数据
            cut_data = self.hyperspectral_data[y_min:y_max, x_min:x_max, :]

            # 构建输出文件名
            base_name = os.path.splitext(os.path.basename(self.hyperspectral_path))[0]
            output_hdr = os.path.join(self.output_path, f"{base_name}_cut_{x_min}_{y_min}_{x_max}_{y_max}.hdr")
            output_img = os.path.join(self.output_path, f"{base_name}_cut_{x_min}_{y_min}_{x_max}_{y_max}.dat")

            # 保存切割后的数据
            if self.hyperspectral_format == 'envi':
                # 对于ENVI格式，保留原始HDR格式
                new_samples = x_max - x_min
                new_lines = y_max - y_min

                # 修改原始HDR内容
                modified_hdr = self._modify_hdr_content(
                    self.original_hdr_content,
                    new_samples,
                    new_lines,
                    x_min,
                    y_min
                )

                # 保存修改后的HDR文件
                with open(output_hdr, 'w') as f:
                    f.write(modified_hdr)

                # 保存图像数据
                # 确保数据格式与原始一致
                if self.hyperspectral_data.metadata['interleave'] != 'bsq':
                    cut_data_bsq = self._convert_to_bsq(cut_data, self.hyperspectral_data.metadata['interleave'])
                else:
                    cut_data_bsq = cut_data
                cut_data_bsq.tofile(output_img)

            elif self.hyperspectral_format == 'dat':
                # 使用DAT参数保存
                envi_header = {
                    'file type': 'ENVI Standard',
                    'samples': cut_data.shape[1],
                    'lines': cut_data.shape[0],
                    'bands': cut_data.shape[2],
                    'data type': self._get_envi_data_type(self.dat_info['dtype']),
                    'interleave': self.dat_info['interleave'],
                    'byte order': 0
                }

                # 保存头文件和数据
                spy.envi.write_envi_header(output_hdr, envi_header)

                # 确保数据格式与原始一致
                if self.dat_info['interleave'] != 'bsq':
                    cut_data_bsq = self._convert_to_bsq(cut_data, self.dat_info['interleave'])
                else:
                    cut_data_bsq = cut_data
                cut_data_bsq.tofile(output_img)

            # 更新状态
            self.root.after(0, lambda: self.status_var.set(f"高光谱数据切割成功，已保存至: {output_hdr}"))
            self.root.after(0, lambda: self.status_bar.config(text=f"高光谱数据切割成功，已保存至: {output_hdr}"))

            # 显示切割结果
            self.root.after(0, lambda: self._show_cut_result(cut_data))

            message = f"高光谱数据切割成功！\n\n" \
                      f"区域大小: {x_max - x_min} x {y_max - y_min}\n" \
                      f"保存路径: {output_hdr}"
            self.root.after(0, lambda: messagebox.showinfo("成功", message))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"切割高光谱数据失败: {str(e)}"))
            self.root.after(0, lambda: messagebox.showerror("错误", f"切割高光谱数据失败: {str(e)}"))

    def _convert_to_bsq(self, data, original_interleave):
        """将数据转换为BSQ格式"""
        if original_interleave == 'bsq':
            return data

        lines, samples, bands = data.shape
        bsq_data = np.zeros((lines, samples, bands), dtype=data.dtype)

        if original_interleave == 'bip':
            # BIP -> BSQ
            for b in range(bands):
                bsq_data[:, :, b] = data[:, :, b]
        elif original_interleave == 'bil':
            # BIL -> BSQ
            for b in range(bands):
                bsq_data[:, :, b] = data[:, :, b]

        return bsq_data

    def _show_cut_result(self, cut_data):
        """显示切割结果"""
        try:
            # 创建RGB合成图像
            if self.hyperspectral_format == 'envi' and self.original_hdr_content:
                # 从原始HDR中提取默认波段
                default_bands = None
                for line in self.original_hdr_content.split('\n'):
                    if line.lower().startswith('default bands ='):
                        match = re.search(r'\{.*\}', line)
                        if match:
                            bands_str = match.group()
                            bands_str = bands_str.replace('{', '').replace('}', '').replace(' ', '')
                            default_bands = [int(b) - 1 for b in bands_str.split(',')]
                        break

            if not default_bands or len(default_bands) < 3:
                default_bands = [29, 19, 9]  # 默认波段

            # 确保波段索引在有效范围内
            num_bands = cut_data.shape[2]
            default_bands = [min(b, num_bands - 1) for b in default_bands]

            # 生成RGB预览图
            rgb = spy.get_rgb(cut_data, default_bands)

            # 转换为PIL图像
            rgb = (rgb * 255).astype(np.uint8)
            rgb_image = Image.fromarray(rgb)

            # 显示在结果选项卡中
            result_photo = ImageTk.PhotoImage(rgb_image)
            self.result_canvas.delete("all")
            self.result_canvas.create_image(0, 0, anchor=tk.NW, image=result_photo)
            self.result_photo = result_photo  # 保存引用防止被垃圾回收

            # 切换到结果选项卡
            self.notebook.select(self.result_tab)
        except Exception as e:
            messagebox.showerror("错误", f"显示切割结果失败: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = HyperspectralCutterApp(root)
    root.mainloop()
