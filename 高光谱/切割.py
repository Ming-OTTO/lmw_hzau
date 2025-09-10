import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import spectral.io.envi as envi
import os
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import threading
import time

# 配置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class HyperspectralCutter:
    """高光谱数据切割工具，基于RGB图像框选"""

    def __init__(self):
        """初始化工具参数"""
        # 路径变量
        self.rgb_path = None
        self.hyperspectral_path = None
        self.output_dir = None

        # 数据变量
        self.rgb_image = None
        self.hyperspectral_data = None
        self.metadata = None
        self.coordinates = []

        # GUI组件
        self.root = None
        self.fig = None
        self.ax = None
        self.canvas = None
        self.toolbar_frame = None  # 用于放置工具栏的框架
        self.toolbar = None
        self.rs = None
        self.progress_bar = None

        # 状态变量
        self.is_loading = False
        self.status_var = None
        self.progress_var = None
        self.rgb_path_var = None
        self.hyperspectral_path_var = None
        self.output_dir_var = None

    def select_rgb_path(self):
        """选择参考RGB图像"""
        if self.is_loading:
            messagebox.showinfo("提示", "正在处理数据，请等待...")
            return

        self.rgb_path = filedialog.askopenfilename(
            title="选择参考RGB图像",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")]
        )

        if self.rgb_path:
            rgb_name = os.path.basename(self.rgb_path)
            self.rgb_path_var.set(rgb_name)
            self.status_var.set(f"已选择参考RGB图像: {rgb_name}")

    def select_hyperspectral_path(self):
        """选择高光谱数据文件"""
        if self.is_loading:
            messagebox.showinfo("提示", "正在处理数据，请等待...")
            return

        self.hyperspectral_path = filedialog.askopenfilename(
            title="选择高光谱数据",
            filetypes=[("ENVI头文件", "*.hdr"), ("所有文件", "*.*")]
        )

        if self.hyperspectral_path:
            hs_name = os.path.basename(self.hyperspectral_path)
            self.hyperspectral_path_var.set(hs_name)
            self.status_var.set(f"已选择高光谱数据: {hs_name}")

    def select_output_dir(self):
        """选择输出目录"""
        if self.is_loading:
            messagebox.showinfo("提示", "正在处理数据，请等待...")
            return

        output_dir = filedialog.askdirectory(title="选择输出目录")
        if output_dir:
            self.output_dir = output_dir
            dir_name = os.path.basename(output_dir)
            self.output_dir_var.set(dir_name)
            self.status_var.set(f"已选择输出目录: {dir_name}")

    def load_data_thread(self):
        """在线程中加载数据，避免界面卡顿"""
        try:
            self.is_loading = True
            self.progress_var.set(0)
            self.status_var.set("开始加载数据...")

            # 验证输入
            if not self.rgb_path:
                raise ValueError("请先选择参考RGB图像")
            if not self.hyperspectral_path:
                raise ValueError("请先选择高光谱数据")

            # 加载RGB图像
            self.status_var.set("正在加载RGB图像...")
            self.rgb_image = np.array(Image.open(self.rgb_path))
            self.progress_var.set(20)
            self.status_var.set(f"RGB图像加载完成，尺寸: {self.rgb_image.shape}")

            # 加载高光谱数据
            self.status_var.set("正在加载高光谱数据...")
            if self.hyperspectral_path.lower().endswith('.hdr'):
                img = envi.open(self.hyperspectral_path)
            else:
                base_path = os.path.splitext(self.hyperspectral_path)[0]
                hdr_path = base_path + '.hdr'
                if os.path.exists(hdr_path):
                    img = envi.open(hdr_path, self.hyperspectral_path)
                else:
                    raise FileNotFoundError(f"找不到对应的头文件: {hdr_path}")

            self.metadata = img.metadata
            self.status_var.set("正在读取高光谱数据内容...")

            # 处理大型数据（分块加载）
            lines = int(self.metadata.get('lines', 0))
            samples = int(self.metadata.get('samples', 0))
            bands = int(self.metadata.get('bands', 0))

            if lines * samples * bands > 1e8:  # 数据量超过1亿，分块加载
                self.hyperspectral_data = np.empty((lines, samples, bands), dtype=np.float32)
                block_size = 200  # 每次加载200行
                total_blocks = (lines + block_size - 1) // block_size

                for i in range(total_blocks):
                    start = i * block_size
                    end = min((i + 1) * block_size, lines)
                    self.hyperspectral_data[start:end] = img.read_subregion((start, end), (0, samples))

                    progress = 20 + (i + 1) / total_blocks * 70
                    self.progress_var.set(progress)
                    self.status_var.set(f"正在加载高光谱数据: 块 {i + 1}/{total_blocks}")
                    time.sleep(0.05)
            else:
                # 小型数据直接加载
                self.hyperspectral_data = img.load()
                self.progress_var.set(90)

            # 验证尺寸匹配性
            rgb_h, rgb_w = self.rgb_image.shape[:2]
            if (lines, samples) != (rgb_h, rgb_w):
                self.status_var.set(f"警告: RGB与高光谱尺寸不匹配 (RGB: {rgb_h}x{rgb_w}, 高光谱: {lines}x{samples})")
                messagebox.showwarning("尺寸警告",
                                       f"RGB图像与高光谱数据尺寸不匹配!\n"
                                       f"RGB: {rgb_h}x{rgb_w}\n"
                                       f"高光谱: {lines}x{samples}\n"
                                       "切割仍可进行，但结果可能不准确")

            self.progress_var.set(100)
            self.status_var.set("数据加载完成，可显示图像进行框选")
            messagebox.showinfo("成功", "数据加载完成，请点击'显示RGB图像'进行框选")

        except Exception as e:
            messagebox.showerror("错误", f"加载数据失败: {str(e)}")
            self.status_var.set(f"加载失败: {str(e)}")
        finally:
            self.is_loading = False
            self.progress_bar.grid_remove()

    def load_data(self):
        """启动数据加载线程"""
        if self.is_loading:
            return

        # 检查是否已选择必要文件
        if not self.rgb_path:
            messagebox.showerror("错误", "请先选择参考RGB图像")
            return
        if not self.hyperspectral_path:
            messagebox.showerror("错误", "请先选择高光谱数据")
            return

        # 显示进度条
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        # 启动加载线程
        load_thread = threading.Thread(target=self.load_data_thread)
        load_thread.daemon = True
        load_thread.start()

    def onselect(self, eclick, erelease):
        """处理框选区域"""
        if eclick.xdata is None or erelease.xdata is None:
            return

        # 计算选择区域坐标
        x1 = int(min(eclick.xdata, erelease.xdata))
        y1 = int(min(eclick.ydata, erelease.ydata))
        x2 = int(max(eclick.xdata, erelease.xdata))
        y2 = int(max(eclick.ydata, erelease.ydata))

        # 限制在图像范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(self.rgb_image.shape[1] - 1, x2)
        y2 = min(self.rgb_image.shape[0] - 1, y2)

        self.coordinates = [(x1, y1), (x2, y2)]
        self.status_var.set(f"已选择区域: 左上角({x1},{y1}) - 右下角({x2},{y2})，大小: {x2 - x1}x{y2 - y1}")

        # 绘制选择框
        for patch in self.ax.patches:
            patch.remove()
        self.ax.add_patch(plt.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor='red', linewidth=2
        ))
        self.canvas.draw_idle()

    def setup_selector(self):
        """设置矩形选择器"""
        if self.rs:
            self.rs.disconnect_events()

        try:
            # 现代API
            self.rs = RectangleSelector(
                self.ax, self.onselect,
                useblit=True,
                button=[1],  # 左键选择
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=dict(edgecolor='red', linewidth=2)
            )
        except TypeError:
            # 兼容旧版本API
            self.rs = RectangleSelector(
                self.ax, self.onselect,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True
            )
            self.rs.to_draw.set_edgecolor('red')
            self.rs.to_draw.set_linewidth(2)

    def show_rgb_image(self):
        """显示RGB图像并启用缩放和框选功能"""
        if self.is_loading:
            messagebox.showinfo("提示", "正在处理数据，请等待...")
            return

        if self.rgb_image is None:
            messagebox.showerror("错误", "请先加载数据")
            return

        # 清除之前的图像和工具栏
        if self.fig:
            plt.close(self.fig)
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if self.toolbar_frame:
            self.toolbar_frame.destroy()

        # 创建新的工具栏框架
        self.toolbar_frame = tk.Frame(self.root)
        self.toolbar_frame.grid(row=7, column=0, columnspan=4, sticky="ew")

        # 创建新图像
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.rgb_image)
        self.ax.set_title("参考RGB图像 - 拖动鼠标框选区域（支持滚轮缩放和平移）")
        self.ax.axis('on')

        # 创建画布
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=6, column=0, columnspan=4,
                                         sticky="nsew", padx=10, pady=5)

        # 添加工具栏到独立框架
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.update()

        # 设置选择器
        self.setup_selector()

        # 调整布局权重
        self.root.grid_rowconfigure(6, weight=1)
        for col in range(4):
            self.root.grid_columnconfigure(col, weight=1)

    def crop_hyperspectral(self):
        """切割高光谱数据"""
        if not self.coordinates:
            messagebox.showinfo("提示", "请先在RGB图像上框选区域")
            return None, None

        (x1, y1), (x2, y2) = self.coordinates
        self.status_var.set(f"正在切割区域: {x2 - x1}x{y2 - y1}")

        try:
            cropped_data = self.hyperspectral_data[y1:y2, x1:x2, :]
            cropped_rgb = self.rgb_image[y1:y2, x1:x2, :]

            self.status_var.set(f"切割完成: 高光谱数据尺寸 {cropped_data.shape}")
            return cropped_data, cropped_rgb
        except Exception as e:
            messagebox.showerror("错误", f"切割失败: {str(e)}")
            self.status_var.set(f"切割失败: {str(e)}")
            return None, None

    def preview_crop(self):
        """预览切割结果"""
        cropped_data, cropped_rgb = self.crop_hyperspectral()
        if cropped_rgb is None:
            return

        # 创建预览窗口
        preview_win = tk.Toplevel(self.root)
        preview_win.title("切割区域预览")
        preview_win.geometry("600x500")
        preview_win.grid_rowconfigure(0, weight=1)
        preview_win.grid_columnconfigure(0, weight=1)

        # 预览图像框架
        preview_toolbar_frame = tk.Frame(preview_win)
        preview_toolbar_frame.grid(row=1, column=0, sticky="ew")

        # 显示预览图像
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.imshow(cropped_rgb)
        ax.set_title(f"切割区域预览: {cropped_rgb.shape[1]}x{cropped_rgb.shape[0]}")
        ax.axis('on')

        canvas = FigureCanvasTkAgg(fig, master=preview_win)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

        # 添加预览工具栏
        toolbar = NavigationToolbar2Tk(canvas, preview_toolbar_frame)
        toolbar.update()

        # 关闭按钮
        tk.Button(preview_win, text="关闭", command=preview_win.destroy).grid(
            row=2, column=0, pady=10)

    def save_cropped_data(self):
        """保存切割后的高光谱数据"""
        cropped_data, _ = self.crop_hyperspectral()
        if cropped_data is None:
            return

        if not self.output_dir:
            self.output_dir = os.path.dirname(self.hyperspectral_path)
            self.status_var.set(f"使用默认输出目录: {self.output_dir}")

        try:
            # 生成输出文件名
            base_name = os.path.splitext(os.path.basename(self.hyperspectral_path))[0]
            output_name = f"cropped_{base_name}"
            output_path = os.path.join(self.output_dir, output_name)

            # 更新元数据
            new_metadata = self.metadata.copy()
            new_metadata['lines'] = cropped_data.shape[0]
            new_metadata['samples'] = cropped_data.shape[1]

            # 保存ENVI格式
            envi.save_image(f"{output_path}.hdr", cropped_data,
                            metadata=new_metadata, force=True)

            self.status_var.set(f"已保存切割数据到: {output_path}.hdr")
            messagebox.showinfo("成功", f"切割数据已保存至:\n{output_path}.hdr")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
            self.status_var.set(f"保存失败: {str(e)}")

    def create_gui(self):
        """创建图形用户界面"""
        self.root = tk.Tk()
        self.root.title("RGB参考高光谱切割工具")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)

        # 创建变量
        self.rgb_path_var = tk.StringVar(value="未选择")
        self.hyperspectral_path_var = tk.StringVar(value="未选择")
        self.output_dir_var = tk.StringVar(value="未选择")
        self.status_var = tk.StringVar(value="就绪 - 请先选择RGB图像和高光谱数据")
        self.progress_var = tk.DoubleVar(value=0)

        # ===== 路径选择区域 =====
        path_frame = tk.LabelFrame(self.root, text="文件选择", padx=10, pady=10)
        path_frame.grid(row=0, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        # RGB图像选择
        tk.Button(path_frame, text="选择参考RGB图像", command=self.select_rgb_path).grid(
            row=0, column=0, padx=5, pady=5, sticky="w")
        tk.Label(path_frame, textvariable=self.rgb_path_var, wraplength=400).grid(
            row=0, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # 高光谱数据选择
        tk.Button(path_frame, text="选择高光谱数据", command=self.select_hyperspectral_path).grid(
            row=1, column=0, padx=5, pady=5, sticky="w")
        tk.Label(path_frame, textvariable=self.hyperspectral_path_var, wraplength=400).grid(
            row=1, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        # 输出目录选择
        tk.Button(path_frame, text="选择输出目录", command=self.select_output_dir).grid(
            row=2, column=0, padx=5, pady=5, sticky="w")
        tk.Label(path_frame, textvariable=self.output_dir_var, wraplength=400).grid(
            row=2, column=1, columnspan=3, padx=5, pady=5, sticky="ew")

        path_frame.grid_columnconfigure(1, weight=1)

        # ===== 操作按钮区域 =====
        btn_frame = tk.Frame(self.root, padx=10, pady=10)
        btn_frame.grid(row=1, column=0, columnspan=4, sticky="ew")

        tk.Button(btn_frame, text="加载数据", command=self.load_data, width=12).grid(
            row=0, column=0, padx=5, pady=5)
        tk.Button(btn_frame, text="显示RGB图像", command=self.show_rgb_image, width=12).grid(
            row=0, column=1, padx=5, pady=5)
        tk.Button(btn_frame, text="预览切割结果", command=self.preview_crop, width=12).grid(
            row=0, column=2, padx=5, pady=5)
        tk.Button(btn_frame, text="保存切割结果", command=self.save_cropped_data, width=12).grid(
            row=0, column=3, padx=5, pady=5)

        # ===== 进度条和状态 =====
        self.progress_bar = ttk.Progressbar(self.root, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=4, sticky="ew", padx=10, pady=5)
        self.progress_bar.grid_remove()  # 初始隐藏

        status_frame = tk.Frame(self.root, padx=10, pady=5)
        status_frame.grid(row=3, column=0, columnspan=4, sticky="ew")
        tk.Label(status_frame, text="状态: ", fg="blue").grid(row=0, column=0, sticky="w")
        tk.Label(status_frame, textvariable=self.status_var, fg="blue").grid(
            row=0, column=1, sticky="ew")
        status_frame.grid_columnconfigure(1, weight=1)

        # ===== 使用说明 =====
        help_frame = tk.LabelFrame(self.root, text="使用说明", padx=10, pady=10)
        help_frame.grid(row=4, column=0, columnspan=4, sticky="ew", padx=10, pady=5)

        instructions = """
        1. 选择参考RGB图像（必须）
        2. 选择高光谱数据（.hdr文件）
        3. 选择输出目录
        4. 点击"加载数据"按钮
        5. 点击"显示RGB图像"按钮
        6. 在图像上操作：
           - 鼠标滚轮：放大/缩小图像
           - 按下鼠标中键拖动：平移图像
           - 左键拖动：框选感兴趣区域
        7. 点击"预览切割结果"查看效果
        8. 点击"保存切割结果"保存高光谱数据
        """
        tk.Label(help_frame, text=instructions, justify=tk.LEFT).grid(
            row=0, column=0, sticky="w")

        # 设置网格权重
        self.root.grid_rowconfigure(6, weight=1)
        for i in range(4):
            self.root.grid_columnconfigure(i, weight=1)

    def run(self):
        """运行应用程序"""
        self.create_gui()
        self.root.mainloop()


if __name__ == "__main__":
    cutter = HyperspectralCutter()
    cutter.run()