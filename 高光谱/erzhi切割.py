import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
from typing import Tuple, Optional


class LongImageCutter:
    def __init__(self):
        self.points = []
        self.image = None
        self.image_path = None
        self.output_path = None
        self.original_img = None
        self.display_img = None
        self.scale_factor = 1.0
        self.max_display_height = 1000

    def load_image(self, image_path: str) -> bool:
        """加载图像"""
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            return False
        self.original_img = self.image.copy()
        self._resize_display_image()
        return True

    def _resize_display_image(self):
        """调整显示图像大小以便在界面中显示"""
        if self.image is None:
            return

        height, width = self.image.shape[:2]
        if height > self.max_display_height:
            self.scale_factor = self.max_display_height / height
            new_width = int(width * self.scale_factor)
            self.display_img = cv2.resize(self.image, (new_width, self.max_display_height))
        else:
            self.scale_factor = 1.0
            self.display_img = self.image.copy()

    def get_scaled_points(self) -> list:
        """获取缩放后的点坐标"""
        return [(int(x / self.scale_factor), int(y / self.scale_factor)) for x, y in self.points]

    def crop_image(self) -> np.ndarray:
        """根据选择的点裁剪图像"""
        if len(self.points) != 2:
            return None

        scaled_points = self.get_scaled_points()
        y_start = min(scaled_points[0][1], scaled_points[1][1])
        y_end = max(scaled_points[0][1], scaled_points[1][1])

        # 确保坐标在有效范围内
        height, width = self.image.shape[:2]
        y_start = max(0, y_start)
        y_end = min(height, y_end)

        if y_start >= y_end:
            return None

        return self.image[y_start:y_end, :]

    def save_cropped_image(self, cropped_image: np.ndarray, output_path: str) -> bool:
        """保存裁剪后的图像"""
        if cropped_image is None:
            return False

        try:
            cv2.imwrite(output_path, cropped_image)
            return True
        except Exception as e:
            print(f"保存图像时出错: {e}")
            return False


class LongImageCutterGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("长图切割工具")
        self.root.geometry("1000x700")

        self.cutter = LongImageCutter()
        self.photo = None
        self.drawing = False
        self.point1 = None
        self.point2 = None

        self._create_widgets()

    def _create_widgets(self):
        """创建界面控件"""
        # 顶部框架 - 文件选择
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="输入图像:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.input_path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.input_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(top_frame, text="浏览...", command=self._browse_input).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(top_frame, text="输出图像:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.output_path_var = tk.StringVar()
        ttk.Entry(top_frame, textvariable=self.output_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(top_frame, text="浏览...", command=self._browse_output).grid(row=1, column=2, padx=5, pady=5)

        ttk.Button(top_frame, text="加载图像", command=self._load_image).grid(row=0, column=3, padx=10, pady=5)
        ttk.Button(top_frame, text="重置选择", command=self._reset_selection).grid(row=1, column=3, padx=10, pady=5)

        # 图像显示区域
        self.canvas_frame = ttk.Frame(self.root)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.canvas = tk.Canvas(self.canvas_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # 底部框架 - 操作按钮
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="请选择并加载图像")
        ttk.Label(bottom_frame, textvariable=self.status_var).pack(side=tk.LEFT, padx=5)

        ttk.Button(bottom_frame, text="裁剪并保存", command=self._crop_and_save).pack(side=tk.RIGHT, padx=5)

    def _browse_input(self):
        """浏览并选择输入图像"""
        filename = filedialog.askopenfilename(
            title="选择图像",
            filetypes=[("图像文件", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
        )
        if filename:
            self.input_path_var.set(filename)
            # 自动生成输出文件名
            base_name = os.path.basename(filename)
            name, ext = os.path.splitext(base_name)
            output_dir = os.path.dirname(filename)
            self.output_path_var.set(os.path.join(output_dir, f"{name}_cropped{ext}"))

    def _browse_output(self):
        """浏览并选择输出图像路径"""
        if not self.input_path_var.get():
            messagebox.showwarning("警告", "请先选择输入图像")
            return

        initial_dir = os.path.dirname(self.input_path_var.get()) if self.input_path_var.get() else ""
        initial_file = os.path.basename(self.input_path_var.get())
        if initial_file:
            name, ext = os.path.splitext(initial_file)
            initial_file = f"{name}_cropped{ext}"

        filename = filedialog.asksaveasfilename(
            title="保存图像",
            initialdir=initial_dir,
            initialfile=initial_file,
            defaultextension=".jpg",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("BMP", "*.bmp"), ("TIFF", "*.tif")]
        )
        if filename:
            self.output_path_var.set(filename)

    def _load_image(self):
        """加载并显示图像"""
        input_path = self.input_path_var.get()
        if not input_path:
            messagebox.showwarning("警告", "请选择输入图像")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("错误", "图像文件不存在")
            return

        # 在单独线程中加载图像，防止界面卡顿
        self.status_var.set("正在加载图像...")
        self.root.update()

        threading.Thread(target=self._load_image_thread, daemon=True).start()

    def _load_image_thread(self):
        """在单独线程中加载图像"""
        try:
            if self.cutter.load_image(self.input_path_var.get()):
                self._display_image()
                self.status_var.set("图像加载成功，请在图像上点击两次选择裁剪区域")
                self._reset_selection()
            else:
                self.status_var.set("加载图像失败")
                messagebox.showerror("错误", "无法加载图像，请检查文件格式和路径")
        except Exception as e:
            self.status_var.set(f"加载图像时出错: {str(e)}")
            messagebox.showerror("错误", f"加载图像时出错: {str(e)}")

    def _display_image(self):
        """在Canvas上显示图像"""
        if self.cutter.display_img is None:
            return

        # 转换为RGB并创建PhotoImage对象
        img_rgb = cv2.cvtColor(self.cutter.display_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.photo = ImageTk.PhotoImage(image=img_pil)

        # 显示图像
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

        # 绘制已选择的点
        if len(self.cutter.points) >= 1:
            x1, y1 = self.cutter.points[0]
            self.canvas.create_oval(x1 - 5, y1 - 5, x1 + 5, y1 + 5, fill="red")

        if len(self.cutter.points) >= 2:
            x2, y2 = self.cutter.points[1]
            self.canvas.create_oval(x2 - 5, y2 - 5, x2 + 5, y2 + 5, fill="red")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2)

    def _on_canvas_click(self, event):
        """处理Canvas上的鼠标点击事件"""
        if self.cutter.display_img is None:
            return

        x, y = event.x, event.y

        # 添加点并更新显示
        self.cutter.points.append((x, y))
        if len(self.cutter.points) > 2:
            self.cutter.points = self.cutter.points[-2:]

        self._display_image()

        # 更新状态
        if len(self.cutter.points) == 1:
            self.status_var.set("已选择第一个点，请选择第二个点")
        elif len(self.cutter.points) == 2:
            scaled_points = self.cutter.get_scaled_points()
            y_start = min(scaled_points[0][1], scaled_points[1][1])
            y_end = max(scaled_points[0][1], scaled_points[1][1])
            height = y_end - y_start
            self.status_var.set(f"已选择裁剪区域: y={y_start} 到 y={y_end}, 高度={height}像素")

    def _reset_selection(self):
        """重置选择的点"""
        self.cutter.points = []
        if self.cutter.display_img is not None:
            self._display_image()
        self.status_var.set("请在图像上点击两次选择裁剪区域")

    def _crop_and_save(self):
        """裁剪并保存图像"""
        if self.cutter.image is None:
            messagebox.showwarning("警告", "请先加载图像")
            return

        if len(self.cutter.points) != 2:
            messagebox.showwarning("警告", "请选择两个点来定义裁剪区域")
            return

        output_path = self.output_path_var.get()
        if not output_path:
            messagebox.showwarning("警告", "请选择输出路径")
            return

        # 在单独线程中执行裁剪和保存，防止界面卡顿
        self.status_var.set("正在裁剪并保存图像...")
        threading.Thread(target=self._crop_and_save_thread, daemon=True).start()

    def _crop_and_save_thread(self):
        """在单独线程中执行裁剪和保存"""
        try:
            cropped_image = self.cutter.crop_image()
            if cropped_image is None:
                self.status_var.set("裁剪失败，请检查选择区域")
                messagebox.showerror("错误", "裁剪失败，请检查选择区域")
                return

            output_path = self.output_path_var.get()  # 获取输出路径
            if self.cutter.save_cropped_image(cropped_image, output_path):
                self.status_var.set(f"图像已成功保存到: {output_path}")
                messagebox.showinfo("成功", f"图像已成功保存到:\n{output_path}")
            else:
                self.status_var.set("保存图像失败")
                messagebox.showerror("错误", "保存图像失败")
        except Exception as e:
            self.status_var.set(f"裁剪或保存图像时出错: {str(e)}")
            messagebox.showerror("错误", f"裁剪或保存图像时出错: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LongImageCutterGUI(root)
    root.mainloop()