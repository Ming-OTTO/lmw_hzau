import os
import shutil
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading


class ImageSplitterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("图片文件夹顺序分割工具")
        self.root.geometry("600x400")
        self.root.resizable(True, True)

        # 设置中文字体支持
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("SimHei", 10))
        self.style.configure("TButton", font=("SimHei", 10))
        self.style.configure("TEntry", font=("SimHei", 10))

        # 源目录和目标目录变量
        self.source_dir = tk.StringVar()
        self.dest_base = tk.StringVar()

        # 创建UI
        self.create_widgets()

        # 进度变量
        self.total_files = 0
        self.processed_files = 0

    def create_widgets(self):
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 源目录选择
        ttk.Label(main_frame, text="源文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.source_dir, width=50).grid(row=0, column=1, pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_source).grid(row=0, column=2, padx=5, pady=5)

        # 目标目录选择
        ttk.Label(main_frame, text="目标文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.dest_base, width=50).grid(row=1, column=1, pady=5)
        ttk.Button(main_frame, text="浏览...", command=self.browse_dest).grid(row=1, column=2, padx=5, pady=5)

        # 说明标签 - 新增内容，说明分割方式
        note_label = ttk.Label(
            main_frame,
            text="注意：图片将按原始顺序分割，前三分之一到part1，中间三分之一到part2，最后三分之一到part3",
            foreground="blue"
        )
        note_label.grid(row=2, column=0, columnspan=3, sticky=tk.W + tk.E, pady=10)

        # 进度条
        ttk.Label(main_frame, text="进度:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, length=400)
        self.progress_bar.grid(row=3, column=1, pady=5)

        # 状态标签
        self.status_label = ttk.Label(main_frame, text="等待开始...")
        self.status_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)

        # 日志区域
        log_frame = ttk.LabelFrame(main_frame, text="日志", padding="10")
        log_frame.grid(row=5, column=0, columnspan=3, sticky=tk.W + tk.E + tk.N + tk.S, pady=10)

        # 添加滚动条
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(log_frame, height=8, width=60, yscrollcommand=scrollbar.set, state=tk.DISABLED)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.log_text.yview)

        # 按钮区域
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=3, pady=10)

        ttk.Button(button_frame, text="开始分割", command=self.start_split).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="退出", command=self.root.quit).pack(side=tk.LEFT, padx=10)

        # 设置网格权重，使控件可以随窗口大小调整
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(5, weight=1)

    def browse_source(self):
        directory = filedialog.askdirectory(title="选择源文件夹")
        if directory:
            self.source_dir.set(directory)

    def browse_dest(self):
        directory = filedialog.askdirectory(title="选择目标文件夹")
        if directory:
            self.dest_base.set(directory)

    def log(self, message):
        """向日志区域添加消息"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)  # 滚动到最后一行
        self.log_text.config(state=tk.DISABLED)

    def update_progress(self):
        """更新进度条"""
        if self.total_files > 0:
            progress = (self.processed_files / self.total_files) * 100
            self.progress_var.set(progress)
            self.status_label.config(text=f"已处理 {self.processed_files}/{self.total_files} 个文件")

    def copy_files(self, file_list, dest_dir):
        """复制文件到目标目录并保持结构"""
        for file_path in file_list:
            # 获取相对路径，用于保持目录结构
            relative_path = os.path.relpath(file_path, self.source_dir.get())
            dest_path = os.path.join(dest_dir, relative_path)

            # 创建目标目录
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # 复制文件
            shutil.copy2(file_path, dest_path)
            self.log(f"复制: {os.path.basename(file_path)} -> {dest_dir.split(os.sep)[-1]}")

            # 更新进度
            self.processed_files += 1
            self.root.after(10, self.update_progress)

    def split_task(self):
        """分割任务的主要逻辑，将在后台线程中运行"""
        source_dir = self.source_dir.get()
        dest_base = self.dest_base.get()

        # 验证输入
        if not source_dir or not os.path.isdir(source_dir):
            self.root.after(0, lambda: messagebox.showerror("错误", "请选择有效的源文件夹"))
            return

        if not dest_base or not os.path.isdir(dest_base):
            self.root.after(0, lambda: messagebox.showerror("错误", "请选择有效的目标文件夹"))
            return

        # 定义图片文件扩展名
        image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')

        # 创建三个目标文件夹
        dest_dirs = [
            os.path.join(dest_base, 'part1'),
            os.path.join(dest_base, 'part2'),
            os.path.join(dest_base, 'part3')
        ]

        for dir_path in dest_dirs:
            os.makedirs(dir_path, exist_ok=True)

        # 收集所有图片文件路径（按顺序）
        image_files = []
        for root, _, files in os.walk(source_dir):
            # 对每个目录中的文件按名称排序
            sorted_files = sorted(files)
            for file in sorted_files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))

        if not image_files:
            self.root.after(0, lambda: messagebox.showinfo("信息", "未找到任何图片文件"))
            return

        # 更新总文件数
        self.total_files = len(image_files)
        self.processed_files = 0
        self.root.after(0, self.update_progress)

        # 计算分割点（均匀分成三份）
        total = len(image_files)
        split_point1 = total // 3
        split_point2 = 2 * total // 3

        # 按顺序分割文件列表（不随机）
        part1_files = image_files[:split_point1]
        part2_files = image_files[split_point1:split_point2]
        part3_files = image_files[split_point2:]

        # 复制文件
        self.root.after(0, lambda: self.log(f"\n开始复制第一部分 ({len(part1_files)} 个文件)"))
        self.copy_files(part1_files, dest_dirs[0])

        self.root.after(0, lambda: self.log(f"\n开始复制第二部分 ({len(part2_files)} 个文件)"))
        self.copy_files(part2_files, dest_dirs[1])

        self.root.after(0, lambda: self.log(f"\n开始复制第三部分 ({len(part3_files)} 个文件)"))
        self.copy_files(part3_files, dest_dirs[2])

        # 完成
        result_msg = (f"分割完成！总共处理了 {total} 个图片文件\n"
                      f"第一部分: {len(part1_files)} 个文件\n"
                      f"第二部分: {len(part2_files)} 个文件\n"
                      f"第三部分: {len(part3_files)} 个文件")

        self.root.after(0, lambda: self.log(f"\n{result_msg}"))
        self.root.after(0, lambda: messagebox.showinfo("完成", result_msg))

    def start_split(self):
        """开始分割任务，在新线程中运行以避免界面冻结"""
        # 创建并启动后台线程
        split_thread = threading.Thread(target=self.split_task)
        split_thread.daemon = True  # 线程会随主线程退出而退出
        split_thread.start()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSplitterApp(root)
    root.mainloop()
