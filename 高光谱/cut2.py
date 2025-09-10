import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QLineEdit, QGroupBox,
                             QScrollArea, QSizePolicy, QGraphicsView, QGraphicsScene)
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
import cv2


class HyperspectralImageCutter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("高光谱图像处理工具")
        self.setGeometry(100, 100, 1200, 800)
        self.initUI()
        self.rgb_image = None
        self.hyperspectral_data = None
        self.roi = None
        self.view_scale = 1.0

    def initUI(self):
        # 创建主窗口部件
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        # 标题标签
        title_label = QLabel("高光谱图像处理工具")
        title_label.setStyleSheet("font-size: 20pt; font-weight: bold; color: #336699; margin-bottom: 20px;")
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # 输入输出组
        io_group = QGroupBox("输入/输出设置")
        io_layout = QVBoxLayout()

        # RGB图像路径
        rgb_layout = QHBoxLayout()
        rgb_label = QLabel("RGB图像路径:")
        self.rgb_path_edit = QLineEdit()
        self.rgb_path_edit.setPlaceholderText("选择RGB图像文件...")
        rgb_browse_btn = QPushButton("浏览...")
        rgb_browse_btn.clicked.connect(self.browse_rgb_image)
        rgb_layout.addWidget(rgb_label)
        rgb_layout.addWidget(self.rgb_path_edit)
        rgb_layout.addWidget(rgb_browse_btn)

        # 高光谱数据路径
        hsi_layout = QHBoxLayout()
        hsi_label = QLabel("高光谱数据路径:")
        self.hsi_path_edit = QLineEdit()
        self.hsi_path_edit.setPlaceholderText("选择高光谱数据文件...")
        hsi_browse_btn = QPushButton("浏览...")
        hsi_browse_btn.clicked.connect(self.browse_hsi_data)
        hsi_layout.addWidget(hsi_label)
        hsi_layout.addWidget(self.hsi_path_edit)
        hsi_layout.addWidget(hsi_browse_btn)

        # 输出路径
        output_layout = QHBoxLayout()
        output_label = QLabel("输出路径:")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("选择输出目录...")
        output_browse_btn = QPushButton("浏览...")
        output_browse_btn.clicked.connect(self.browse_output_path)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(output_browse_btn)

        io_layout.addLayout(rgb_layout)
        io_layout.addLayout(hsi_layout)
        io_layout.addLayout(output_layout)
        io_group.setLayout(io_layout)
        main_layout.addWidget(io_group)

        # 预览区域
        preview_group = QGroupBox("图像预览")
        preview_layout = QHBoxLayout()

        # RGB预览区域
        rgb_preview_box = QVBoxLayout()
        rgb_preview_label = QLabel("RGB图像预览")
        self.rgb_graphics_view = ZoomableGraphicsView()
        self.rgb_scene = QGraphicsScene()
        self.rgb_graphics_view.setScene(self.rgb_scene)
        rgb_preview_box.addWidget(rgb_preview_label)
        rgb_preview_box.addWidget(self.rgb_graphics_view)

        # 切割预览区域
        crop_preview_box = QVBoxLayout()
        crop_preview_label = QLabel("切割结果预览")
        self.crop_graphics_view = ZoomableGraphicsView()
        self.crop_scene = QGraphicsScene()
        self.crop_graphics_view.setScene(self.crop_scene)
        crop_preview_box.addWidget(crop_preview_label)
        crop_preview_box.addWidget(self.crop_graphics_view)

        preview_layout.addLayout(rgb_preview_box, 3)
        preview_layout.addLayout(crop_preview_box, 3)
        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group)

        # 控制按钮区域
        button_group = QGroupBox("操作控制")
        button_layout = QHBoxLayout()

        # 预览按钮
        self.preview_btn = QPushButton("预览图像")
        self.preview_btn.clicked.connect(self.preview_images)
        self.preview_btn.setStyleSheet("background-color: #4CAF50; color: white;")

        # 选择区域按钮
        self.select_roi_btn = QPushButton("选择切割区域")
        self.select_roi_btn.clicked.connect(self.select_roi)
        self.select_roi_btn.setStyleSheet("background-color: #2196F3; color: white;")

        # 应用切割按钮
        self.apply_cut_btn = QPushButton("应用切割")
        self.apply_cut_btn.clicked.connect(self.apply_cut)
        self.apply_cut_btn.setStyleSheet("background-color: #FF9800; color: white;")

        # 保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setStyleSheet("background-color: #f44336; color: white;")

        button_layout.addWidget(self.preview_btn)
        button_layout.addWidget(self.select_roi_btn)
        button_layout.addWidget(self.apply_cut_btn)
        button_layout.addWidget(self.save_btn)
        button_group.setLayout(button_layout)
        main_layout.addWidget(button_group)

        # 状态栏
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("就绪")

        # 设置中心部件
        self.setCentralWidget(main_widget)

        # 确保所有按钮和标签都能正确显示中文
        self.setStyleSheet("""
            QLabel, QPushButton, QGroupBox, QLineEdit {
                font-size: 12pt;
                font-family: "Microsoft YaHei", "SimHei", sans-serif;
            }
            QGroupBox {
                border: 1px solid gray;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-weight: bold;
            }
            QGraphicsView {
                background-color: #333;
                border: 1px solid #555;
            }
        """)

    def browse_rgb_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择RGB图像", "", "图像文件 (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.rgb_path_edit.setText(file_path)

    def browse_hsi_data(self):
        # 在实际应用中，这里应该加载高光谱数据文件
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择高光谱数据", "", "数据文件 (*.hdr *.raw *.npy *.mat)"
        )
        if file_path:
            self.hsi_path_edit.setText(file_path)

    def browse_output_path(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if dir_path:
            self.output_path_edit.setText(dir_path)

    def preview_images(self):
        rgb_path = self.rgb_path_edit.text()
        hsi_path = self.hsi_path_edit.text()

        if not rgb_path:
            self.status_bar.showMessage("请先选择RGB图像路径")
            return

        try:
            # 加载RGB图像
            self.rgb_image = cv2.imread(rgb_path)
            if self.rgb_image is None:
                self.status_bar.showMessage("无法加载RGB图像，请检查文件路径和格式")
                return

            # 调整图像大小以适应超大图像
            self.rgb_image = self.resize_image(self.rgb_image)

            # 转换为QImage
            height, width, channel = self.rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # 在场景中显示
            self.rgb_scene.clear()
            self.rgb_scene.addPixmap(QPixmap.fromImage(q_img))
            self.rgb_graphics_view.fitInView(self.rgb_scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.view_scale = self.rgb_graphics_view.get_scale()

            # 模拟加载高光谱数据
            self.hyperspectral_data = self.rgb_image.copy()

            self.status_bar.showMessage(f"图像已加载: {width}x{height}像素")
            self.preview_btn.setStyleSheet("background-color: #81C784; color: white;")
        except Exception as e:
            self.status_bar.showMessage(f"图像加载错误: {str(e)}")

    def resize_image(self, image, max_height=1000):
        """调整图像大小以适应预览窗口，同时保持超大图像的特性"""
        height, width = image.shape[:2]

        # 限制高度，保持原始宽度比例
        if height > max_height:
            scale_factor = max_height / height
            new_height = max_height
            new_width = int(width * scale_factor)
            image = cv2.resize(image, (new_width, new_height))

        return image

    def select_roi(self):
        if self.rgb_image is None:
            self.status_bar.showMessage("请先加载RGB图像")
            return

        self.status_bar.showMessage("选择切割区域：请点击并拖动鼠标在图像上绘制矩形区域")

        # 创建可选择的场景
        self.rgb_graphics_view.set_draw_roi(True)
        self.rgb_graphics_view.roi_selected.connect(self.roi_selected)

    def roi_selected(self, roi):
        self.roi = roi
        x, y, w, h = roi

        # 显示选择的区域
        if self.rgb_image is not None:
            # 绘制ROI矩形
            img_with_roi = self.rgb_image.copy()
            cv2.rectangle(img_with_roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

            # 转换为QImage
            height, width, channel = img_with_roi.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_with_roi.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # 在场景中显示
            self.rgb_scene.clear()
            self.rgb_scene.addPixmap(QPixmap.fromImage(q_img))
            self.rgb_graphics_view.setSceneRect(QRectF(0, 0, width, height))

            self.status_bar.showMessage(f"已选择区域: X={x}, Y={y}, 宽度={w}, 高度={h}")
            self.select_roi_btn.setStyleSheet("background-color: #64B5F6; color: white;")

    def apply_cut(self):
        if self.roi is None or self.rgb_image is None:
            self.status_bar.showMessage("请先选择切割区域")
            return

        x, y, w, h = self.roi

        # 在实际应用中，这里应该处理高光谱数据
        if self.hyperspectral_data is None:
            self.status_bar.showMessage("警告: 没有真实的高光谱数据，使用RGB图像模拟")
            cropped = self.rgb_image[y:y + h, x:x + w]
        else:
            # 模拟从高光谱数据中切割
            cropped = self.hyperspectral_data[y:y + h, x:x + w]

        # 转换为QImage
        if cropped is None or cropped.size == 0:
            self.status_bar.showMessage("切割区域无效或超出图像范围")
            return

        height, width, channel = cropped.shape
        bytes_per_line = 3 * width
        q_img = QImage(cropped.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # 在切割预览场景中显示
        self.crop_scene.clear()
        self.crop_scene.addPixmap(QPixmap.fromImage(q_img))
        self.crop_graphics_view.fitInView(self.crop_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        self.status_bar.showMessage(f"切割完成: 区域大小 {width}x{height}")
        self.apply_cut_btn.setStyleSheet("background-color: #FFB74D; color: white;")

    def save_results(self):
        if self.crop_scene.itemsBoundingRect().isEmpty():
            self.status_bar.showMessage("没有结果可保存")
            return

        output_dir = self.output_path_edit.text()
        if not output_dir:
            self.status_bar.showMessage("请先选择输出目录")
            return

        # 在实际应用中，这里应该保存高光谱数据
        output_path = os.path.join(output_dir, "cropped_result.png")

        # 获取切割预览场景中的图像
        pixmap = self.crop_scene.items()[0].pixmap()
        pixmap.save(output_path, "PNG")

        self.status_bar.showMessage(f"结果已保存到: {output_path}")
        self.save_btn.setStyleSheet("background-color: #EF9A9A; color: white;")


class ZoomableGraphicsView(QGraphicsView):
    roi_selected = pyqtSignal(tuple)  # 定义信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.draw_roi = False
        self.roi_start = None
        self.roi_end = None

    def mousePressEvent(self, event):
        if self.draw_roi and event.button() == Qt.LeftButton:
            # 开始绘制ROI
            scene_pos = self.mapToScene(event.pos())
            self.roi_start = scene_pos
            self.roi_end = scene_pos
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.draw_roi and self.roi_start is not None:
            # 更新ROI结束点
            scene_pos = self.mapToScene(event.pos())
            self.roi_end = scene_pos

            # 重绘场景
            self.scene().update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.draw_roi and event.button() == Qt.LeftButton and self.roi_start is not None:
            # 完成ROI绘制
            scene_pos = self.mapToScene(event.pos())
            self.roi_end = scene_pos

            # 计算ROI矩形
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            # 发送ROI信号
            self.roi_selected.emit((int(x), int(y), int(w), int(h)))

            # 重置状态
            self.draw_roi = False
            self.roi_start = None
            self.roi_end = None

            self.scene().update()
        else:
            super().mouseReleaseEvent(event)

    def set_draw_roi(self, enabled):
        self.draw_roi = enabled
        if enabled:
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            self.setDragMode(QGraphicsView.ScrollHandDrag)

    def get_scale(self):
        # 获取当前的缩放比例
        transform = self.transform()
        return transform.m11()

    def drawForeground(self, painter, rect):
        # 覆盖这个方法以在场景上绘制ROI矩形
        if self.roi_start and self.roi_end:
            painter.setPen(QPen(QColor(0, 255, 0), 3))
            x1, y1 = self.roi_start.x(), self.roi_start.y()
            x2, y2 = self.roi_end.x(), self.roi_end.y()

            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)

            painter.drawRect(x, y, w, h)

        super().drawForeground(painter, rect)

    def wheelEvent(self, event):
        # 缩放功能
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # 保存当前场景位置
        old_pos = self.mapToScene(event.pos())

        # 缩放
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

        # 获取缩放后的位置
        new_pos = self.mapToScene(event.pos())

        # 移动场景以保持鼠标位置不变
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 确保中文正常显示
    font = app.font()
    font.setFamily("Microsoft YaHei" if sys.platform == "win32" else "WenQuanYi Micro Hei")
    app.setFont(font)

    window = HyperspectralImageCutter()
    window.show()
    sys.exit(app.exec_())