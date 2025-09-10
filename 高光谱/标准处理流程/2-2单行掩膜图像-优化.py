import os
import sys
import cv2
import numpy as np
from skimage.morphology import skeletonize
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QSpinBox, QVBoxLayout, QHBoxLayout, QPushButton,
                             QFileDialog, QCheckBox, QMessageBox, QSplitter,
                             QGraphicsView, QGraphicsScene, QGraphicsLineItem,
                             QLineEdit, QGridLayout, QGroupBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QCursor, QTransform


class ImageProcessor(QThread):
    update_signal = pyqtSignal(np.ndarray, list, np.ndarray)

    def __init__(self):
        super().__init__()
        self.params = {
            'open_iter': 2,
            'min_area': 100,
            'prune_iter': 2,
            'dilate_iter': 5,
            'line_thickness': 2
        }
        self.original_image = None
        self.auto_red_lines = []
        self.skeleton_img = None

    def load_image(self, path):
        self.original_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if self.original_image is not None:
            self.process_image()

    def set_param(self, key, value):
        self.params[key] = value

    def detect_auto_lines(self, pruned):
        boundaries = []
        current_bound = None
        for y in range(pruned.shape[0]):
            if np.all(pruned[y, :] == 0):
                if current_bound is None:
                    current_bound = [y, y]
                else:
                    current_bound[1] = y
            else:
                if current_bound is not None:
                    boundaries.append(tuple(current_bound))
                    current_bound = None

        auto_lines = []
        for (upper, lower) in boundaries:
            center_y = (upper + lower) // 2
            region = pruned[upper:lower, :]
            if np.mean(region) <= 5:
                auto_lines.append(center_y)
        return auto_lines

    def process_image(self):
        try:
            img = self.original_image.copy()
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                      iterations=self.params['open_iter'])

            skeleton = skeletonize(opened / 255).astype(np.uint8) * 255

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
            denoised = np.zeros_like(skeleton)
            for label in range(1, num_labels):
                if stats[label, cv2.CC_STAT_AREA] >= self.params['min_area']:
                    denoised[labels == label] = 255
            skeleton = denoised

            self.skeleton_img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)

            pruned = skeleton.copy()
            endpoint_kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
            for _ in range(self.params['prune_iter']):
                conv = cv2.filter2D(pruned, cv2.CV_8U, endpoint_kernel)
                endpoints = (conv == 11).astype(np.uint8) * 255
                branches = cv2.dilate(endpoints, None, iterations=self.params['dilate_iter'])
                pruned = cv2.subtract(pruned, branches)

            self.auto_red_lines = self.detect_auto_lines(pruned)
            result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self.update_signal.emit(result, self.auto_red_lines, self.skeleton_img)
        except Exception as e:
            print(f"处理错误: {str(e)}")


class SyncGraphicsView(QGraphicsView):
    viewChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.pixmap_item = None
        self.zoom_factor = 1.0
        self.sync_viewer = None
        self.manual_lines = []
        self.current_line = None
        self.edit_mode = 'add'
        self.dragging = False
        self.original_width = 0
        self.original_height = 0

        self.h_scroll = self.horizontalScrollBar()
        self.v_scroll = self.verticalScrollBar()
        self.h_scroll.valueChanged.connect(self.sync_scroll)
        self.v_scroll.valueChanged.connect(self.sync_scroll)
        self.synced_views = []

    def link_viewers(self, *viewers):
        self.synced_views = list(viewers)
        for v in viewers:
            if v != self:
                self.h_scroll.valueChanged.connect(v.h_scroll.setValue)
                self.v_scroll.valueChanged.connect(v.v_scroll.setValue)
                v.h_scroll.valueChanged.connect(self.h_scroll.setValue)
                v.v_scroll.valueChanged.connect(self.v_scroll.setValue)

    def sync_scroll(self):
        for view in self.synced_views:
            if view != self:
                view.setTransform(self.transform())

    def set_image(self, pixmap):
        self.scene.clear()
        self.manual_lines.clear()
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.original_width = pixmap.width()
        self.original_height = pixmap.height()
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitToWindow()
        for v in self.synced_views:
            v.viewport().update()

    def fitToWindow(self):
        if self.pixmap_item:
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
            self.zoom_factor = self.transform().m11()

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            zoom_factor = 1.1 if event.angleDelta().y() > 0 else 0.9
            for view in self.synced_views:
                view.scale(zoom_factor, zoom_factor)
                view.zoom_factor *= zoom_factor
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.edit_mode == 'add':
                pos = self.mapToScene(event.pos())
                self.current_line = pos.y()
            elif self.edit_mode == 'delete':
                pos = self.mapToScene(event.pos())
                self._delete_nearest_line(pos.y())
            else:
                self.setDragMode(QGraphicsView.ScrollHandDrag)
                super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.current_line is not None:
            pos = self.mapToScene(event.pos())
            self.current_line = pos.y()
            self._draw_temp_line()
        else:
            super().mouseMoveEvent(event)
        self.sync_scroll()

    def mouseReleaseEvent(self, event):
        if self.current_line is not None:
            self.manual_lines.append(self.current_line)
            self.current_line = None
            self._draw_lines()
        self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def _draw_temp_line(self):
        self._draw_lines(temp_line=self.current_line)

    def _draw_lines(self, temp_line=None):
        for item in self.scene.items():
            if isinstance(item, QGraphicsLineItem) and not hasattr(item, 'permanent'):
                self.scene.removeItem(item)

        pen = QPen(Qt.red, 2)
        for y in self.manual_lines:
            line = QGraphicsLineItem(0, y, self.original_width, y)
            line.setPen(pen)
            line.setZValue(1)
            self.scene.addItem(line)

        if temp_line is not None:
            temp_line = QGraphicsLineItem(0, temp_line, self.original_width, temp_line)
            temp_line.setPen(QPen(Qt.red, 2, Qt.DashLine))
            self.scene.addItem(temp_line)

    def _delete_nearest_line(self, click_y):
        threshold = 10
        nearest = None
        min_dist = float('inf')

        for i, y in enumerate(self.manual_lines):
            dist = abs(y - click_y)
            if dist < threshold and dist < min_dist:
                min_dist = dist
                nearest = i

        if nearest is not None:
            del self.manual_lines[nearest]
            self._draw_lines()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("三视图同步分割工具")
        self.setGeometry(100, 100, 1600, 900)
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # 创建三个同步视图
        self.rgb_view = SyncGraphicsView()
        self.skeleton_view = SyncGraphicsView()
        self.main_view = SyncGraphicsView()

        # 建立同步连接
        views = [self.rgb_view, self.skeleton_view, self.main_view]
        for v in views:
            v.link_viewers(*views)

        # 主分割布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.rgb_view)
        splitter.addWidget(self.skeleton_view)
        splitter.addWidget(self.main_view)
        splitter.setSizes([300, 300, 600])

        # 控制面板
        control_layout = QGridLayout()
        self._create_processing_controls(control_layout)
        self._create_filename_controls(control_layout)
        self._create_action_buttons(control_layout)

        # 主界面布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)
        control_panel = QWidget()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        main_layout.addWidget(control_panel)
        main_layout.addWidget(splitter)

        self.setCentralWidget(main_widget)
        self.processor = ImageProcessor()
        self.processor.update_signal.connect(self._update_image_display)

    def _create_processing_controls(self, layout):
        group_box = QGroupBox("处理参数")
        group_layout = QVBoxLayout()

        self.open_iter_spin = self._create_param_spin("开运算迭代:", 1, 5, 2, 'open_iter')
        self.min_area_spin = self._create_param_spin("最小面积:", 1, 500, 100, 'min_area')
        self.prune_iter_spin = self._create_param_spin("修剪迭代:", 1, 5, 2, 'prune_iter')
        self.dilate_iter_spin = self._create_param_spin("膨胀迭代:", 1, 10, 5, 'dilate_iter')
        self.line_thickness_spin = self._create_param_spin("红线粗细:", 1, 10, 2, 'line_thickness')

        for spin in [self.open_iter_spin, self.min_area_spin,
                     self.prune_iter_spin, self.dilate_iter_spin,
                     self.line_thickness_spin]:
            group_layout.addWidget(spin)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box, 0, 0)

    def _create_filename_controls(self, layout):
        group_box = QGroupBox("文件名设置")
        group_layout = QVBoxLayout()

        # RGB图像加载
        group_layout.addWidget(QLabel("RGB图像:"))
        self.load_rgb_btn = QPushButton("加载RGB图像")
        self.load_rgb_btn.clicked.connect(self._load_rgb_image)
        group_layout.addWidget(self.load_rgb_btn)

        # 掩膜命名
        group_layout.addWidget(QLabel("基础名称:"))
        self.base_name_input = QLineEdit("line")
        group_layout.addWidget(self.base_name_input)

        group_layout.addWidget(QLabel("起始编号:"))
        self.start_num_spin = QSpinBox()
        self.start_num_spin.setMinimum(1)
        self.start_num_spin.setValue(1)
        group_layout.addWidget(self.start_num_spin)

        group_layout.addWidget(QLabel("尾缀位数:"))
        self.suffix_digits_spin = QSpinBox()
        self.suffix_digits_spin.setMinimum(0)
        self.suffix_digits_spin.setMaximum(3)
        self.suffix_digits_spin.setValue(0)
        group_layout.addWidget(self.suffix_digits_spin)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box, 1, 0)

    def _create_action_buttons(self, layout):
        group_box = QGroupBox("操作")
        group_layout = QVBoxLayout()

        self.edit_mode_check = QCheckBox("删除模式")
        self.load_btn = QPushButton("加载处理图像")
        self.update_btn = QPushButton("更新处理")
        self.auto_line_btn = QPushButton("一键添加自动红线")
        self.clear_lines_btn = QPushButton("清除所有红线")
        self.gen_mask_btn = QPushButton("生成掩膜")

        for btn in [self.edit_mode_check, self.load_btn, self.update_btn,
                    self.auto_line_btn, self.clear_lines_btn, self.gen_mask_btn]:
            group_layout.addWidget(btn)

        group_box.setLayout(group_layout)
        layout.addWidget(group_box, 2, 0)

    def _connect_signals(self):
        self.edit_mode_check.stateChanged.connect(
            lambda: setattr(self.main_view, 'edit_mode',
                            'delete' if self.edit_mode_check.isChecked() else 'add'))
        self.load_btn.clicked.connect(self._load_image)
        self.update_btn.clicked.connect(self._update_processing)
        self.auto_line_btn.clicked.connect(self.add_auto_lines)
        self.clear_lines_btn.clicked.connect(self.clear_all_lines)
        self.gen_mask_btn.clicked.connect(self.generate_masks)
        self.load_rgb_btn.clicked.connect(self._load_rgb_image)

    def _create_param_spin(self, label, min_val, max_val, default, param_key):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        spin.setPrefix(label)
        spin.valueChanged.connect(lambda v: self.processor.set_param(param_key, v))
        return spin

    def generate_masks(self):
        manual_lines = sorted(self.main_view.manual_lines)
        if len(manual_lines) < 2:
            QMessageBox.warning(self, "警告", "至少需要两条红线生成掩膜")
            return

        original = self.processor.original_image
        if original is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        save_dir = QFileDialog.getExistingDirectory(self, "选择保存目录")
        if not save_dir:
            return

        base_name = self.base_name_input.text().strip()
        if not base_name:
            QMessageBox.warning(self, "警告", "基础名称不能为空")
            return

        start_num = self.start_num_spin.value()
        suffix_digits = self.suffix_digits_spin.value()
        total_files = len(manual_lines) - 1

        for i in range(total_files):
            y1 = int(manual_lines[i])
            y2 = int(manual_lines[i + 1])
            if y1 > y2:
                y1, y2 = y2, y1

            mask = np.zeros_like(original)
            h, w = original.shape
            start = max(0, min(y1, h - 1))
            end = max(0, min(y2, h - 1))
            mask[start:end + 1, :] = original[start:end + 1, :]

            # 生成文件名
            if suffix_digits == 1:
                file_num = start_num + i
                filename = f"{base_name}{file_num}"
            else:
                group = i // suffix_digits
                subgroup = i % suffix_digits + 1
                main_num = start_num + group
                filename = f"{base_name}{main_num}-{subgroup}"

            if not filename.lower().endswith(('.png', '.jpg', '.bmp', '.tif')):
                filename += ".png"

            save_path = os.path.join(save_dir, filename)
            cv2.imwrite(save_path, mask)

        QMessageBox.information(self, "完成", f"生成{total_files}个掩膜到：{save_dir}")

    def _load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开处理图像", "", "影像文件 (*.png *.jpg *.bmp *.dcm)")
        if path:
            self.main_view.manual_lines.clear()
            self.processor.load_image(path)

    def _load_rgb_image(self):
        # 打开文件对话框，选择RGB图像
        path, _ = QFileDialog.getOpenFileName(
            self, "打开RGB图像", "", "影像文件 (*.png *.jpg *.bmp *.jpeg)")

        if path:
            # 使用OpenCV加载图像
            rgb_image = cv2.imread(path)
            if rgb_image is not None:
                # 将BGR格式转换为RGB格式
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

                # 将图像转换为QImage
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

                # 将QImage转换为QPixmap并显示在rgb_view中
                pixmap = QPixmap.fromImage(qimg)
                self.rgb_view.set_image(pixmap)
            else:
                QMessageBox.warning(self, "错误", "无法加载RGB图像，请检查文件格式和路径。")

    def _update_processing(self):
        if self.processor.original_image is not None:
            self.processor.process_image()

    def add_auto_lines(self):
        if not hasattr(self.processor, 'auto_red_lines'):
            return
        new_lines = [y for y in self.processor.auto_red_lines
                     if y not in self.main_view.manual_lines]
        self.main_view.manual_lines.extend(new_lines)
        self.main_view._draw_lines()

    def clear_all_lines(self):
        self.main_view.manual_lines.clear()
        self.main_view._draw_lines()

    def _update_image_display(self, img, auto_lines, skeleton_img):
        sk_h, sk_w = skeleton_img.shape[:2]
        sk_qimg = QImage(skeleton_img.data, sk_w, sk_h, 3 * sk_w, QImage.Format_RGB888)
        sk_pixmap = QPixmap.fromImage(sk_qimg)
        self.skeleton_view.set_image(sk_pixmap)

        h, w = img.shape[:2]
        qimg = QImage(img.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.main_view.set_image(pixmap)
        self.processor.auto_red_lines = auto_lines

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())