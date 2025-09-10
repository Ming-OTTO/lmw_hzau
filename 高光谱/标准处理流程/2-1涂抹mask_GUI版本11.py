# -*- coding: utf-8 -*-
"""
@author:puzhen
@data:2023/6/5 10:26
"""
from PyQt5.QtWidgets import (QGraphicsEffect, QGraphicsBlurEffect, QGraphicsOpacityEffect, QLabel,
                             QGraphicsProxyWidget, QVBoxLayout, QHBoxLayout, QToolTip, QApplication,
                             QShortcut, QSystemTrayIcon, QMenu, QTextEdit, QMessageBox, QWidget,
                             QPushButton, QLineEdit, QScrollArea, QFileDialog, QMainWindow)
from PyQt5.QtGui import (QBrush, QColor, QPainter, QPixmap, QBitmap, QPainterPath, QPen,
                         QCursor, QKeySequence, QFont, QImage)
from PyQt5 import QtCore
from PyQt5.QtCore import Qt, QRectF, QPropertyAnimation, QPoint, QRect, QEvent, QTimer, QObject
import pandas as pd
import sys
import cv2
import numpy as np


class Ui_MainWindow(QObject):  # 继承QObject以支持事件过滤器
    def __init__(self):
        super().__init__()  # 初始化QObject父类

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 800)
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setMouseTracking(True)

        # 初始化参数
        self.radius = 10  # 画笔大小
        self.times = 0  # 列表读取序号
        self.chongfu = 2  # 盆栽重复个数
        self.i = 1  # 命名计数器

        # 初始化变量，避免未定义访问
        self.lines = []
        self.painted_pixels = []
        self.Names = []
        self.original_pixmap = QPixmap()
        self.binary_pixmap = QPixmap()
        self.original_pixmap_copy = None
        self.binary_pixmap_copy = None
        self.binary_image = None
        self.binary_array_shape = (0, 0)
        self.last_pos = None

        # 初始化掩码为None，准备累积绘制
        self.mask_original = None
        self.mask_binary = None

        # 创建界面元素
        self.xlsx_label = QLabel("xlsx文件路径:", self.centralwidget)
        self.xlsx_path = QLineEdit(self.centralwidget)
        self.image_folder_label = QLabel("图片文件夹路径:", self.centralwidget)
        self.image_folder_path = QLineEdit(self.centralwidget)

        self.xlsx_btn = QPushButton("选择命名文件")
        self.xlsx_btn.clicked.connect(self.select_xlsx)
        self.folder_btn = QPushButton("选择保存文件夹")
        self.folder_btn.clicked.connect(self.select_folder)

        # 创建垂直布局管理器
        main_layout = QVBoxLayout(self.centralwidget)

        # 创建水平布局容器，用于图片和滚动区
        image_layout = QHBoxLayout()

        self.original_image_box = QLabel(self.centralwidget)
        self.original_image_box.setObjectName("original_image_box")
        self.original_image_box.setMouseTracking(True)
        self.original_image_box.setAlignment(Qt.AlignCenter)  # 图像居中显示

        self.binary_image_box = QLabel(self.centralwidget)
        self.binary_image_box.setObjectName("binary_image_box")
        self.binary_image_box.setMouseTracking(True)
        self.binary_image_box.setAlignment(Qt.AlignCenter)  # 图像居中显示

        # 设置滚动区域
        self.scroll_area_original = QScrollArea(self.centralwidget)
        self.scroll_area_original.setWidgetResizable(False)  # 禁止自动拉伸内部widget
        self.scroll_area_original.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area_original.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area_original.setWidget(self.original_image_box)
        self.scroll_area_original.installEventFilter(self)  # 安装事件过滤器用于缩放

        self.scroll_area_binary = QScrollArea(self.centralwidget)
        self.scroll_area_binary.setWidgetResizable(False)  # 禁止自动拉伸内部widget
        self.scroll_area_binary.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area_binary.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area_binary.setWidget(self.binary_image_box)
        self.scroll_area_binary.installEventFilter(self)  # 安装事件过滤器用于缩放

        image_layout.addWidget(self.scroll_area_original)
        image_layout.addWidget(self.scroll_area_binary)
        main_layout.addLayout(image_layout)

        # 路径布局
        path_layout = QHBoxLayout()
        path_layout.addWidget(self.xlsx_label)
        path_layout.addWidget(self.xlsx_path)
        path_layout.addWidget(self.xlsx_btn)

        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.image_folder_label)
        folder_layout.addWidget(self.image_folder_path)
        folder_layout.addWidget(self.folder_btn)

        main_layout.addLayout(path_layout)
        main_layout.addLayout(folder_layout)

        # 同步滚动条
        self.scroll_area_original.verticalScrollBar().valueChanged.connect(
            lambda value: self.sync_vertical_scroll_bar(value, self.scroll_area_original.verticalScrollBar(),
                                                        self.scroll_area_binary.verticalScrollBar()))
        self.scroll_area_binary.verticalScrollBar().valueChanged.connect(
            lambda value: self.sync_vertical_scroll_bar(value, self.scroll_area_binary.verticalScrollBar(),
                                                        self.scroll_area_original.verticalScrollBar()))

        self.scroll_area_original.horizontalScrollBar().valueChanged.connect(
            lambda value: self.sync_horizontal_scroll_bar(value, self.scroll_area_original.horizontalScrollBar(),
                                                          self.scroll_area_binary.horizontalScrollBar()))
        self.scroll_area_binary.horizontalScrollBar().valueChanged.connect(
            lambda value: self.sync_horizontal_scroll_bar(value, self.scroll_area_binary.horizontalScrollBar(),
                                                          self.scroll_area_original.horizontalScrollBar()))

        # 按钮布局
        self.load_image_button_Binary = QPushButton(self.centralwidget)
        self.load_image_button_Binary.setObjectName("load_image_button_Binary")
        self.load_image_button_Original = QPushButton(self.centralwidget)
        self.load_image_button_Original.setObjectName("load_image_button_Original")
        self.save_roi_button = QPushButton(self.centralwidget)
        self.save_roi_button.setObjectName("save_roi_button")
        self.clear_button = QPushButton(self.centralwidget)  # 新增清除按钮
        self.clear_button.setText("清除绘制")
        self.clear_button.clicked.connect(self.clear_drawing)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_image_button_Binary)
        button_layout.addStretch()
        button_layout.addWidget(self.load_image_button_Original)
        button_layout.addStretch()
        button_layout.addWidget(self.save_roi_button)
        button_layout.addStretch()
        button_layout.addWidget(self.clear_button)  # 添加到布局
        main_layout.addLayout(button_layout)

        # 连接按钮和拖放事件
        self.load_image_button_Binary.clicked.connect(self.load_binary_image)
        self.load_image_button_Original.clicked.connect(self.load_original_image)
        self.original_image_box.setAcceptDrops(True)
        self.binary_image_box.setAcceptDrops(True)
        self.original_image_box.dragEnterEvent = self.drag_enter_event
        self.original_image_box.dropEvent = lambda event: self.drop_event(event, self.original_image_box)
        self.binary_image_box.dragEnterEvent = self.drag_enter_event
        self.binary_image_box.dropEvent = lambda event: self.drop_event(event, self.binary_image_box)

        # 设置绘图工具
        self.original_painter = QPainter()
        self.binary_painter = QPainter()
        self.original_pen = QPen(Qt.red, 2 * self.radius, Qt.SolidLine)
        self.binary_pen = QPen(Qt.red, 2 * self.radius, Qt.SolidLine)

        # 设置鼠标事件
        self.original_image_box.mousePressEvent = lambda event: self.mouse_press_event(event, self.original_image_box)
        self.original_image_box.mouseMoveEvent = lambda event: self.mouse_move_event(event, self.original_image_box)
        self.original_image_box.mouseReleaseEvent = lambda event: self.mouse_release_event(event,
                                                                                           self.original_image_box)
        self.binary_image_box.mousePressEvent = lambda event: self.mouse_press_event(event, self.binary_image_box)
        self.binary_image_box.mouseMoveEvent = lambda event: self.mouse_move_event(event, self.binary_image_box)
        self.binary_image_box.mouseReleaseEvent = lambda event: self.mouse_release_event(event, self.binary_image_box)

        # 保存ROI
        self.save_roi_button.clicked.connect(self.save_roi)

        # 鼠标进入离开事件
        self.original_image_box.enterEvent = self.enterEvent
        self.original_image_box.leaveEvent = self.leaveEvent
        self.binary_image_box.enterEvent = self.enterEvent
        self.binary_image_box.leaveEvent = self.leaveEvent

        # 画笔大小设置
        self.radius_label = QLabel("画笔大小:", self.centralwidget)
        self.radius_input = QLineEdit(self.centralwidget)
        self.radius_input.setText(str(self.radius))  # 显示当前值

        radius_layout = QHBoxLayout()
        radius_layout.addWidget(self.radius_label)
        radius_layout.addWidget(self.radius_input)
        main_layout.addLayout(radius_layout)

        # 应用按钮
        self.apply_button = QPushButton(self.centralwidget)
        self.apply_button.setText("应用")
        main_layout.addWidget(self.apply_button)
        self.apply_button.clicked.connect(self.apply_changes)

        # 快捷键Ctrl+Space保存
        self.save_roi_shortcut = QShortcut(QKeySequence("Ctrl+Space"), self.centralwidget)
        self.save_roi_shortcut.activated.connect(self.save_roi)

        # 设置主窗口
        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def sync_vertical_scroll_bar(self, value, sender, receiver):
        receiver.setValue(value)

    def sync_horizontal_scroll_bar(self, value, sender, receiver):
        receiver.setValue(value)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "高光谱图像Mask处理工具"))
        self.load_image_button_Binary.setText(_translate("MainWindow", "加载二值图像"))
        self.load_image_button_Original.setText(_translate("MainWindow", "加载原始图像"))
        self.save_roi_button.setText(_translate("MainWindow", "保存ROI"))

    def load_original_image(self):
        fname = QFileDialog.getOpenFileName(None, '打开文件', './', "图像文件 (*.png *.jpg *.bmp *.tif)")
        if not fname[0]:  # 取消选择
            return

        self.original_pixmap = QPixmap(fname[0])
        if self.original_pixmap.isNull():
            QMessageBox.warning(self.centralwidget, "错误", "原始图像加载失败！")
            return

        # 显示原始尺寸并让QLabel自适应
        self.original_image_box.setPixmap(self.original_pixmap)
        self.original_image_box.adjustSize()  # 调整QLabel大小以适应图像

        # 初始化掩码（清除之前的绘制）
        self.mask_original = QPixmap(self.original_pixmap.size())
        self.mask_original.fill(QColor(0, 0, 0, 0))

        # 保存OpenCV格式图像
        try:
            self.original_pixmap_copy = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            QMessageBox.warning(self.centralwidget, "错误", f"图像转换失败: {str(e)}")
            self.original_pixmap_copy = None

    def load_binary_image(self):
        fname = QFileDialog.getOpenFileName(None, '打开文件', './', "图像文件 (*.png *.jpg *.bmp *.tif)")
        if not fname[0]:  # 取消选择
            return

        self.binary_pixmap = QPixmap(fname[0])
        if self.binary_pixmap.isNull():
            QMessageBox.warning(self.centralwidget, "错误", "二值图像加载失败！")
            return

        # 显示原始尺寸并让QLabel自适应
        self.binary_image_box.setPixmap(self.binary_pixmap)
        self.binary_image_box.adjustSize()  # 调整QLabel大小以适应图像

        # 初始化掩码（清除之前的绘制）
        self.mask_binary = QPixmap(self.binary_pixmap.size())
        self.mask_binary.fill(QColor(0, 0, 0, 0))

        # 保存OpenCV格式图像
        try:
            self.binary_pixmap_copy = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
            self.binary_image = cv2.imdecode(np.fromfile(fname[0], dtype=np.uint8), -1)
            _, binary_image = cv2.threshold(self.binary_image, 128, 255, cv2.THRESH_BINARY)
            self.binary_array_shape = binary_image.shape[:2]
        except Exception as e:
            QMessageBox.warning(self.centralwidget, "错误", f"图像转换失败: {str(e)}")
            self.binary_pixmap_copy = None
            self.binary_array_shape = (0, 0)

    def drag_enter_event(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def drop_event(self, event, image_box):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            pixmap = QPixmap(file_path)

            if pixmap.isNull():
                QMessageBox.warning(self.centralwidget, "错误", "图像加载失败！")
                event.ignore()
                return

            # 显示原始尺寸并让QLabel自适应
            image_box.setPixmap(pixmap)
            image_box.adjustSize()  # 调整QLabel大小以适应图像

            if image_box == self.original_image_box:
                self.original_pixmap = pixmap.copy()
                # 初始化掩码
                self.mask_original = QPixmap(self.original_pixmap.size())
                self.mask_original.fill(QColor(0, 0, 0, 0))
                try:
                    self.original_pixmap_copy = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8),
                                                             cv2.IMREAD_GRAYSCALE)
                except Exception as e:
                    QMessageBox.warning(self.centralwidget, "错误", f"图像转换失败: {str(e)}")
            elif image_box == self.binary_image_box:
                self.binary_pixmap = pixmap.copy()
                # 初始化掩码
                self.mask_binary = QPixmap(self.binary_pixmap.size())
                self.mask_binary.fill(QColor(0, 0, 0, 0))
                try:
                    self.binary_pixmap_copy = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    self.binary_image = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                    _, binary_image = cv2.threshold(self.binary_image, 128, 255, cv2.THRESH_BINARY)
                    self.binary_array_shape = binary_image.shape[:2]
                except Exception as e:
                    QMessageBox.warning(self.centralwidget, "错误", f"图像转换失败: {str(e)}")
                    self.binary_array_shape = (0, 0)

            event.accept()
        else:
            event.ignore()

    def enterEvent(self, event):
        # 设置自定义光标
        if event.type() == QEvent.Enter:
            cursor_radius = self.radius
            cursor_size = cursor_radius * 2 + 1
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.transparent)

            painter = QPainter(pixmap)
            pen = QPen(Qt.green, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawEllipse(0, 0, cursor_size - 1, cursor_size - 1)
            painter.end()

            cursor = QCursor(pixmap, cursor_radius, cursor_radius)
            self.original_image_box.setCursor(cursor)
            self.binary_image_box.setCursor(cursor)

    def leaveEvent(self, event):
        # 还原默认光标
        if event.type() == QEvent.Leave:
            self.original_image_box.unsetCursor()
            self.binary_image_box.unsetCursor()

    def mouse_press_event(self, event, image_box):
        if event.button() == Qt.LeftButton and not self.original_pixmap.isNull() and not self.binary_pixmap.isNull():
            self.last_pos = event.pos()
            # 确保掩码已初始化
            if self.mask_original is None:
                self.mask_original = QPixmap(self.original_pixmap.size())
                self.mask_original.fill(QColor(0, 0, 0, 0))
            if self.mask_binary is None:
                self.mask_binary = QPixmap(self.binary_pixmap.size())
                self.mask_binary.fill(QColor(0, 0, 0, 0))

    def mouse_move_event(self, event, image_box):
        # 检查图像是否已加载
        if self.original_pixmap.isNull() or self.binary_pixmap.isNull():
            return

        # 确保掩码已初始化
        if self.mask_original is None or self.mask_binary is None:
            return

        # 设置画笔
        self.original_pen = QPen(Qt.red, 2 * self.radius, Qt.SolidLine)
        self.binary_pen = QPen(Qt.red, 2 * self.radius, Qt.SolidLine)
        self.original_pen.setCapStyle(Qt.RoundCap)
        self.binary_pen.setCapStyle(Qt.RoundCap)

        if (event.buttons() & Qt.LeftButton) and self.last_pos is not None:
            # 在现有掩码上继续绘制（关键修复：不再重新创建空白掩码）
            self.binary_painter.begin(self.mask_binary)
            self.original_painter.begin(self.mask_original)

            self.binary_painter.setPen(self.binary_pen)
            self.original_painter.setPen(self.original_pen)

            # 绘制当前线段（累积到掩码上）
            self.binary_painter.drawLine(event.pos(), self.last_pos)
            self.original_painter.drawLine(event.pos(), self.last_pos)

            self.binary_painter.end()
            self.original_painter.end()

            # 合并原图和掩码
            temp_original = self.original_pixmap.copy()
            temp_binary = self.binary_pixmap.copy()

            painter_original = QPainter(temp_original)
            painter_original.drawPixmap(0, 0, self.mask_original)
            painter_original.end()

            painter_binary = QPainter(temp_binary)
            painter_binary.drawPixmap(0, 0, self.mask_binary)
            painter_binary.end()

            # 更新显示并保持尺寸
            self.binary_image_box.setPixmap(temp_binary)
            self.binary_image_box.adjustSize()
            self.original_image_box.setPixmap(temp_original)
            self.original_image_box.adjustSize()

            # 记录绘制轨迹
            self.painted_pixels.append((event.pos().x(), event.pos().y()))
            self.lines.append((event.pos(), self.last_pos))

            self.last_pos = event.pos()

    def mouse_release_event(self, event, image_box):
        if event.button() == Qt.LeftButton:
            self.last_pos = None

    def apply_changes(self):
        # 更新画笔大小
        try:
            new_radius = int(self.radius_input.text())
            if 1 <= new_radius <= 50:  # 限制在合理范围
                self.radius = new_radius
                QMessageBox.information(self.centralwidget, "提示", f"画笔大小已设置为 {new_radius}")
            else:
                QMessageBox.warning(self.centralwidget, "错误", "画笔大小需在1-50之间！")
        except ValueError:
            QMessageBox.warning(self.centralwidget, "错误", "请输入有效的整数！")

    def clear_drawing(self):
        """清除当前绘制的内容"""
        if not self.original_pixmap.isNull() and self.mask_original is not None:
            self.mask_original.fill(QColor(0, 0, 0, 0))
            self.original_image_box.setPixmap(self.original_pixmap.copy())
            self.original_image_box.adjustSize()

        if not self.binary_pixmap.isNull() and self.mask_binary is not None:
            self.mask_binary.fill(QColor(0, 0, 0, 0))
            self.binary_image_box.setPixmap(self.binary_pixmap.copy())
            self.binary_image_box.adjustSize()

        self.lines = []
        self.painted_pixels = []

    def save_roi(self):
        # 检查必要条件
        if self.binary_array_shape == (0, 0):
            QMessageBox.warning(self.centralwidget, "错误", "请先加载二值图像！")
            return

        if not self.Names:
            QMessageBox.warning(self.centralwidget, "错误", "请先选择命名文件！")
            return

        if not self.image_folder_path.text():
            QMessageBox.warning(self.centralwidget, "错误", "请先选择保存文件夹！")
            return

        if self.times >= len(self.Names):
            QMessageBox.warning(self.centralwidget, "错误", "命名列表已遍历完毕！")
            return

        if not self.lines:
            QMessageBox.warning(self.centralwidget, "警告", "未绘制任何区域，无需保存！")
            return

        try:
            binary_height, binary_width = self.binary_array_shape
            if binary_height <= 0 or binary_width <= 0:
                QMessageBox.warning(self.centralwidget, "错误", "二值图像尺寸无效！")
                return

            # 创建ROI图像
            roi_image = np.zeros((binary_height, binary_width), dtype=np.uint8)

            # 绘制ROI区域
            for line in self.lines:
                start_pos, end_pos = line
                # 确保坐标在有效范围内
                start_x = max(0, min(start_pos.x(), binary_width - 1))
                start_y = max(0, min(start_pos.y(), binary_height - 1))
                end_x = max(0, min(end_pos.x(), binary_width - 1))
                end_y = max(0, min(end_pos.y(), binary_height - 1))
                cv2.line(roi_image, (start_x, start_y), (end_x, end_y), color=255, thickness=self.radius * 2)

            # 与原始二值图像相乘
            if self.binary_image is not None:
                roi_masked = cv2.bitwise_and(self.binary_image, self.binary_image, mask=roi_image)
            else:
                roi_masked = roi_image

            # 保存图像
            save_path = f"{self.image_folder_path.text()}\\{self.Names[self.times]}_{self.i}.png"
            success = cv2.imwrite(save_path, roi_masked)

            if not success:
                QMessageBox.warning(self.centralwidget, "错误", f"保存失败: {save_path}")
                return

            # 显示提示
            msg_box = QMessageBox(self.centralwidget)
            msg_box.setText(f"{self.Names[self.times]}_{self.i}.png 已保存")
            msg_box.setWindowTitle("提示")
            msg_box.show()
            QTimer.singleShot(300, msg_box.close)  # 300ms后自动关闭

            # 重置绘制轨迹并更新计数器
            self.clear_drawing()  # 清除绘制内容
            self.i += 1

            if self.i > self.chongfu:
                self.i = 1
                self.times += 1

        except Exception as e:
            QMessageBox.warning(self.centralwidget, "错误", f"保存过程出错: {str(e)}")

    def select_xlsx(self):
        filename, _ = QFileDialog.getOpenFileName(self.centralwidget, "选择命名文件",
                                                  filter="Excel文件 (*.xlsx);;CSV文件 (*.csv)")
        if not filename:
            return

        try:
            if filename.endswith('.xlsx'):
                self.data = pd.read_excel(filename)
            elif filename.endswith('.csv'):
                self.data = pd.read_csv(filename)
            else:
                QMessageBox.warning(self.centralwidget, "错误", "不支持的文件格式！")
                return

            self.Names = self.data.iloc[:, 0].tolist()
            self.xlsx_path.setText(filename)
            QMessageBox.information(self.centralwidget, "提示", f"已加载 {len(self.Names)} 个名称")

        except Exception as e:
            QMessageBox.warning(self.centralwidget, "错误",
                                f"文件读取失败: {str(e)}\n可能需要安装openpyxl: pip install openpyxl")
            self.Names = []

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self.centralwidget, "选择保存文件夹")
        if folder:
            self.image_folder_path.setText(folder)
            print("图片文件夹路径:", folder)

    def eventFilter(self, obj, event):
        """事件过滤器：实现Ctrl+鼠标滚轮缩放图像"""
        if obj in (self.scroll_area_original, self.scroll_area_binary) and event.type() == QtCore.QEvent.Wheel:
            # 仅在按住Ctrl键时触发缩放
            if event.modifiers() & Qt.ControlModifier:
                # 获取当前滚动区域中的标签
                label = obj.widget()
                current_pixmap = label.pixmap()

                if current_pixmap and not current_pixmap.isNull():
                    # 计算缩放因子（滚轮向上放大，向下缩小）
                    factor = 1.1 if event.angleDelta().y() > 0 else 0.9

                    # 计算新尺寸（限制最小尺寸为100x100，最大为原始尺寸的5倍）
                    original_width = current_pixmap.width()
                    original_height = current_pixmap.height()
                    new_width = int(current_pixmap.width() * factor)
                    new_height = int(current_pixmap.height() * factor)

                    # 限制缩放范围
                    min_size = 100
                    max_size = max(original_width, original_height) * 5
                    new_width = max(min_size, min(new_width, max_size))
                    new_height = max(min_size, min(new_height, max_size))

                    # 缩放图像并保持比例
                    scaled_pixmap = current_pixmap.scaled(
                        new_width, new_height,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation  # 平滑缩放
                    )

                    # 更新标签显示
                    label.setPixmap(scaled_pixmap)
                    label.adjustSize()

                    # 保持当前视图中心（优化体验）
                    self.center_scroll_view(obj, scaled_pixmap.size(), current_pixmap.size())

                # 拦截事件，防止滚动条响应
                return True

        # 其他事件正常处理
        return super().eventFilter(obj, event)

    def center_scroll_view(self, scroll_area, new_size, old_size):
        """缩放后保持视图中心位置"""
        # 计算尺寸变化比例
        scale_x = new_size.width() / old_size.width() if old_size.width() > 0 else 1
        scale_y = new_size.height() / old_size.height() if old_size.height() > 0 else 1

        # 获取当前滚动条位置
        h_bar = scroll_area.horizontalScrollBar()
        v_bar = scroll_area.verticalScrollBar()
        old_h_pos = h_bar.value()
        old_v_pos = v_bar.value()

        # 计算新位置（保持原中心）
        new_h_pos = old_h_pos * scale_x + (new_size.width() - old_size.width() * scale_x) / 2
        new_v_pos = old_v_pos * scale_y + (new_size.height() - old_size.height() * scale_y) / 2

        # 设置新位置
        h_bar.setValue(int(new_h_pos))
        v_bar.setValue(int(new_v_pos))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
