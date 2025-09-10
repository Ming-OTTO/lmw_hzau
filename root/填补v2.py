# binary_tool.py
import os
import sys
import copy
import cv2
import numpy as np
from collections import deque
from PyQt5.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QFileDialog,
                             QSpinBox, QCheckBox, QMessageBox, QLabel,
                             QOpenGLWidget)
from PyQt5.QtCore import Qt, QPointF, QRectF
from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPen, QBrush,
                         QColor)

# ------------------------------------------------------------
# 1. GPU 加速 + 连续绘制视图
# ------------------------------------------------------------
class SmoothViewer(QGraphicsView):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        # OpenGL 视口
        self.gl_widget = QOpenGLWidget()
        self.setViewport(self.gl_widget)
        self.setRenderHints(QPainter.Antialiasing |
                            QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        # 状态
        self.drawing = False
        self.right_dragging = False
        self.temp_path = []

    # ---------------- 坐标转换 ----------------
    def map_to_img(self, pos):
        p = self.mapToScene(pos)
        return int(p.x()), int(p.y())

    # ---------------- 载入 ndarray ----------------
    def set_image(self, img):
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))
        self.setSceneRect(QRectF(0, 0, w, h))
        self.resetTransform()

    # ---------------- 工具：一次画完整连续线 ----------------
    def _stroke_line(self, img, pts, color, thickness):
        if len(pts) < 2:
            return
        pts = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [pts], isClosed=False,
                      color=color, thickness=thickness,
                      lineType=cv2.LINE_AA)

    # ---------------- 鼠标事件 ----------------
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            x, y = self.map_to_img(ev.pos())
            if 0 <= x < self.parent.w and 0 <= y < self.parent.h:
                self.drawing = True
                self.parent.save_state()
                self.temp_path = [(x, y)]
        elif ev.button() == Qt.RightButton:
            self.right_dragging = True
            self.last_drag = ev.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.right_dragging:
            delta = ev.pos() - self.last_drag
            self.last_drag = ev.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
        elif self.drawing:
            x, y = self.map_to_img(ev.pos())
            if 0 <= x < self.parent.w and 0 <= y < self.parent.h:
                self.temp_path.append((x, y))
                self.draw_temp()
        # 指示器
        self.mouse_pos = ev.pos()
        self.viewport().update()
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.commit_path()
            self.temp_path = []
        elif ev.button() == Qt.RightButton:
            self.right_dragging = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(ev)

    # ---------------- 滚轮缩放 ----------------
    def wheelEvent(self, ev):
        if ev.modifiers() & Qt.ControlModifier:
            factor = 1.25 if ev.angleDelta().y() > 0 else 0.8
            self.scale(factor, factor)
        else:
            super().wheelEvent(ev)

    # ---------------- 临时绘制 ----------------
    def draw_temp(self):
        if len(self.temp_path) < 2:
            return
        img = self.parent.current_img.copy()
        color = 0 if self.parent.erase_mode else 255
        self._stroke_line(img, self.temp_path, color, self.parent.brush_size)
        self._update_qpixmap(img)

    # ---------------- 提交最终轨迹 ----------------
    def commit_path(self):
        if len(self.temp_path) < 2:
            return
        color = 0 if self.parent.erase_mode else 255
        self._stroke_line(self.parent.current_img, self.temp_path,
                          color, self.parent.brush_size)
        self.parent.viewer.set_image(self.parent.current_img)

    # ---------------- 更新显示 ----------------
    def _update_qpixmap(self, img):
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8)
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    # ---------------- 前景指示器 ----------------
    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        if not hasattr(self, 'mouse_pos'):
            return
        x, y = self.map_to_img(self.mouse_pos)
        if not (0 <= x < self.parent.w and 0 <= y < self.parent.h):
            return
        rad = self.parent.brush_size * self.transform().m11()
        color = Qt.red if self.parent.erase_mode else Qt.green
        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(color, Qt.SolidPattern))
        painter.setOpacity(0.4)
        painter.drawEllipse(QPointF(self.mapToScene(self.mouse_pos)),
                            rad, rad)

# ------------------------------------------------------------
# 主窗口
# ------------------------------------------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.current_img = None
        self.h = self.w = 0
        self.folder = ""
        self.files = []
        self.idx = 0
        self.brush_size = 10
        self.erase_mode = True

        self.history = deque(maxlen=50)
        self.redo_stack = []

        self.init_ui()
        self.choose_folder()
        if not self.files:
            QMessageBox.warning(self, "提示", "未选择有效文件夹或文件夹为空，即将退出")
            sys.exit(0)
        self.load_image()

    def init_ui(self):
        self.setWindowTitle("二值图批量编辑器（连续平滑绘制版）")
        self.setGeometry(100, 100, 900, 600)
        vbox = QVBoxLayout(self)

        self.viewer = SmoothViewer(self)
        vbox.addWidget(self.viewer)

        hbox = QHBoxLayout()
        hbox.addWidget(QLabel("画笔大小:"))
        self.size_spin = QSpinBox()
        self.size_spin.setRange(1, 100)
        self.size_spin.setValue(self.brush_size)
        self.size_spin.valueChanged.connect(lambda v: setattr(self, 'brush_size', v))
        hbox.addWidget(self.size_spin)

        self.mode_check = QCheckBox("擦除模式（取消勾选=填补）")
        self.mode_check.setChecked(self.erase_mode)
        self.mode_check.toggled.connect(lambda c: setattr(self, 'erase_mode', c))
        hbox.addWidget(self.mode_check)

        self.undo_btn = QPushButton("撤销 (Ctrl+Z)")
        self.redo_btn = QPushButton("重做 (Ctrl+Y)")
        self.undo_btn.clicked.connect(self.undo)
        self.redo_btn.clicked.connect(self.redo)
        hbox.addWidget(self.undo_btn)
        hbox.addWidget(self.redo_btn)
        vbox.addLayout(hbox)

        hbox2 = QHBoxLayout()
        self.prev_btn = QPushButton("上一张 (←)")
        self.next_btn = QPushButton("下一张 (→)")
        self.save_btn = QPushButton("保存 (S)")
        self.prev_btn.clicked.connect(self.prev_image)
        self.next_btn.clicked.connect(self.next_image)
        self.save_btn.clicked.connect(self.save_image)
        hbox2.addWidget(self.prev_btn)
        hbox2.addWidget(self.next_btn)
        hbox2.addWidget(self.save_btn)
        vbox.addLayout(hbox2)

    # ---------------- 递归遍历子目录 ----------------
    def choose_folder(self):
        self.folder = QFileDialog.getExistingDirectory(
            self, "请选择包含二值图的文件夹", "")
        if not self.folder:
            return
        self.files = []
        for root, _, files in os.walk(self.folder):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.bmp')):
                    self.files.append(os.path.join(root, f))
        self.files.sort()

    # ---------------- 撤销/重做 ----------------
    def save_state(self):
        self.history.append(copy.deepcopy(self.current_img))
        self.redo_stack.clear()

    def undo(self):
        if self.history:
            self.redo_stack.append(copy.deepcopy(self.current_img))
            self.current_img = self.history.pop()
            self.viewer.set_image(self.current_img)

    def redo(self):
        if self.redo_stack:
            self.history.append(copy.deepcopy(self.current_img))
            self.current_img = self.redo_stack.pop()
            self.viewer.set_image(self.current_img)

    # ---------------- 加载 ----------------
    def load_image(self):
        if not self.files:
            return
        self.current_img = cv2.imread(self.files[self.idx], cv2.IMREAD_GRAYSCALE)
        if self.current_img is None:
            QMessageBox.warning(self, "警告", f"无法读取 {self.files[self.idx]}")
            return
        self.h, self.w = self.current_img.shape
        self.history.clear()
        self.redo_stack.clear()
        self.save_state()
        self.viewer.set_image(self.current_img)
        self.setWindowTitle(f"{os.path.basename(self.files[self.idx])}  ({self.idx+1}/{len(self.files)})")

    # ---------------- 保存为新文件 ----------------
    def save_image(self):
        base, ext = os.path.splitext(self.files[self.idx])
        save_path = base + "_edit" + ext
        ok = cv2.imwrite(save_path, self.current_img)
        if ok:
            QMessageBox.information(self, "保存", f"已保存为\n{os.path.basename(save_path)}")
        else:
            QMessageBox.warning(self, "保存", "保存失败！")

    # ---------------- 键盘 ----------------
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Left:
            self.prev_image()
        elif ev.key() == Qt.Key_Right:
            self.next_image()
        elif ev.key() == Qt.Key_S:
            self.save_image()
        elif ev.modifiers() == Qt.ControlModifier:
            if ev.key() == Qt.Key_Z:
                self.undo()
            elif ev.key() == Qt.Key_Y:
                self.redo()

    def prev_image(self):
        if self.idx > 0:
            self.idx -= 1
            self.load_image()

    def next_image(self):
        if self.idx < len(self.files) - 1:
            self.idx += 1
            self.load_image()

# ------------------------------------------------------------
# 主程序
# ------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())