#!/usr/bin/python3
#
# Copyright 2018 Dustin Kleckner
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, \
#      QHBoxLayout, QVBoxLayout, QOpenGLWidget, QSlider, QAction, QOpenGLShader, \
#      QOpenGLShaderProgram

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import sys
import traceback
import math

from .view import View

GAMMA_CORRECT = 2.2


class BoolViewSetting(QCheckBox):
    def __init__(self, gl_display, varname, default, parent=None):
        super().__init__(None, parent)
        self.setChecked(default)
        self.stateChanged.connect(lambda state: gl_display.update_view_settings(
            **{varname:(state==Qt.Checked)}))


class LinearViewSetting(QDoubleSpinBox):
    def __init__(self, gl_display, varname, default, minval, maxval, step, decimals=None, parent=None):
        super().__init__(parent)
        self.setValue(default)
        self.setRange(minval, maxval)
        self.setMaximum(maxval)
        self.setSingleStep(step)
        if decimals is None: decimals= math.ceil(0.5-math.log10(step))
        self.setDecimals(decimals)

        self.valueChanged.connect(lambda val: gl_display.update_view_settings(
            **{varname:val}))


    def set_from_slider(self, val, slider_max):
        minval = self.minumum()
        maxval = self.maximum()
        self.setValue(minval + (val/slider_max)*(maxval-minval))


class IntViewSetting(QSpinBox):
    def __init__(self, gl_display, varname, default, minval, maxval, rollover=True, parent=None):
        super().__init__(parent)
        self.setValue(default)
        self.setRange(minval, maxval)
        self.setMaximum(maxval)
        self.setSingleStep(1)
        self.rollover = rollover

        self.valueChanged.connect(lambda val: gl_display.update_view_settings(
            **{varname:val}))

    def advance(self):
        maxval = self.maximum()
        if maxval != 0:
            i = self.value() + 1
            if self.rollover: i %= self.maximum()
            self.setValue(i)

# class SlaveSlider(QSlider):
#     def __init__(self, master, ticks=1000, parent=None):
#         super().__init__(parent)
#         self.valueChange.connect(_valueChanged)
#         self.ticks = ticks()
#
#     def _valueChanged(self, event):
#         if self.active:
#             self.master.set_from_slider(self.value(), self.ticks)
#
#     def silent_change()

class LogViewSetting(LinearViewSetting):
    def __init__(self, gl_display, varname, default, minval, maxval, step, decimals=None, change_func=None, parent=None):
        if decimals is None: decimals= math.ceil(0.5-math.log10(minval))
        super().__init__(gl_display, varname, default, minval, maxval, step, decimals=decimals, parent=parent)

    def stepBy(self, step):
        sm = self.singleStep()
        i = math.log(self.value(), sm) + step
        self.setValue(sm**round(i))


def fromQColor(qc, has_alpha):
    if has_alpha:
        return (qc.redF()**GAMMA_CORRECT, qc.greenF()**GAMMA_CORRECT,
            qc.blueF()**GAMMA_CORRECT, qc.alphaF())
    else:
        return (qc.redF()**GAMMA_CORRECT, qc.greenF()**GAMMA_CORRECT,
            qc.blueF()**GAMMA_CORRECT)


def toQColor(t):
    if len(t) == 3:
        return QColor(255*t[0]**(1./GAMMA_CORRECT), 255*t[1]**(1./GAMMA_CORRECT), 255*t[2]**(1./GAMMA_CORRECT),)
    elif len(t) == 4:
        return QColor(255*t[0]**(1./GAMMA_CORRECT), 255*t[1]**(1./GAMMA_CORRECT), 255*t[2]**(1./GAMMA_CORRECT), 255*t[3])
    else:
        raise ValueError('input should have 3 or 4 components')


class ColorViewSetting(QFrame):
    def __init__(self, gl_display, varname, default, parent=None):
        self.has_alpha = len(default) == 4
        super().__init__(parent=parent)
        self.setValue(toQColor(default))
        self.gl_display = gl_display
        self.varname = varname

    def setValue(self, color):
        self.color = color
        self.setStyleSheet("QWidget { background-color: %s }" %
                self.color.name())

    def mousePressEvent(self, event):
        if self.has_alpha:
            self.setValue(QColorDialog.getColor(self.color, options=QColorDialog.ShowAlphaChannel))
        else:
            self.setValue(QColorDialog.getColor(self.color))
        self.gl_display.update_view_settings(
            **{self.varname:fromQColor(self.color, self.has_alpha)})


class ViewWidget(QOpenGLWidget):
    xRotationChanged = pyqtSignal(int)
    yRotationChanged = pyqtSignal(int)
    zRotationChanged = pyqtSignal(int)

    def __init__(self, parent=None, volume=None):
        super().__init__(parent)
        self.parent = parent
        self.lastPos = QPoint()
        self.volume = volume


    def minimumSizeHint(self):
        return QSize(300, 300)


    def sizeHint(self):
        return QSize(800, 600)


    def initializeGL(self):
        # These are not needed, and throw errors on Windows installs
        # self.gl = self.context().versionFunctions()
        # self.gl.initializeOpenGLFunctions()
        self.dpr = self.devicePixelRatio()
        try:
            self.view = View(volume=self.volume, width=self.width()*self.dpr,
                height=self.height()*self.dpr)
            if hasattr(self, "frame_setting"):
                self.frame_setting.setMaximum(len(self.volume)-1)
        except:
            traceback.print_exc()
            self.parent.close()


    def paintGL(self):
        try:
            if hasattr(self, 'view'): self.view.draw()
        except:
            traceback.print_exc()
            self.parent.close()


    def resizeGL(self, width, height):
        if hasattr(self, 'view'): self.view.resize(width*self.dpr, height*self.dpr)


    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.view.buttons_pressed = int(event.buttons())
        self.update()


    def mouseReleaseEvent(self, event):
        self.view.buttons_pressed = int(event.buttons())
        self.update()


    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        dx = x - self.lastPos.x()
        dy = y - self.lastPos.y()
        self.lastPos = event.pos()
        self.view.buttons_pressed = int(event.buttons())
        self.view.mouse_move(x*self.dpr, y*self.dpr, dx*self.dpr, dy*self.dpr)
        self.update()


    def wheelEvent(self, event):
        self.view.scale *= 1.25**(event.angleDelta().y()/120)
        self.update()


    def update_view_settings(self, **kwargs):
        self.view.update_view_settings(**kwargs)
        self.update()


class CollapsableVBox(QWidget):
    def __init__(self, parent=None, title='', isOpen=False):
        super().__init__(parent=parent)

        self.headerLine = QFrame()
        self.toggleButton = QToolButton()
        self.mainLayout = QGridLayout()
        self.contentsWidget = QWidget()
        self.contentsLayout = QGridLayout()
        self.contentsWidget.setLayout(self.contentsLayout)

        self.toggleButton.setStyleSheet("QToolButton { border: none; }")
        self.toggleButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggleButton.setText(str(title))
        self.toggleButton.setCheckable(True)
        self.toggleButton.setChecked(isOpen)
        self.toggleButton.setArrowType(Qt.DownArrow if self.toggleButton.isChecked() else Qt.RightArrow)

        self.headerLine.setFrameShape(QFrame.HLine)
        self.headerLine.setFrameShadow(QFrame.Sunken)
        self.headerLine.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        self.mainLayout.setVerticalSpacing(0)
        self.mainLayout.setContentsMargins(0, 0, 0, 0)
        self.mainLayout.addWidget(self.toggleButton, 0, 0, 1, 1, Qt.AlignLeft)
        self.mainLayout.addWidget(self.headerLine, 0, 2, 1, 1)
        self.mainLayout.addWidget(self.contentsWidget, 1, 0, 1, 3)
        self.setLayout(self.mainLayout)
        self.contentsWidget.setVisible(self.toggleButton.isChecked())

        self.toggleButton.clicked.connect(self.toggle)
        self.current_row = 0

    def toggle(self, checked):
        self.toggleButton.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.contentsWidget.setVisible(checked)

    def addWidget(self, *args, **kwargs):
        self.contentsLayout.addWidget(*args, **kwargs)

    def add_row(self, *args):
        for i, widget in enumerate(args):
            if type(widget) == str:
                widget = QLabel(widget)
            self.addWidget(widget, self.current_row, i)
        self.current_row += 1


class PlayButton(QPushButton):
    def __init__(self, timer, labels=["Play", "Pause"], parent=None):
        super().__init__(labels[0], parent=parent)
        self.timer = timer
        self.labels = labels
        self.clicked.connect(self.toggle)

    def toggle(self):
        if self.timer.isActive():
            self.timer.stop()
            self.setText(self.labels[0])
        else:
            self.timer.start()
            self.setText(self.labels[1])


class ViewerApp(QMainWindow):
    def __init__(self, window_name="Volumetric Viewer", volume=None):
        super().__init__()
        self.setWindowTitle(window_name)

        self.gl_display = ViewWidget(self, volume=volume)

        self.v_box = QVBoxLayout()
        self.v_box.setContentsMargins(0, 0, 0, 0)


        # for n in range(1, 4):
        #     # cb = QVBoxLayout()
        #     c = CollapsableVBox(self, title="Test Collapse %d" % n, isOpen=bool(n%2))
        #     # c.setContentLayout(cb)
        #
        #
        #     for n in range(1, 11):
        #         c.addWidget(QPushButton('Button %d' % n))
        #
        #     c.contentsWidget.setFixedWidth(200)
        #     self.v_box.addWidget(c)

        self.view_options = CollapsableVBox(self, "View Options", isOpen=True)
        self.v_box.addWidget(self.view_options)

        self.gl_display.frame_setting = IntViewSetting(self.gl_display, "frame", 0, 0, 0)
        self.view_options.add_row("Frame:", self.gl_display.frame_setting)

        self.frame_timer = QTimer()
        self.frame_timer.setSingleShot(False)
        self.frame_timer.setInterval(1000//30)
        self.frame_timer.timeout.connect(self.gl_display.frame_setting.advance)
        self.play_pause = PlayButton(self.frame_timer)
        self.view_options.add_row("Frame:", self.gl_display.frame_setting, self.play_pause)

        self.view_options.add_row("Cloud Opacity:",
            LogViewSetting(self.gl_display, "opacity", 0.1, 1E-3, 1, 10**(1/4)))

        self.view_options.add_row("Cloud Tint:",
            ColorViewSetting(self.gl_display, "tint", (0.0, 0.3, 0.3)))

        self.view_options.add_row("Show Isosurface:",
            BoolViewSetting(self.gl_display, "show_isosurface", True))

        self.view_options.add_row("Isosurface Level:",
            LinearViewSetting(self.gl_display, "iso_level", 0.5, 0.0, 1.0, 0.1))

        self.view_options.add_row("Isosurface Color:",
            ColorViewSetting(self.gl_display, "surface_color", (1.0, 0.0, 0.0, 0.5)))

        self.adv_view_options = CollapsableVBox(self, "Advanced View Options", isOpen=True)
        self.v_box.addWidget(self.adv_view_options)

        self.adv_view_options.add_row("Shine:",
            LinearViewSetting(self.gl_display, "shine", 0.2, 0, 1, 0.1))

        self.adv_view_options.add_row("Show Grid:",
            BoolViewSetting(self.gl_display, "show_grid", False))

        self.v_box.addStretch()

        self.v_scroll = QScrollArea()
        self.v_widget = QWidget()
        self.v_widget.setLayout(self.v_box)
        self.v_scroll.setFixedWidth(300)
        self.v_scroll.setWidget(self.v_widget)

        self.h_box = QHBoxLayout()
        self.h_box.addWidget(self.gl_display)
        self.h_box.addWidget(self.v_scroll)
        self.h_box.setContentsMargins(0, 0, 0, 0)
        self.h_widget = QWidget()
        self.h_widget.setLayout(self.h_box)

        # self.setLayout(self.h_box)

        self.setCentralWidget(self.h_widget)

        menu = self.menuBar()
        self.file_menu = menu.addMenu("File")

        quit_button = QAction('Quit', self)
        quit_button.setShortcut('Ctrl+Q')
        quit_button.setStatusTip('Quit')
        quit_button.triggered.connect(self.close)
        self.file_menu.addAction(quit_button)

        self.view_menu = menu.addMenu("View")
        show_button = QAction('Show Settings', self)
        show_button.setShortcut('Tab')
        show_button.setStatusTip('Show Settings')
        show_button.triggered.connect(self.toggle_view)
        self.view_menu.addAction(show_button)

        self.show()

    def toggle_view(self):
        self.v_scroll.setVisible(not self.v_scroll.isVisible())

    def keyPressEvent(self, event):
        """Close application from escape key.

        results in QMessageBox dialog from closeEvent, good but how/why?
        """
        if event.key() == Qt.Key_Escape:
            self.close()

def view_volume(vol, window_name="Volumetric Viewer"):
    app = QApplication(sys.argv)
    app.setApplicationDisplayName(window_name)
    ex = ViewerApp(volume=vol, window_name=window_name)
    return(app.exec_())
