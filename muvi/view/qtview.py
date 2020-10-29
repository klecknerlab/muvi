#!/usr/bin/python3
#
# Copyright 2020 Dustin Kleckner
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
import glob
import numpy as np

from .view import View

GAMMA_CORRECT = 2.2

ORG_NAME = "MUVI Lab"
APP_NAME = "QT viewer"

class BoolViewSetting(QCheckBox):
    def __init__(self, gl_display, varname, default, parent=None):
        super().__init__(None, parent)
        self.setChecked(default)
        self.stateChanged.connect(lambda state: gl_display.update_params(
            **{varname:(state==Qt.Checked)}))


class LinearViewSetting(QDoubleSpinBox):
    def __init__(self, gl_display, varname, default, minval, maxval, step, index=None, decimals=None, parent=None):
        super().__init__(parent)
        self.setRange(minval, maxval)
        self.setValue(default)
        self.setMaximum(maxval)
        self.setSingleStep(step)
        if decimals is None: decimals= math.ceil(0.5-math.log10(step))
        self.setDecimals(decimals)

        if index is not None:
            def change_func(val):
                par = gl_display.view.params[varname].copy()
                par[index] = val
                gl_display.update_params(**{varname:par})
        else:
            change_func = lambda val: gl_display.update_params(**{varname:val})

        self.valueChanged.connect(change_func)


    def set_from_slider(self, val, slider_max):
        minval = self.minumum()
        maxval = self.maximum()
        self.setValue(minval + (val/slider_max)*(maxval-minval))


class IntViewSetting(QSpinBox):
    def __init__(self, gl_display, varname, default, minval, maxval, rollover=True, parent=None, force_update=True):
        super().__init__(parent)
        self.setRange(minval, maxval)
        self.setValue(default)
        self.setMaximum(maxval)
        self.setSingleStep(1)
        self.rollover = rollover

        self.valueChanged.connect(lambda val: gl_display.update_params(
            **{varname:val, 'force_update':force_update}))

    def advance(self):
        maxval = self.maximum()
        if maxval != 0:
            i = self.value() + 1
            if self.rollover: i %= self.maximum()
            self.setValue(i)



class ListViewSetting(IntViewSetting):
    def __init__(self, gl_display, varname, default, vals, parent=None, force_update=True):
        super().__init__(gl_display, varname, default, min(vals), max(vals), rollover=False, parent=parent, force_update=force_update)
        self.vals = list(vals)

    def stepBy(self, step):
        if step > 0:
            self.setValue(list(filter(lambda x: x > self.value(), self.vals))[0])
        elif step < 0:
            self.setValue(list(filter(lambda x: x < self.value(), self.vals))[-1])


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


class OptionsViewSetting(QComboBox):
    def __init__(self, gl_display, varname, force_update=True, parent=None):
        super().__init__(None)

        default = gl_display.get(varname)
        self.gl_display = gl_display
        self.varname = varname
        self.options = []
        self.update_options(selected=default)

        self.currentIndexChanged.connect(lambda index: gl_display.update_params(
             **{varname:self.options[index], 'force_update':force_update}))


    def update_options(self, selected=None):
        if selected is None:
            selected = self.options[self.currentIndex()]
        # print(selected)

        self.clear()

        for i, (sn, ln) in enumerate(self.gl_display.get_options(self.varname).items()):
            self.addItem(ln)
            self.options.append(sn)

            if sn == selected:
                self.setCurrentIndex(i)


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
        self.gl_display.update_params(
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

        self.view = View(volume=self.volume, )

        # if hasattr(self, "frame_setting"):
        #     self.frame_setting.setMaximum(len(self.volume)-1)


    def minimumSizeHint(self):
        return QSize(300, 300)


    def sizeHint(self):
        return QSize(800, 600)


    def initializeGL(self):
        try:
            self.dpr = self.devicePixelRatio()
            self.view.resize(width=self.width()*self.dpr,
                height=self.height()*self.dpr)
        except:
            traceback.print_exc()
            self.parent.close()


    def paintGL(self):
        try:
            # if hasattr(self, 'view'):
            self.view.draw()
        except:
            traceback.print_exc()
            self.parent.close()


    def resizeGL(self, width, height):
        if hasattr(self, 'view'):
            self.view.resize(width*self.dpr, height*self.dpr)


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
        self.view.params['scale'] *= 1.25**(event.angleDelta().y()/120)
        self.update()


    def update_params(self, force_update=True, **kwargs):
        # print(kwargs)
        self.view.update_params(**kwargs)

        if force_update:
            self.update()


    def get_options(self, varname):
        return self.view.get_options(varname)


    def get(self, varname):
        return self.view.params[varname]


    def preview_image(self):
        self.makeCurrent()
        img = self.view.draw(offscreen=True, return_image=True)
        self.doneCurrent()

        return img

    def save_image(self, fn=None):
        # fn = 'test.png'
        if fn is None:
            fns = glob.glob('muvi_screenshot_*.png')
            for i in range(10**4):
                fn = 'muvi_screenshot_%08d.png' % i
                if fn not in fns:
                    break

        self.makeCurrent()
        img = self.view.draw(offscreen=True, save_image=fn)
        self.doneCurrent()

        return fn, img


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



class ExportWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent=parent, flags=Qt.Tool)
        self.setWindowTitle("Image Export")
        self.parent = parent

    def closeEvent(self, e):
        self.parent.toggle_export()


class ViewerApp(QMainWindow):
    def __init__(self, window_name="Volumetric Viewer", volume=None):
        super().__init__()
        self.setWindowTitle(window_name)

        self.gl_display = ViewWidget(self, volume=volume)

        self.v_box = QVBoxLayout()
        self.v_box.setContentsMargins(0, 0, 0, 0)

        self.settings = QSettings(ORG_NAME, APP_NAME)


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

        self.gl_display.frame_setting = IntViewSetting(self.gl_display, "frame", 0, 0, len(volume))
        # self.view_options.add_row("Frame:", self.gl_display.frame_setting)

        self.frame_timer = QTimer()
        self.frame_timer.setSingleShot(False)
        self.frame_timer.setInterval(1000//30)
        self.frame_timer.timeout.connect(self.gl_display.frame_setting.advance)
        self.play_pause = PlayButton(self.frame_timer)

        # self.view_options.add_row("Frame:")
        self.view_options.add_row(self.play_pause, self.gl_display.frame_setting)

        self.view_options.add_row("Exposure (stops)",
            LinearViewSetting(self.gl_display, "exposure", View.uniform_defaults['exposure'], -5, 5, 0.5))

        self.view_options.add_row("Cloud Opacity:",
            LogViewSetting(self.gl_display, "opacity", View.uniform_defaults['opacity'], 1E-3, 2, 10**(1/8)))

        # self.view_options.add_row("Cloud Tint:",
        #     ColorViewSetting(self.gl_display, "tint", (0.0, 0.3, 0.3)))

        self.view_options.add_row("Cloud Color:", OptionsViewSetting(self.gl_display, "cloud_color"))
        self.view_options.add_row("Colormap:", OptionsViewSetting(self.gl_display, "colormap"))

        self.view_options.add_row("Left Edge:", LinearViewSetting(self.gl_display, "X0", 0, 0, volume.shape[2], 64, index=0))
        self.view_options.add_row("Right Edge:", LinearViewSetting(self.gl_display, "X1", volume.shape[2], 0, volume.shape[2], 64, index=0))

        self.view_options.add_row("Bottom Edge:", LinearViewSetting(self.gl_display, "X0", 0, 0, volume.shape[1], 64, index=1))
        self.view_options.add_row("Top Edge:", LinearViewSetting(self.gl_display, "X1", volume.shape[1], 0, volume.shape[1], 64, index=1))

        self.view_options.add_row("Back Edge:", LinearViewSetting(self.gl_display, "X0", 0, 0, volume.shape[0], 64, index=2))
        self.view_options.add_row("Front Edge:", LinearViewSetting(self.gl_display, "X1", volume.shape[0], 0, volume.shape[0], 64, index=2))


        # self.view_options.add_row("Show Isosurface:",
        #     BoolViewSetting(self.gl_display, "show_isosurface", View.shader_defaults["show_isosurface"]))
        #
        # self.view_options.add_row("Isosurface Level:",
        #     LinearViewSetting(self.gl_display, "iso_offset", View.uniform_defaults['iso_offset'], 0.0, 1.0, 0.1))


        # self.view_options.add_row("Isosurface Color:",
        #     ColorViewSetting(self.gl_display, "surface_color", (1.0, 0.0, 0.0, 0.5)))

        self.adv_view_options = CollapsableVBox(self, "Advanced View Options", isOpen=True)
        self.v_box.addWidget(self.adv_view_options)

        # self.adv_view_options.add_row("Shine:",
        #     LinearViewSetting(self.gl_display, "shine", 0.2, 0, 1, 0.1))

        # self.adv_view_options.add_row("Show Grid:",
        #     BoolViewSetting(self.gl_display, "show_grid", False))

        self.adv_view_options.add_row("Render Step:",
            LogViewSetting(self.gl_display, "step_size", View.uniform_defaults['step_size'], 0.125, 2, 2**0.5))

        self.adv_view_options.add_row("Perspective X Coefficient",
            LinearViewSetting(self.gl_display, "perspective_xfact", View.uniform_defaults['perspective_xfact'], -0.25, 0.25, 0.01))

        self.adv_view_options.add_row("Perspective Z Coefficient",
            LinearViewSetting(self.gl_display, "perspective_zfact", View.uniform_defaults['perspective_zfact'], -0.25, 0.25, 0.01))


        self.v_box.addStretch()

        self.v_scroll = QScrollArea()
        self.v_widget = QWidget()
        self.v_widget.setLayout(self.v_box)
        # self.v_scroll.setSizeHint(int(self.settings.value("settings width", 300)))
        self.v_scroll.setWidget(self.v_widget)

        self.h_box = QHBoxLayout()
        self.h_box.addWidget(self.gl_display, 1)
        self.h_box.addWidget(self.v_scroll, 0)
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
        self.show_settings = QAction('Hide View Settings', self)
        self.show_settings.setShortcut('Tab')
        self.show_settings.setStatusTip('Show or hide settings option on right side of main window')
        self.show_settings.triggered.connect(self.toggle_settings)
        self.view_menu.addAction(self.show_settings)

        self.show_export = QAction('Show Export Window', self)
        self.show_export.setShortcut(QKeySequence('Ctrl+E'))
        self.show_export.setStatusTip('Show or hide the export window, used to take screenshots or make movies')
        self.show_export.triggered.connect(self.toggle_export)
        self.view_menu.addAction(self.show_export)


        self.export_window = ExportWindow(self)
        self.export_window_vbox = QVBoxLayout()

        e_widget = QWidget()
        e_widget.setLayout(self.export_window_vbox)
        self.export_window.setCentralWidget(e_widget)

        self.export_window_image = QLabel()
        # self.export_window_image.sizeHint(QSize(500, 500))
        self.export_window_vbox.addWidget(self.export_window_image)

        self.export_window_settings = QGridLayout()
        self.export_window_vbox.addLayout(self.export_window_settings)

        self.export_image = QPushButton("Export Current Frame")
        self.export_image.clicked.connect(self.save_frame)
        self.export_window_settings.addWidget(self.export_image, 1, 1, 1, 2)

        self.preview_image = QPushButton("Preview Current Frame")
        self.preview_image.clicked.connect(self.preview_frame)
        self.export_window_settings.addWidget(self.preview_image, 1, 3, 1, 2)

        # self.ss_button = QPushButton("Save View")
        # self.ss_button.clicked.connect(self.gl_display.save_image)
        # self.view_options.add_row(self.ss_button)

        self.ss_sizes = [256, 384, 512, 640, 768, 1024, 1080, 1280, 1920, 2048, 2160, 3072, 3840, 4096]
        self.export_window_settings.setColumnStretch(0, 1)
        self.export_window_settings.setColumnStretch(5, 1)
        self.export_window_settings.addWidget(QLabel("Export Width:"), 0, 1)
        self.export_window_settings.addWidget(ListViewSetting(self.gl_display, "os_width", 1920, self.ss_sizes, force_update=False), 0, 2)
        self.export_window_settings.addWidget(QLabel("Export Height:"), 0, 3)
        self.export_window_settings.addWidget(ListViewSetting(self.gl_display, "os_height", 1080, self.ss_sizes, force_update=False), 0, 4)
        self.export_window_status = self.export_window.statusBar()


        # self.view_options.add_row("Saved Width:",
        #     ListViewSetting(self.gl_display, "os_width", 1920, self.ss_sizes, force_update=False))
        # self.view_options.add_row("Saved Height:",
        #     ListViewSetting(self.gl_display, "os_height", 1080, self.ss_sizes, force_update=False))


        self.show()

    def toggle_settings(self):
        if self.v_scroll.isVisible():
            self.v_scroll.setVisible(False)
            self.show_settings.setText('Show View Settings')
        else:
            self.v_scroll.setVisible(True)
            self.show_settings.setText('Hide View Settings')


    def toggle_export(self):
        if self.export_window.isVisible():
            self.export_window.hide()
            self.show_export.setText('Show Export Window')
        else:
            self.export_window.show()
            self.show_export.setText('Hide Export Window')


    def save_frame(self, e):
        fn, img = self.gl_display.save_image()
        self.export_window_status.showMessage('Saved image: ' + fn)
        # print(img.shape, img.dtype)
        self.update_preview(img)

    def preview_frame(self, e):
        self.update_preview(self.gl_display.preview_image())



    def update_preview(self, img):
        img = QImage(np.require(img[::-1, :, :3], np.uint8, 'C'), img.shape[1], img.shape[0], QImage.Format_RGB888)
        self.export_window_image.setPixmap(QPixmap(img).scaledToWidth(1024))

    def keyPressEvent(self, event):
        """Close application from escape key.

        results in QMessageBox dialog from closeEvent, good but how/why?
        """
        if event.key() == Qt.Key_Escape:
            self.close()

def view_volume(vol, window_name="Volumetric Viewer"):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setApplicationDisplayName(window_name)
    ex = ViewerApp(volume=vol, window_name=window_name)
    return(app.exec_())
