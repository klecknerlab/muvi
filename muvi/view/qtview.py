#!/usr/bin/python3
#
# Copyright 2021 Dustin Kleckner
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

'''
This is the top-level module for creating a Qt based volumetric viewer
application.
'''

import sys, os
# from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget, QHBoxLayout, \
    QVBoxLayout, QLabel, QWidget, QScrollArea, QAction, QFrame, QMessageBox, \
    QFileDialog, QGridLayout, QPushButton, QStyle
from .qtview_widgets import param_list_to_vbox, control_from_param, \
    VolumetricView, ListControl
import numpy as np

from .params import PARAM_CATAGORIES, PARAMS, range_from_volume
from .. import open_3D_movie, VolumetricMovie

GAMMA_CORRECT = 2.2

ORG_NAME = "MUVI Lab"
APP_NAME = "MUVI Volumetric Movie Viewer"

if sys.platform.startswith('darwin'):
        # Python 3: pip3 install pyobjc-framework-Cocoa
        try:
            from Foundation import NSBundle
            bundle = NSBundle.mainBundle()
            if bundle:
                app_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
                app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
                if app_info:
                    app_info['CFBundleName'] = APP_NAME
        except ImportError:
            pass

class ExportWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent, flags=QtCore.Qt.Window)
        self.setWindowTitle("Image Export")
        self.parent = parent

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)

        self.image = QLabel()
        self.vbox.addWidget(self.image)

        self.settings = QGridLayout()
        self.vbox.addLayout(self.settings)

        self.export_button = QPushButton("Export Current Frame")
        self.export_button.clicked.connect(self.save_frame)
        self.settings.addWidget(self.export_button, 2, 1, 1, 2)

        self.preview_button = QPushButton("Preview Current Frame")
        self.preview_button.clicked.connect(self.preview_frame)
        self.settings.addWidget(self.preview_button, 2, 3, 1, 2)

        halign = QHBoxLayout()
        self.folder_label = QLabel(os.getcwd())
        self.folder_button = QPushButton()
        self.folder_button.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.folder_button.clicked.connect(self.select_export_folder)
        halign.addWidget(self.folder_button, 0)
        self.settings.addLayout(halign, 3, 1, 1, 4)
        halign.addWidget(self.folder_label, 1)

        self.file_label = QLabel("")
        self.file_label.setFixedWidth(512)
        self.settings.addWidget(self.file_label, 4, 1, 1, 4)

        self.ss_sizes = [256, 384, 512, 640, 720, 768, 1024, 1080, 1280, 1920,
            2048, 2160, 3072, 3840, 4096, 6144, 8192]
        self.width_control = ListControl('Width:', 1920, self.ss_sizes, param='os_width')
        self.height_control = ListControl('Height:', 1080, self.ss_sizes, param='os_height')

        def print_param(p, v):
            print(f'{p}: {v}')
        self.width_control.paramChanged.connect(self.parent.display.updateHiddenParam)
        self.height_control.paramChanged.connect(self.parent.display.updateHiddenParam)

        self.settings.setColumnStretch(0, 1)
        self.settings.setColumnStretch(5, 1)
        self.settings.addWidget(self.width_control, 0, 1, 1, 2)
        self.settings.addWidget(self.height_control, 0, 3, 1, 2)

        for i, (label, w, h) in enumerate([
                    ('720p', 1280, 720),
                    ('1080p', 1920, 1080),
                    ('1440p', 2560, 1440),
                    ('2160p (4K)', 3840, 2160),
                ]):
            button = QPushButton(label)

            def cr(state, w=w, h=h):
                self.width_control.setValue(w)
                self.height_control.setValue(h)

            button.clicked.connect(cr)
            self.settings.addWidget(button, 1, i+1)

    def closeEvent(self, e):
        self.parent.toggle_export()

    def select_export_folder(self):
        self.folder_label.setText(QFileDialog.getExistingDirectory(
            self, "Select Export Folder", self.folder_label.text()))

    def save_frame(self, event=None):
        fn, img = self.parent.display.saveImage(
            dir=self.folder_label.text())
        # self.export_window_status.showMessage('Saved image: ' + fn)
        # print(img.shape, img.dtype)
        self.update_preview(img)
        self.file_label.setText(f'Saved to: {os.path.split(fn)[1]}')

    def preview_frame(self, event=None):
        self.update_preview(self.parent.display.previewImage())

    def update_preview(self, img):
        img = QtGui.QImage(np.require(img[::-1, :, :3], np.uint8, 'C'),
            img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.setPixmap(QtGui.QPixmap(img).scaledToWidth(1024))


class VolumetricViewer(QMainWindow):
    def __init__(self, volume=None, parent=None, window_name=None):
        super().__init__(parent)

        self.setWindowTitle(window_name)

        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)

        self.hbox = QHBoxLayout()
        self.display = VolumetricView(parent=self)
        self.hbox.addWidget(self.display, 1)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setSpacing(0)

        self.vbox.addLayout(self.hbox, 1)
        self.playback = control_from_param(PARAMS['frame'])
        self.vbox.addWidget(self.playback)
        self.playback.playingChanged.connect(self.display.setPlaying)
        self.display.frameChanged.connect(self.playback.setSilent)

        self.param_controls = {'frame': self.playback}

        self.param_tabs = QTabWidget()
        self.param_tabs.setFixedWidth(230)

        for cat, params in PARAM_CATAGORIES.items():
            if cat == 'Playback': continue

            vbox = QVBoxLayout()
            # vbox.setContentsMargins(5, 5, 5, 5)
            vbox.setSpacing(10)
            self.param_controls.update(param_list_to_vbox(params, vbox))

            vbox.addStretch(1)

            sa = QScrollArea()
            sa.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
            sa.setContentsMargins(0, 0, 0, 0)
            sa.setFrameShape(QFrame.NoFrame)

            widget = QWidget()
            widget.setLayout(vbox)
            widget.setFixedWidth(210)
            sa.setWidget(widget)
            self.param_tabs.addTab(sa, cat)

        self.hbox.addWidget(self.param_tabs)

        widget = QWidget()
        widget.setLayout(self.vbox)
        self.setCentralWidget(widget)

        self.setWindowTitle(APP_NAME)
        # self.resize(800, 600)

        for param, control in self.param_controls.items():
            if hasattr(control, 'paramChanged'):
                # control.paramChanged.connect(lambda p, v: print(p, v))
                control.paramChanged.connect(self.display.updateParam)


        self.export_window = ExportWindow(self)

        menu = self.menuBar()
        self.file_menu = menu.addMenu("File")

        self.add_menu_item(self.file_menu, 'Quit', self.close, 'Ctrl+Q',
            'Quit the viewer.')
        self.add_menu_item(self.file_menu, '&Open Volumetric Movie',
            self.open_file, 'Ctrl+O')

        self.view_menu = menu.addMenu("View")

        self.show_settings = self.add_menu_item(self.view_menu,
            'Hide View Settings', self.toggle_settings, 'Ctrl+/',
            'Show or hide settings option on right side of main window')

        self.save_image = self.add_menu_item(self.view_menu,
            'Save Screenshot', self.export_window.save_frame, 's',
            'Save a screenshot with the current export settings (use export window to control resolution).')

        self.show_export = self.add_menu_item(self.view_menu,
            'Show Export Window', self.toggle_export, 'Ctrl+E',
            'Show or hide the export window, used to take screenshots or make movies')

        for i in range(3):
            axis = chr(ord('X') + i)

            def f(event, a=i):
                self.orient_camera(a)
            self.add_menu_item(self.view_menu,
                f'Look down {axis}-axis', f, axis.lower())

            def f2(event, a=i):
                self.orient_camera(a+3)
            self.add_menu_item(self.view_menu,
                f'Look down -{axis}-axis', f2, 'Shift+'+axis.lower())


        self.setAcceptDrops(True)
        self.show()
        if volume is not None:
            self.open_volume(volume)

    def refresh_param_values(self):
        if getattr(self.display, 'volume', None) is not None:
            vol = self.display.volume
            # self.param_controls['frame'].setRange(0, len(vol)-1)
            ranges = range_from_volume(vol)
            for param, r in ranges.items():
                if param in self.param_controls:
                    self.param_controls[param].setRange(*r)

        for name, control in self.param_controls.items():
            control.setValue(self.display.view.params[name])


    def add_menu_item(self, menu, title, func=None, shortcut=None, tooltip=None):
        action = QAction(title, self)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tooltip is not None:
            action.setStatusTip(tooltip)
            action.setToolTip(tooltip)
        if func is not None:
            action.triggered.connect(func)
        menu.addAction(action)

        return action

    def open_file(self):
        try:
            fn, ext = QFileDialog.getOpenFileName(self, 'Open Volumetric Movie', os.getcwd(), "VTI (*.vti)")
            if fn:
                vol = open_3D_movie(fn)
        except:
            QMessageBox.critical(self, "ERROR", f'"{fn}" does not appear to be a 3D movie file!')
        else:
            if fn:
                self.open_volume(vol)

    def open_volume(self, vol):
        if isinstance(vol, str):
            try:
                vv = open_3D_movie(vol)
            except Exception as e:
                ec = e.__class__.__name__
                QMessageBox.critical(self, ec, str(e))
                return
            else:
                self.setWindowTitle(vol)
                vol = vv
        elif isinstance(vol, np.ndarray):
            vol = VolumetricMovie(vol)

        if isinstance(vol, VolumetricMovie):
            self.display.attachVolume(vol)
            self.refresh_param_values()

        else:
            raise ValueError('Volume must be a VolumetricMovie object or convertable to one (a filename or numpy array)')

    def toggle_settings(self):
        if self.param_tabs.isVisible():
            self.param_tabs.setVisible(False)
            self.show_settings.setText('Show View Settings')
        else:
            self.param_tabs.setVisible(True)
            self.show_settings.setText('Hide View Settings')

    def toggle_export(self):
        if self.export_window.isVisible():
            self.export_window.hide()
            self.show_export.setText('Show Export Window')
        else:
            self.export_window.show()
            self.show_export.setText('Hide Export Window')

    def closeEvent(self, event):
        # Prevents an error message by controlling deallocation order!
        del self.display.view

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            l = []
            if len(event.mimeData().urls()):
                self.open_volume(event.mimeData().urls()[0].toLocalFile())
        else:
            event.ignore()

    def orient_camera(self, axis):
        X, Y, Z = np.eye(3)

        # R = np.eye(3)
        # if flip:
        #     R[0] *= -1
        #     R[2] *= -1
        #
        # R = np.roll(R, 2-axis, axis=1)
        # print(R, axis, flip)
        #
        # self.display.updateParams(R=R)

        if axis == 0:
            R = np.array([-Z, Y, X]).T
        elif axis == 1:
            R = np.array([X, -Z, Y]).T
        elif axis == 2:
            R = np.array([X, Y, Z]).T
        elif axis == 3:
            R = np.array([Z, Y, -X]).T
        elif axis == 4:
            R = np.array([X, Z, -Y]).T
        else:
            R = np.array([-X, Y, -Z]).T

        self.display.updateParams(R=R)

def view_volume(vol=None, args=None, window_name=None):
    if window_name is None:
        window_name = APP_NAME

    if args is None:
        args = sys.argv

    app = QApplication(args)
    app.setStyle('Fusion')
    app.setStyleSheet('''
        QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox {
            padding: 0px;
            margin: 0px;
        }
        QGroupBox {
            padding: 0px;
            padding-top: 20px;
            margin: 0px;
        }
        #SectionLabel {
            padding-top: 10px;
            /* font-weight: bold; */
        }
        QScrollBar:vertical {
            width: 15px;
        }
    ''')
    app.setApplicationDisplayName(window_name)
    window = VolumetricViewer(vol, window_name=window_name)
    app.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(os.path.split(__file__)[0], 'muvi_logo.png'))))
    return(app.exec_())


def qt_viewer(args=None, window_name=None):
    if args is None:
        args = sys.argv

    if len(args) > 1:
        vol = args.pop(1)
    else:
        vol = None

    view_volume(vol, args, window_name)


if __name__ == '__main__':
    qt_viewer()
