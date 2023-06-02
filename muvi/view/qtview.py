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
Qt = QtCore.Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QTabWidget, QHBoxLayout, \
    QVBoxLayout, QLabel, QWidget, QScrollArea, QAction, QFrame, QMessageBox, \
    QFileDialog, QGridLayout, QPushButton, QStyle, QSplitter, QMenu, \
    QSizePolicy, QProgressBar
from .qt_script import KeyframeEditor
from .qtview_widgets import paramListToVBox, controlFromParam, ListControl, \
    IntControl, ViewWidget, AssetList, AssetItem, generateDarkPalette, \
    BoolControl, LinearControl
import time
import traceback
import glob
from PIL import Image

from .params import PARAM_CATEGORIES, PARAMS, THEMES
# from .. import open_3D_movie, VolumetricMovie

ORG_NAME = "MUVI Lab"
APP_NAME = "MUVI Volumetric Movie Viewer"
ICON_DIR = os.path.split(__file__)[0]

PARAM_WIDTH = 250
SCROLL_WIDTH = 15
LOGICAL_DPI_BASELINE = None
UI_EXTRA_SCALING = 1.0

if sys.platform == 'win32':
    # On Windows, it appears to need a bit more width to display text
    # We need to play some games to get the icon to show up correctly!
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MuviLab.Viewer")
    LOGICAL_DPI_BASELINE = 96 #Used to correct fractional scaling, which does not show up in DPR!

elif sys.platform == 'darwin':
    # Python 3: pip3 install pyobjc-framework-Cocoa
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        if bundle:
            app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if app_info:
                app_info['CFBundleName'] = APP_NAME
                app_info['NSRequiresAquaSystemAppearance'] = 'NO'
    except ImportError:
        raise ImportError('pyobjc-framework-Cocoa not installed (OS X only) -- run "pip3 install pyobjc-framework-Cocoa" or "conda install pyobjc-framework-Cocoa" first')

    # OS X always does integer high DPI scaling, so no need to check logical DPI

class ImageDisplay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.scaleFactor = 1
        self.dpr = self.devicePixelRatio()

    def paintEvent(self, event=None):
        if self.image is not None:
            qp = QtGui.QPainter()
            qp.begin(self)
            qp.drawPixmap(0, 0, self.w * self.scale, self.h * self.scale, self.image, 0, 0, self.w, self.h)
            qp.end()

    def adjustSize(self, event=None):
        if self.image is not None:
            self.scale = self.scaleFactor / self.dpr
            self.setFixedSize(int(self.w*self.scale), int(self.h*self.scale))
            self.update()

    def zoomIn(self, event=None):
        self.setScaleFactor(self.scaleFactor * 2)

    def zoomOut(self, event=None):
        self.setScaleFactor(self.scaleFactor / 2)

    def zoomFit(self, event=None):
        if self.image is not None:
            parentSize = self.parent().size()
            w, h = parentSize.width(), parentSize.height()
            sf = min(w / self.w * self.dpr, h / self.h * self.dpr)
            # print(w, h, sf)
            self.setScaleFactor(sf)

    def setScaleFactor(self, factor):
        self.scaleFactor = factor
        self.adjustSize()

    def setImage(self, image):
        self.image =  QtGui.QPixmap(image)
        size = self.image.size()
        self.w, self.h = size.width(), size.height()
        self.adjustSize()


# class ExportWorker(QtCore.QObject):
#     finished = QtCore.pyqtSignal()
#     progress = QtCore.pyqtSignal(int)
#     progressRange = QtCore.pyqtSignal(int, int)
#
#     def __init__(self, parent):
#         super().__init__() # No parent!
#         self.parent = parent
#         self.queue = []
#
#     def run(self):
#         print('start')
#         for n in range(10):
#             time.sleep(0.1)
#
#         for fn, params in self.queue:
#             img = self.parent.renderImage()
#             img.save(fn)
#             self.parent.updatePreview(img)
#             self.parent.fileLabel.setText(f'Saved to: {os.path.split(fn)[1]}')
#
#         self.queue = []
#
#         # self.progressRange.emit(0, 1)
#
#         self.progressRange.emit(0, 1)
#         self.finished.emit()
#         print('end')
#
#     def exportSingle(self, fn):
#         self.queue = [(fn, {})]
#         self.progressRange.emit(0, 0)
#         # self.parent.progress.repaint()
#         # self.progress.emit(5)
#         self.parent.exportThread.start()


class ExportWindow(QWidget):
    exportFrame = QtCore.pyqtSignal()

    def __init__(self, parent):
        super().__init__(parent=parent, flags=Qt.Window)
        self.setWindowTitle("Image Export")
        self.parent = parent

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)

        self.scrollArea = QScrollArea()
        self.scrollArea.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        self.image = ImageDisplay(self.scrollArea)
        self.scrollArea.setWidget(self.image)
        self.scrollArea.setMinimumSize(1920//3, 1080//3)
        self.vbox.addWidget(self.scrollArea)

        self.zoomOut = QPushButton("Zoom Out")
        self.zoomOut.clicked.connect(self.image.zoomOut)
        self.zoomFit = QPushButton("Fit")
        self.zoomFit.clicked.connect(self.image.zoomFit)
        self.zoomIn = QPushButton("Zoom In")
        self.zoomIn.clicked.connect(self.image.zoomIn)
        self.hbox = QHBoxLayout()
        self.hbox.addStretch(1)
        self.hbox.addWidget(self.zoomOut)
        self.hbox.addWidget(self.zoomFit)
        self.hbox.addWidget(self.zoomIn)
        self.vbox.addLayout(self.hbox)
        self.hbox.addStretch(1)

        self.exportButton = QPushButton("Export Current Frame")
        self.exportButton.clicked.connect(self.saveFrame)

        self.previewButton = QPushButton("Preview Current Frame")
        self.previewButton.clicked.connect(self.previewFrame)

        self.folderControl = QHBoxLayout()
        self.folderLabel = QLabel(os.path.abspath(os.getcwd()))
        self.folderButton = QPushButton()
        self.folderButton.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.folderButton.clicked.connect(self.selectExportFolder)
        self.folderControl.addWidget(self.folderButton, 0)
        self.folderControl.addWidget(self.folderLabel, 1)

        self.movieButton = QPushButton("Export Movie Frames")
        self.movieButton.clicked.connect(self.exportMovie)

        self.fileLabel = QLabel("")
        self.fileLabel.setFixedWidth(512)

        self.ss_sizes = [512, 640, 720, 768, 1024, 1080, 1280, 1920,
            2048, 2160, 3072, 3240, 3840, 4096]
        self.widthControl = ListControl('Width:', 1920, self.ss_sizes, param='os_width')
        self.heightControl = ListControl('Height:', 1080, self.ss_sizes, param='os_height')
        self.scaleControl = ListControl('Scale Height:', 1080, self.ss_sizes, param='scaling_height',
            tooltip='Effective screen height to use for axis scaling.  Used to prevent super thin lines and tiny text for high resolutions!')
        self.oversampleControl = IntControl('Oversample', 1, 1, 3, step=1, param='os_oversample',
            tooltip='If > 1, render at a higher resolution and then downsample.\bThis will make export (much) slower, but is useful for publication-quality images.')

        self.printWidthControl = LinearControl('Width (mm):', 80.0, 0.0, 500.0, step=10, decimals=1, param='osp_width')
        self.printHeightControl = LinearControl('Height (mm):', 80.0, 0.0, 500.0, step=10, decimals=1, param='osp_height')
        self.dpiControl = IntControl('DPI:', 300, 0, 1200, step=100, param='osp_dpi',
            tooltip="Dots per inch in output: typically 300-600 for publication.")
        self.printOversampleControl = IntControl('Oversample', 1, 1, 3, step=1, param='osp_oversample',
            tooltip='If > 1, render at a higher resolution and then downsample.\bThis will make export (much) slower, but is useful for publication-quality images.')
        self.printSizeLabel = QLabel()
        self.printDPILabel = QLabel("(Line/font sizes in pt)")

        self.printWidthControl.paramChanged.connect(self.updatePrintResolution)
        self.printHeightControl.paramChanged.connect(self.updatePrintResolution)
        self.dpiControl.paramChanged.connect(self.updatePrintResolution)

        # def print_param(p, v):
        #     print(f'{p}: {v}')
        # self.widthControl.paramChanged.connect(self.updateBuffer)
        # self.heightControl.paramChanged.connect(self.updateBuffer)
        # self.oversampleControl.paramChanged.connect(self.updateBuffer)

        # self.scaleHeight = self.scaleControl.value()
        # self.scaleControl.paramChanged.connect(self.updatescaleHeight)
        self.progress = QProgressBar()
        self.pauseButton = QPushButton()
        self.pauseButton.setEnabled(False)

        self.transparentControl = BoolControl('Transparent export:', False, param='transparent_export')
        self.bgTransparentControl = BoolControl('Force transparent axis background:', False, param='bg_transparent')


        # self.exportWorker = ExportWorker(self)
        # self.exportThread = QtCore.QThread()
        # self.exportWorker.progressRange.connect(self.progress.setRange)
        # self.exportWorker.progress.connect(self.progress.setValue)
        # self.exportWorker.moveToThread(self.exportThread)
        # self.exportWorker.finished.connect(self.exportThread.quit)
        # self.exportThread.started.connect(self.exportWorker.run)

        self.queue = []
        # Why this strangeness?  By triggering exportFrame through the signals
        #   method, we give a chance for other updates to happen!
        # self.exportFrame.connect(self._exportFrame)
        self.exportTimer = QtCore.QTimer()
        # This is how long we wait to export each frame.
        # Nominally, this gives time for the progressbar to update!
        # Theoretically, a timer of 1 should be ok, but that doesn't actually
        #   seem to work like I expect...
        self.exportTimer.setInterval(35)
        self.exportTimer.timeout.connect(self._exportFrame)

        self.settings = QGridLayout()
        self.vbox.addLayout(self.settings)
        self.settings.setColumnStretch(0, 1)
        self.settings.setColumnStretch(5, 1)

        self.exportTabs = QTabWidget()

        self.imageTabLayout = QGridLayout()
        self.imageTab = QWidget()
        self.imageTab.setLayout(self.imageTabLayout)
        self.exportTabs.addTab(self.imageTab, 'Image')

        self.imageTabLayout.addWidget(self.widthControl,      0, 1, 1, 2)
        self.imageTabLayout.addWidget(self.heightControl,     0, 3, 1, 2)
        self.imageTabLayout.addWidget(self.scaleControl,      1, 1, 1, 2)
        self.imageTabLayout.addWidget(self.oversampleControl, 1, 3, 1, 2)

        for i, (label, w, h) in enumerate([
                    ('720p', 1280, 720),
                    ('1080p', 1920, 1080),
                    ('1440p', 2560, 1440),
                    ('2160p (4K)', 3840, 2160),
                ]):
            button = QPushButton(label)

            def cr(state, w=w, h=h):
                self.widthControl.setValue(w)
                self.heightControl.setValue(h)

            button.clicked.connect(cr)
            j = i+1
            self.imageTabLayout.addWidget(button, 2, j)

        self.printTabLayout = QGridLayout()
        self.printTab = QWidget()
        self.printTab.setLayout(self.printTabLayout)
        self.exportTabs.addTab(self.printTab, 'Print')

        self.printTabLayout.addWidget(self.printWidthControl,      0, 1, 1, 2)
        self.printTabLayout.addWidget(self.printHeightControl,     0, 3, 1, 2)
        self.printTabLayout.addWidget(self.dpiControl,             1, 1, 1, 2)
        self.printTabLayout.addWidget(self.printOversampleControl, 1, 3, 1, 2)
        self.printTabLayout.addWidget(self.printSizeLabel,        2, 1, 1, 2)
        self.printTabLayout.addWidget(self.printDPILabel,       2, 3, 1, 2)
        self.updatePrintResolution()

        self.settings.addWidget(self.exportTabs, 0, 1, 3, 4)

        # Transparent export disabled -- it does not give good results!
        # The issue is the way alpha blending is handled post-facto, which
        #  does not deal with gamma correction properly.
        # self.settings.addWidget(self.transparentControl, 3, 1, 1, 2)
        # self.settings.addWidget(self.bgTransparentControl, 3, 3, 1, 2)
        self.settings.addWidget(self.exportButton,      3, 1, 1, 2)
        self.settings.addWidget(self.previewButton,     3, 3, 1, 2)
        self.settings.addLayout(self.folderControl,     4, 1, 1, 4)
        self.settings.addWidget(self.movieButton,       5, 1, 1, 4)
        self.settings.addWidget(self.fileLabel,         6, 1, 1, 4)
        self.settings.addWidget(self.progress,          7, 0, 1, 5)
        self.settings.addWidget(self.pauseButton,       7, 5, 1, 1)

    # def updatescaleHeight(self, key, val):
    #     self.scaleHeight = val
    #
    # def updateBuffer(self, key=None, val=None):
    #     width, height = self.widthControl.value(), self.heightControl.value()
    #     oversample = self.oversampleControl.value()
    #
    #     self.parent.display.makeCurrent()
    #     if not hasattr(self, 'bufferId'):
    #         self.bufferId = self.parent.display.view.addBuffer(width * oversample, height * oversample)
    #     else:
    #         self.parent.display.view.resizeBuffer(self.bufferId, width * oversample, height * oversample)
    #     self.parent.display.doneCurrent()

    def updatePrintResolution(self, e=None):
        dpi = self.dpiControl.value()
        self.printWidth = int(self.printWidthControl.value() / 25.4 * dpi + 0.5)
        self.printHeight = int(self.printHeightControl.value() / 25.4 * dpi + 0.5)
        self.printScaleHeight = self.printHeightControl.value() / 25.4 * 72 # Height of output in points
        self.printSizeLabel.setText(f'Actual Image Size: {self.printWidth}×{self.printHeight}')

    def closeEvent(self, e):
        self.parent.toggleExport()

    def selectExportFolder(self):
        self.folderLabel.setText(QFileDialog.getExistingDirectory(
            self, "Select Export Folder", self.folderLabel.text()))

    # def renderImage(self):
    #     # self.progress.setRange(0, 0)
    #     # self.progress.update()
    #     # print('start')
    #     # QApplication.processEvents()
    #
    #     if not hasattr(self, 'bufferId'):
    #         self.updateBuffer()
    #
    #     img = self.parent.display.offscreenRender(self.bufferId, scaleHeight=self.scaleHeight)
    #     img = Image.fromarray(img[::-1])
    #
    #     oversample = self.oversampleControl.value()
    #     if oversample > 1:
    #         w, h = img.size
    #         img = img.resize((w//oversample, h//oversample), Image.LANCZOS)
    #     # self.progress.setRange(0, 1)
    #
    #     return img

    def exportMovie(self, event=None):
        dir = os.path.join(self.folderLabel.text(), 'muvi_frames')
        if not os.path.exists(dir):
            os.makedirs(dir)

        frames = self.parent.keyframeEditor.frames()
        # print(frames)
        for frame, params in enumerate(frames):
            fn = os.path.join(dir, f'frame{frame:08d}.png')
            self.queue.append((fn, params))

        # This triggers an update w/o drawing anything!
        self.queue.append((False, self.parent.allParams()))

        self.progress.setRange(0, len(frames))
        self.progress.setValue(0)
        self.startExport()

    def startExport(self):
        self.exportButton.setDisabled(True)
        self.previewButton.setDisabled(True)
        self.movieButton.setDisabled(True)
        self.parent.display.disable()

        self.exportTimer.start()

    def _exportFrame(self):
        if not self.queue:
            return

        QApplication.processEvents()

        fn, updates = self.queue.pop(0)

        # if not hasattr(self, 'bufferId'):
        #     self.updateBuffer()

        if updates:
            self.parent.display.updateParams(updates)

        if fn is not False:
            tab = self.exportTabs.currentIndex()
            if tab == 1: # Print tab
                width = self.printWidth
                height = self.printHeight
                dpi = self.dpiControl.value()
                scaleHeight = self.printScaleHeight
                oversample = self.printOversampleControl.value()
            else: # Image tab
                width = self.widthControl.value()
                height = self.heightControl.value()
                scaleHeight = self.scaleControl.value()
                oversample = self.oversampleControl.value()
                dpi = 256

            img = Image.fromarray(self.parent.display.offscreenRender(
                width, height, oversample,
                transparent = self.transparentControl.value(),
                bgTransparent = self.bgTransparentControl.value(),
                scaleHeight=scaleHeight
            ))
            # img = self.parent.display.offscreenRender(self.bufferId, scaleHeight=self.scaleHeight)
            # img = Image.fromarray(img[::-1])

            # oversample = self.oversampleControl.value()
            # if oversample > 1:
            #     w, h = img.size
            #     img = img.resize((w//oversample, h//oversample), Image.LANCZOS)

            if fn is not None:
                img.save(fn, dpi=(dpi, dpi))
                self.fileLabel.setText(f'Saved to: {os.path.split(fn)[1]}')

            img = QtGui.QImage(img.tobytes("raw", "RGB"), img.size[0], img.size[1],
                QtGui.QImage.Format_RGB888)
            self.image.setImage(img)

        if self.queue:
            self.progress.setValue(self.progress.value() + 1)
            self.progress.repaint()
        else:
            self.exportButton.setEnabled(True)
            self.previewButton.setEnabled(True)
            self.movieButton.setEnabled(True)
            self.exportTimer.stop()
            self.progress.setRange(0, 1)
            self.progress.reset()
            self.parent.display.enable()

    def saveFrame(self, event=None):
        dir = self.folderLabel.text()
        fns = glob.glob(os.path.join(dir, 'muvi_screenshot_*.png'))
        for i in range(10**4):
            fn = os.path.join(dir, 'muvi_screenshot_%08d.png' % i)
            if fn not in fns:
                break
        # self.exportWorker.exportSingle(os.path.join(dir, fn))
        self.progress.setRange(0, 0)
        self.progress.repaint()
        self.queue.append((fn, {}))
        self.startExport()
        # self.exportFrame.emit()

    def previewFrame(self, event=None):
        self.progress.setRange(0, 0)
        self.progress.repaint()
        self.queue.append((None, {}))
        self.startExport()
        # self.exportFrame.emit()
        # self.updatePreview(self.renderImage())

    # def updatePreview(self, img):
    #     img = QtGui.QImage(img.tobytes("raw", "RGB"), img.size[0], img.size[1],
    #         QtGui.QImage.Format_RGB888)
    #     # img = QtGui.QImage(np.require(img[::-1, :, :3], np.uint8, 'C'),
    #     #     img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
    #     # self.image.setPixmap(QtGui.QPixmap(img).scaledToWidth(1024))
    #     # pixmap = QtGui.QPixmap(img)
    #     # pixmap.setDevicePixelRatio(1)
    #     self.image.setImage(img)
    #     # self.image.adjustSize()



class VolumetricViewer(QMainWindow):
    def __init__(self, parent=None, clipboard=None, window_name=None):
        super().__init__(parent)

        self.setWindowTitle(window_name)

        self.clipboard = clipboard
        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)

        self.keyframeEditor = KeyframeEditor(self)
        self.keyframeEditor.setFixedWidth(PARAM_WIDTH)
        self.keyframeEditor.setVisible(False)

        self.hbox = QHBoxLayout()
        self.display = ViewWidget(parent=self, uiScale=UI_EXTRA_SCALING)
        self.hbox.addWidget(self.keyframeEditor)
        self.hbox.addWidget(self.display, 1)
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.setSpacing(0)

        self.vbox.addLayout(self.hbox, 1)
        self.playback = controlFromParam(PARAMS['frame'])
        self.vbox.addWidget(self.playback)

        self.playback.playingChanged.connect(self.display.setPlaying)
        self.display.frameChanged.connect(self.playback.setSilent)
        self.playback.paramChanged.connect(self.display.updateParam)

        self.paramControls = {'frame': self.playback}

        self.paramTabs = QTabWidget()
        self.paramTabs.setFixedWidth(PARAM_WIDTH)
        self.numTabs = 0

        self.nullTab = QWidget()
        layout = QVBoxLayout()
        button = QPushButton('Open Data File')
        button.pressed.connect(self.openFile)
        layout.addWidget(button)
        layout.addStretch(1)
        self.nullTab.setLayout(layout)
        self.paramTabs.addTab(self.nullTab, 'Data')

        # for cat, params in PARAM_CATEGORIES.items():
            # if cat != 'Playback':
        for cat in ["Limits", "View", "Display"]:
            params = PARAM_CATEGORIES[cat]
            self.paramTabs.addTab(self.buildParamTab(params), cat)

        self.paramTabs.setObjectName('paramTabs')
        # self.paramTabs.setStyleSheet(f'''
        #     #paramTabs::QTabBar::tab {{
        #         width: {(PARAM_WIDTH-SCROLL_WIDTH)/self.paramTabs.count()}px;
        #         padding: 5px 0px 5px 0px;
        #         background-color: red;
        #     }}
        # ''')
        # self.paramTabs.setTabShape(QTabWidget.Rounded)

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setHandleWidth(15)

        self.assetVBox = QVBoxLayout()
        self.assetVBox.setContentsMargins(5, 5, 5, 0)
        self.assetList = AssetList(self)
        self.assetVBox.addWidget(self.assetList)


        # button = QPushButton('Print all params')
        # button.clicked.connect(self.allParams)
        # self.assetVBox.addWidget(button)

        resetView = QPushButton('Recenter/Rescale View')
        resetView.clicked.connect(self.display.resetView)
        self.assetVBox.addWidget(resetView)

        self.addParamCategory('Asset List', self.assetVBox)

        widget = QWidget()
        widget.setLayout(self.assetVBox)

        self.splitter.addWidget(widget)
        self.splitter.addWidget(self.paramTabs)
        self.splitter.setSizes([100, 200])
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)

        self.hbox.addWidget(self.splitter)

        widget = QWidget()
        widget.setLayout(self.vbox)
        self.setCentralWidget(widget)

        self.setWindowTitle(APP_NAME)

        self.exportWindow = ExportWindow(self)

        menu = self.menuBar()
        self.fileMenu = menu.addMenu("File")

        self.addMenuItem(self.fileMenu, 'Quit', self.close, 'Ctrl+Q',
            'Quit the viewer.')
        self.addMenuItem(self.fileMenu, '&Open Data',
            self.openFile, 'Ctrl+O')
        self.addMenuItem(self.fileMenu, '&Save Script File',
            self.keyframeEditor.saveScript, 'Ctrl+S')

        self.editMenu = menu.addMenu("Edit")
        self.addMenuItem(self.editMenu, 'Insert &Keyframe',
            self.addKeyframe, "Ctrl+K")

        self.viewMenu = menu.addMenu("View")

        self.showSettings = self.addMenuItem(self.viewMenu,
            'Hide View Settings', self.toggleSettings, 'Ctrl+/',
            'Show or hide settings option on right side of main window')

        self.showKeyframeEditor = self.addMenuItem(self.viewMenu,
            'Hide Keyframe List', self.toggleKeyframe, 'Ctrl+L',
            'Show or hide keyframe list on right side of main window')

        self.save_image = self.addMenuItem(self.viewMenu,
            'Save Screenshot', self.exportWindow.saveFrame, 's',
            'Save a screenshot with the current export settings (use export window to control resolution).')

        self.showExport = self.addMenuItem(self.viewMenu,
            'Show Export Window', self.toggleExport, 'Ctrl+E',
            'Show or hide the export window, used to take screenshots or make movies')

        self.addMenuItem(self.viewMenu, 'Match Aspect Ratio to Export', self.matchAspect, "Ctrl+A",
            tooltip="Adjust aspect ratio of main display to match export size; useful for previewing movies!")

        self.paramMenu = menu.addMenu("Parameters")

        self.orientMenu = self.viewMenu.addMenu('Orientation')

        for i in range(3):
            axis = chr(ord('X') + i)

            def f(event, a=i):
                self.orient_camera(a)

            self.addMenuItem(self.orientMenu,
                f'Look down {axis}-axis', f, axis.lower())

            def f2(event, a=i):
                self.orient_camera(a+3)

            self.addMenuItem(self.orientMenu,
                f'Look down -{axis}-axis', f2, 'Shift+'+axis.lower())

        self.themeMenu = self.viewMenu.addMenu('Theme')

        for name in THEMES:
            def f(event, theme=name):
                self.setTheme(theme)

            self.addMenuItem(self.themeMenu, name, f)

        self.setAcceptDrops(True)
        self.show()

    def setTheme(self, theme):
        if theme in THEMES:
            for k, v in THEMES[theme].items():
                self.valueCallback(k, v)

            # self.display.updateParams(THEMES[theme])


    def addKeyframe(self):
        self.keyframeEditor.addKeyframe()
        self.toggleKeyframe(show = True)

    # def createScript(self):
    #     fn, ext = QFileDialog.getSaveFileName(self, 'Create MUVI Script File', os.getcwd(), "MUVI script (*.muvi_script)")
    #
    #     # with open(fn, 'wt') as f:
    #     #     pass

    def valueCallback(self, param, value):
        control = self.paramControls.get(param, None)
        if control is not None:
            control.setValue(value)
            control.update()

    def rangeCallback(self, param, minVal, maxVal, delta=None):
        control = self.paramControls.get(param, None)
        if control is not None and hasattr(control, 'setRange'):
            control.setRange(minVal, maxVal, delta)

    def addMenuItem(self, menu, title, func=None, shortcut=None, tooltip=None):
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

    def openFile(self):
        fn, ext = QFileDialog.getOpenFileName(self,
            'Open Volumetric Movie / Mesh Sequence', os.getcwd(),
            "Volumetric Movie (*.vti);; Polygon Mesh (*.ply);; MUVI Script (*.muvi_script)")
        if fn:
            self.openData(fn)

    def openData(self, dat):
        try:
            if isinstance(dat, str) and os.path.splitext(dat)[1] == ".muvi_script":
                self.keyframeEditor.openScript(dat)
                asset = None
            else:
                self.display.makeCurrent()
                asset = self.display.view.openData(dat)
                self.display.doneCurrent()
        except Exception as e:
            ec = e.__class__.__name__
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle(str(ec))
            msg.setText(str(ec) + ": " + str(e))
            msg.setDetailedText(traceback.format_exc())
            msg.setStyleSheet("QTextEdit {font-family: Courier; min-width: 600px;}")
            msg.setStandardButtons(QMessageBox.Cancel)
            msg.exec_()
            # raise
        else:
            if asset is not None:
                self.assetList.addItem(AssetItem(asset, self))
            self.update()

    def openAssets(self, assets):
        try:
            self.newIds = self.display.view.openAssets(assets)
        except Exception as e:
            ec = e.__class__.__name__
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle(str(ec))
            msg.setText(str(ec) + ": " + str(e))
            msg.setDetailedText(traceback.format_exc())
            msg.setStyleSheet("QTextEdit {font-family: Courier; min-width: 600px;}")
            msg.setStandardButtons(QMessageBox.Cancel)
            msg.exec_()
        else:
            relabel = {}
            for id, asset in self.newIds.items():
                if isinstance(asset, int):
                    relabel[id] = asset
                else:
                    relabel[id] = asset.id
                    self.assetList.addItem(AssetItem(asset, self))

            self.update()
            return relabel

    def buildParamTab(self, params, prefix="", defaults={}):
        vbox = QVBoxLayout()
        vbox.setSpacing(10)

        self.addParams(params, vbox, prefix=prefix, defaults=defaults)

        vbox.addStretch(1)

        sa = QScrollArea()
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        sa.setContentsMargins(0, 0, 0, 0)
        sa.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        widget.setLayout(vbox)
        widget.setFixedWidth(PARAM_WIDTH - (SCROLL_WIDTH + 5))
        sa.setWidget(widget)

        return sa

    def addParams(self, params, vbox, prefix="", defaults={}):
        paramControls = paramListToVBox(params, vbox, self.display.view, prefix=prefix, defaults=defaults)

        for param, control in paramControls.items():
            if hasattr(control, 'paramChanged'):
                control.paramChanged.connect(self.display.updateParam)

        self.paramControls.update(paramControls)

    def addParamCategory(self, cat, vbox, prefix="", defaults={}):
        self.addParams(PARAM_CATEGORIES[cat], vbox, prefix="", defaults=defaults)

    def matchAspect(self, event=None):
        size = self.display.size()
        w, h = size.width(), size.height()
        we = self.exportWindow.widthControl.value()
        he = self.exportWindow.heightControl.value()
        newWidth = (we * h) // he
        self.display.resize(newWidth, w)
        self.update()

    def selectAssetTab(self, asset):
        if asset is None:
            tab, label = self.nullTab, 'Data'
        elif isinstance(asset, AssetItem):
            tab, label = asset.tab, asset.label
        else:
            raise ValueError('selectAssetTab should receive an int or AssetItem object')

        self.paramTabs.setUpdatesEnabled(False)
        self.paramTabs.removeTab(0)
        self.paramTabs.insertTab(0, tab, label)
        self.paramTabs.setCurrentIndex(0)
        self.paramTabs.setUpdatesEnabled(True)

    def toggleSettings(self):
        if self.splitter.isVisible():
            self.splitter.setVisible(False)
            self.showSettings.setText('Show View Settings')
        else:
            self.splitter.setVisible(True)
            self.showSettings.setText('Hide View Settings')

    def toggleKeyframe(self, event=None, show=None):
        if show is None:
            show = not self.keyframeEditor.isVisible()
        if show:
            self.keyframeEditor.setVisible(True)
            self.showKeyframeEditor.setText('Hide Keyframe List')
        else:
            self.keyframeEditor.setVisible(False)
            self.showKeyframeEditor.setText('Show Keyframe List')


    def toggleExport(self):
        if self.exportWindow.isVisible():
            self.exportWindow.hide()
            self.showExport.setText('Show Export Window')
        else:
            self.exportWindow.show()
            self.showExport.setText('Hide Export Window')

    def getExportSettings(self):
        return {
            "width": self.exportWindow.widthControl.value(),
            "height": self.exportWindow.heightControl.value(),
            "oversample": self.exportWindow.oversampleControl.value(),
            "scale_height": self.exportWindow.scaleControl.value(),
        }

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
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(Qt.CopyAction)
            event.accept()
            l = []
            if len(event.mimeData().urls()):
                self.openData(event.mimeData().urls()[0].toLocalFile())
        else:
            event.ignore()

    def orient_camera(self, axis):
        if axis == 0:
            self.display.view.resetView(direction=(1, 0, 0), up=(0, 1, 0))
        elif axis == 1:
            self.display.view.resetView(direction=(0, 1, 0), up=(0, 0, -1))
        elif axis == 2:
            self.display.view.resetView(direction=(0, 0, 1), up=(0, 1, 0))
        elif axis == 3:
            self.display.view.resetView(direction=(-1, 0, 0), up=(0, 1, 0))
        elif axis == 4:
            self.display.view.resetView(direction=(0, -1, 0), up=(0, 0, 1))
        else:
            self.display.view.resetView(direction=(0, 0, -1), up=(0, 1, 0))

        self.display.update()

    def allParams(self):
        d = self.display.view.allParams()
        return d

def view_volume(vol=None, args=None, window_name=None):
    global PARAM_WIDTH, SCROLL_WIDTH, UI_EXTRA_SCALING, CLIPBOARD, app

    if window_name is None:
        window_name = APP_NAME

    if args is None:
        args = sys.argv

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(args)

    if LOGICAL_DPI_BASELINE is not None:
        # This is used to fix fractional scaling in Windows, which does
        #   not show up as a devicePixelRatio!
        UI_EXTRA_SCALING = QWidget().logicalDpiX() / LOGICAL_DPI_BASELINE
        PARAM_WIDTH = int(PARAM_WIDTH * UI_EXTRA_SCALING)
        SCROLL_WIDTH = int(SCROLL_WIDTH * UI_EXTRA_SCALING)

    app.setStyle('Fusion')
    app.setPalette(generateDarkPalette())
    app.setStyleSheet(f'''
        QWidget {{
            font-size: {int(12 * UI_EXTRA_SCALING)}px;
        }}

        QLabel, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox {{
            padding: 0px;
            margin: 0px;
        }}
        QGroupBox {{
            padding: 0px;
            padding-top: 20px;
            margin: 0px;
        }}
        QScrollBar:vertical {{
            width: {SCROLL_WIDTH}px;
        }}
        #Border {{
            border: 1px solid #808080;
            border-radius: 4px;
            margin-top: 0px;
            margin-bottom: 5px;
        }}
        QTabBar::tab:selected {{
            color: palette(Text);
        }}
        QTabBar::tab:!selected {{
            color: #A0A0A0;
        }}
    ''')

    app.setApplicationDisplayName(window_name)
    window = VolumetricViewer(clipboard=app.clipboard(), window_name=window_name)
    if vol is not None:
        if isinstance(vol, (list, tuple)):
            for item in vol:
                window.openData(item)
        else:
            window.openData(vol)
    app.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(ICON_DIR, 'muvi_logo.png'))))
    return(app.exec())


def qt_viewer(args=None, window_name=None):
    if args is None:
        args = sys.argv

    if len(args) > 1:
        vol = args[1:]
    else:
        vol = None

    view_volume(vol, args, window_name)


if __name__ == '__main__':
    qt_viewer()
