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
    QFileDialog, QGridLayout, QPushButton, QStyle, QOpenGLWidget, \
    QListWidget, QSplitter, QListWidgetItem, QMenu
from .qtview_widgets import paramListToVBox, controlFromParam, ListControl
from .view import View
import numpy as np
import time
import traceback
Qt = QtCore.Qt
import glob
from PIL import Image

from .params import PARAM_CATEGORIES, PARAMS
# from .. import open_3D_movie, VolumetricMovie

ORG_NAME = "MUVI Lab"
APP_NAME = "MUVI Volumetric Movie Viewer"
ICON_DIR = os.path.split(__file__)[0]


if sys.platform == 'win32':
    # On Windows, it appears to need a bit more width to display text
    PARAM_WIDTH = 250
    SCROLL_WIDTH = 15
elif sys.platform == 'darwin':
    # Good default for OS X
    PARAM_WIDTH = 250
    SCROLL_WIDTH = 15

    # Python 3: pip3 install pyobjc-framework-Cocoa
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        if bundle:
            app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if app_info:
                app_info['CFBundleName'] = APP_NAME
                app_info['NSRequiresAquaSystemAppearance'] = 'No'
                # print(app_info)
    except ImportError:
        raise ImportError('pyobjc-framework-Cocoa not installed (OS X only) -- run "pip3 install pyobjc-framework-Cocoa" or "conda install pyobjc-framework-Cocoa" first')
else:
    # Fallback -- if anyone ever uses this on Linux let me know what looks good!
    PARAM_WIDTH = 250
    SCROLL_WIDTH = 15


class ExportWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent, flags=Qt.Window)
        self.setWindowTitle("Image Export")
        self.parent = parent

        self.vbox = QVBoxLayout()
        self.setLayout(self.vbox)

        self.image = QLabel()
        self.vbox.addWidget(self.image)

        self.settings = QGridLayout()
        self.vbox.addLayout(self.settings)

        self.exportButton = QPushButton("Export Current Frame")
        self.exportButton.clicked.connect(self.saveFrame)
        self.settings.addWidget(self.exportButton, 2, 2, 1, 2)

        self.previewButton = QPushButton("Preview Current Frame")
        self.previewButton.clicked.connect(self.previewFrame)
        self.settings.addWidget(self.previewButton, 2, 4, 1, 2)

        halign = QHBoxLayout()
        self.folderLabel = QLabel(os.getcwd())
        self.folderButton = QPushButton()
        self.folderButton.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        self.folderButton.clicked.connect(self.selectExportFolder)
        halign.addWidget(self.folderButton, 0)
        self.settings.addLayout(halign, 3, 1, 1, 6)
        halign.addWidget(self.folderLabel, 1)

        self.fileLabel = QLabel("")
        self.fileLabel.setFixedWidth(512)
        self.settings.addWidget(self.fileLabel, 4, 1, 1, 6)

        self.ss_sizes = [512, 640, 720, 768, 1024, 1080, 1280, 1920,
            2048, 2160, 3072, 3240, 3840, 4096, 4320, 5760, 6144, 7680, 8192]
        self.widthControl = ListControl('Width:', 1920, self.ss_sizes, param='os_width')
        self.heightControl = ListControl('Height:', 1080, self.ss_sizes, param='os_height')

        def print_param(p, v):
            print(f'{p}: {v}')
        self.widthControl.paramChanged.connect(self.updateBuffer)
        self.heightControl.paramChanged.connect(self.updateBuffer)

        self.scaleControl = ListControl('Scale Height:', 1080, self.ss_sizes, param='scaling_height',
            tooltip='Effective screen height to use for axis scaling.  Used to prevent super thin lines and tiny text for high resolutions!')
        self.scaleHeight = self.scaleControl.value()
        self.scaleControl.paramChanged.connect(self.updatescaleHeight)

        self.settings.setColumnStretch(0, 1)
        self.settings.setColumnStretch(7, 1)
        self.settings.addWidget(self.widthControl, 0, 1, 1, 2)
        self.settings.addWidget(self.heightControl, 0, 3, 1, 2)
        self.settings.addWidget(self.scaleControl, 0, 5, 1, 2)


        for i, (label, w, h) in enumerate([
                    ('720p', 1280, 720),
                    ('1080p', 1920, 1080),
                    ('1440p', 2560, 1440),
                    ('2160p (4K)', 3840, 2160),
                    ('3240p (6K)', 5760, 3240),
                    ('4320p (8K)', 7680, 4320,)
                ]):
            button = QPushButton(label)

            def cr(state, w=w, h=h):
                self.widthControl.setValue(w)
                self.heightControl.setValue(h)

            button.clicked.connect(cr)
            self.settings.addWidget(button, 1, i+1)

    def updatescaleHeight(self, key, val):
        self.scaleHeight = val

    def updateBuffer(self, key=None, val=None):
        width, height = self.widthControl.value(), self.heightControl.value()

        self.parent.display.makeCurrent()
        if not hasattr(self, 'bufferId'):
            self.bufferId = self.parent.display.view.addBuffer(width, height)
        else:
            self.parent.display.view.resizeBuffer(self.bufferId, width, height)
        self.parent.display.doneCurrent()

    def closeEvent(self, e):
        self.parent.toggleExport()

    def selectExportFolder(self):
        self.folderLabel.setText(QFileDialog.getExistingDirectory(
            self, "Select Export Folder", self.folderLabel.text()))

    def renderImage(self):
        if not hasattr(self, 'bufferId'):
            self.updateBuffer()

        return self.parent.display.offscreenRender(self.bufferId, scaleHeight=self.scaleHeight)

    def saveFrame(self, event=None):
        dir = self.folderLabel.text()
        fns = glob.glob(os.path.join(dir, 'muvi_screenshot_*.png'))
        for i in range(10**4):
            fn = os.path.join(dir, 'muvi_screenshot_%08d.png' % i)
            if fn not in fns:
                break

        img = self.renderImage()
        Image.fromarray(img[::-1]).save(os.path.join(dir, fn))

        self.updatePreview(img)
        self.fileLabel.setText(f'Saved to: {os.path.split(fn)[1]}')

    def previewFrame(self, event=None):
        self.updatePreview(self.renderImage())

    def updatePreview(self, img):
        img = QtGui.QImage(np.require(img[::-1, :, :3], np.uint8, 'C'),
            img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
        self.image.setPixmap(QtGui.QPixmap(img).scaledToWidth(1024))




class GLWidget(QOpenGLWidget):
    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent):
        self.parent = parent

        fmt = QtGui.QSurfaceFormat()
        fmt.setProfile(QtGui.QSurfaceFormat.CoreProfile)
        fmt.setSwapInterval(1)
        fmt.setVersion(3, 3)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.dpr = 1
        self.view = View(getattr(parent, "valueCallback", None),
                         getattr(parent, "rangeCallback", None))
        self.view['fontSize'] = self.font().pointSize() / 72 * self.physicalDpiX() * 1.2

        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update)

        self._isPlaying = False
        self.hasUpdate = True

    def updateParam(self, k, v):
        self.view[k] = v
        self.hasUpdate = True
        self.update()

    def updateParams(self, params):
        self.view.update(params)
        self.hasUpdate = True
        self.update()

    def paintGL(self):
        try:
            if self._isPlaying and self.view.frameRange is not None:
                t = time.time()
                fps = self.view['framerate']
                advance = int((t - self._tStart) * fps)
                if advance:
                    frame = (self.view['frame'] + advance) % self.view.frameRange[1]
                    self.view['frame'] = frame
                    self._tStart += advance / fps
                    self.frameChanged.emit(frame)

            if self.hasUpdate:
                self.view.draw(0, True)

        except:
            traceback.print_exc()
            self.parent.close()

    def offscreenRender(self, bufferId, scaleHeight=None):
        self.makeCurrent()
        self.view.draw(bufferId, scaleHeight=scaleHeight)
        img = np.array(self.view.buffers[bufferId].texture)
        self.doneCurrent()
        return img

    def resizeGL(self, width, height):
        self.width, self.height = width * self.dpr, height * self.dpr
        self.view.resizeBuffer(0, self.width, self.height)
        self.hasUpdate = True
        self.update()

    def initializeGL(self):
        # print(f'GL Version: {GL.glGetString(GL.GL_VERSION)}')
        # print(f'GLSL Version: {GL.glGetString(GL.GL_SHADING_LANGUAGE_VERSION)}')
        self.dpr = self.devicePixelRatio()
        self.view.setup(self.width() * self.dpr, self.height() * self.dpr)
        self.view['display_scaling'] = self.dpr

    def close(self):
        self.view.cleanup()

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.update()

    def mouseReleaseEvent(self, event):
        self.lastPos = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        dx = x - self.lastPos.x()
        dy = y - self.lastPos.y()
        self.lastPos = event.pos()
        buttonsPressed = int(event.buttons())

        if dx or dy:
            self.view.mouseMove(x*self.dpr, y*self.dpr, dx*self.dpr, dy*self.dpr, buttonsPressed)
            self.hasUpdate = True
            self.update()

    def wheelEvent(self, event):
        self.view.zoomCamera(1.25**(event.angleDelta().y()/120))
        self.hasUpdate = True
        self.update()

    def setPlaying(self, isPlaying):
        if isPlaying:
            self.play()
        else:
            self.pause()

    def play(self):
        if self.view.frameRange is not None:
            self._isPlaying = True
            self._tStart = time.time()
            self.timer.start()

    def pause(self):
        self._isPlaying = False
        self.timer.stop()

    def resetView(self):
        self.view.resetView()
        self.update()


class AssetItem(QListWidgetItem):
    def __init__(self, asset, mainWindow, parent=None):
        super().__init__(asset.label, parent)
        self.id = asset.id
        self.isVolume = asset.isVolume
        self.setFlags(self.flags() | Qt.ItemIsUserCheckable ) #
        self.prefix = f'#{self.id}_'
        self.setCheckState(Qt.Checked if asset.visible else Qt.Unchecked)
        self.setToolTip('\n'.join(asset.info))
        self.asset = asset

        self.tab = mainWindow.buildParamTab(asset.paramList(), prefix=self.prefix)
        self.label = asset.shader.capitalize()


class AssetList(QListWidget):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.mainWindow = mainWindow

        self.currentRowChanged.connect(self.selectAssetTab)
        self.itemChanged.connect(self.assetChanged)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.contextMenu)

        self.openAction = QAction('Open File')
        self.openAction.triggered.connect(self.mainWindow.openFile)
        self.removeAction = QAction('Remove Item')
        self.limitsAction = QAction('Set Display Limits to Object Extent')

        self.assets = {}

    def selectAssetTab(self, row):
        self.mainWindow.selectAssetTab(self.item(row))

    def contextMenu(self, point):
        item = self.itemAt(point)

        menu = QMenu()
        menu.addAction(self.openAction)

        if item is not None:
            menu.addAction(self.removeAction)
            menu.addAction(self.limitsAction)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == self.removeAction:
            self.removeAsset(item)
        elif action == self.limitsAction:
            item.asset.X0
            item.asset.X1
            self.mainWindow.display.updateParams({
                'disp_X0': item.asset.X0,
                'disp_X1': item.asset.X1,
            })

    def assetChanged(self, asset):
        checked = asset.checkState() == Qt.Checked

        # Check if it's fully set up first...
        if not hasattr(asset, 'prefix'):
            return

        params = {asset.prefix + "visible": checked}

        # If we are activating a volume, deactivate all others!
        if asset.isVolume and checked:
            self.blockSignals(True)

            # Turn off all other volumes, but block signals to prevent this
            #   from getting called again
            for asset2 in self.assets.values():
                if asset.id == asset2.id or (not asset2.isVolume):
                    continue

                asset2.setCheckState(Qt.Unchecked)
                params[asset2.prefix + "visible"] = False

            self.blockSignals(False)
        self.mainWindow.display.updateParams(params)

    def removeAsset(self, item):
        id = item.id
        self.mainWindow.display.view.removeAsset(id)
        self.takeItem(self.indexFromItem(item).row())
        del self.assets[id]
        self.mainWindow.display.update()

    def addItem(self, assetItem):
        self.assets[assetItem.id] = assetItem
        super().addItem(assetItem)
        self.setCurrentRow(self.indexFromItem(assetItem).row())
        assetItem.setCheckState(Qt.Checked)


class VolumetricViewer(QMainWindow):
    def __init__(self, parent=None, window_name=None):
        super().__init__(parent)

        self.setWindowTitle(window_name)

        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(0, 0, 0, 0)

        self.hbox = QHBoxLayout()
        self.display = GLWidget(parent=self)
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

        self.viewMenu = menu.addMenu("View")

        self.showSettings = self.addMenuItem(self.viewMenu,
            'Hide View Settings', self.toggleSettings, 'Ctrl+/',
            'Show or hide settings option on right side of main window')

        self.save_image = self.addMenuItem(self.viewMenu,
            'Save Screenshot', self.exportWindow.saveFrame, 's',
            'Save a screenshot with the current export settings (use export window to control resolution).')

        self.showExport = self.addMenuItem(self.viewMenu,
            'Show Export Window', self.toggleExport, 'Ctrl+E',
            'Show or hide the export window, used to take screenshots or make movies')

        for i in range(3):
            axis = chr(ord('X') + i)

            def f(event, a=i):
                self.orient_camera(a)
            self.addMenuItem(self.viewMenu,
                f'Look down {axis}-axis', f, axis.lower())

            def f2(event, a=i):
                self.orient_camera(a+3)
            self.addMenuItem(self.viewMenu,
                f'Look down -{axis}-axis', f2, 'Shift+'+axis.lower())

        self.setAcceptDrops(True)
        self.show()

    def valueCallback(self, param, value):
        control = self.paramControls.get(param, None)
        if control is not None:
            control.setValue(value)

    def rangeCallback(self, param, minVal, maxVal):
        control = self.paramControls.get(param, None)
        if control is not None and hasattr(control, 'setRange'):
            control.setRange(minVal, maxVal)

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
        fn, ext = QFileDialog.getOpenFileName(self, 'Open Volumetric Movie / Mesh Sequence', os.getcwd(), "Volumetric Movie (*.vti);; Polygon Mesh (*.ply)")
        if fn:
            self.openData(fn)

    def openData(self, dat):
        try:
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
            self.assetList.addItem(AssetItem(asset, self))
            self.update()

    def buildParamTab(self, params, prefix=""):
        vbox = QVBoxLayout()
        vbox.setSpacing(10)
        paramControls = paramListToVBox(params, vbox, self.display.view, prefix=prefix)

        vbox.addStretch(1)

        sa = QScrollArea()
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        sa.setContentsMargins(0, 0, 0, 0)
        sa.setFrameShape(QFrame.NoFrame)

        widget = QWidget()
        widget.setLayout(vbox)
        widget.setFixedWidth(PARAM_WIDTH - (SCROLL_WIDTH + 5))
        sa.setWidget(widget)

        for param, control in paramControls.items():
            if hasattr(control, 'paramChanged'):
                control.paramChanged.connect(self.display.updateParam)
        # print(list(paramControls.keys()))

        self.paramControls.update(paramControls)

        return sa

    def addParamCategory(self, cat, vbox):
        paramControls = paramListToVBox(PARAM_CATEGORIES[cat], vbox,
            self.display.view)

        for param, control in paramControls.items():
            if hasattr(control, 'paramChanged'):
                control.paramChanged.connect(self.display.updateParam)

        self.paramControls.update(paramControls)

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

    def toggleExport(self):
        if self.exportWindow.isVisible():
            self.exportWindow.hide()
            self.showExport.setText('Show Export Window')
        else:
            self.exportWindow.show()
            self.showExport.setText('Hide Export Window')

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
        for k, v in d.items():
            print(f'{k:>20s}: {v}')

        print(f'\nTotal bytes: {sum(sys.getsizeof(v) for v in d.values()) + sys.getsizeof(d)}')




def generateDarkPalette():
    c = QtGui.QColor

    palette = QtGui.QPalette()
    palette.setColor(palette.Window,            c( 53,  53,  53))
    palette.setColor(palette.WindowText,        c(255, 255, 255))
    palette.setColor(palette.Text,              c(255, 255, 255))
    palette.setColor(palette.HighlightedText,   c(255, 255, 255))
    palette.setColor(palette.ButtonText,        c(255, 255, 255))
    palette.setColor(palette.ToolTipBase,       c(  0,   0,   0))
    palette.setColor(palette.ToolTipText,       c(255, 255, 255))
    palette.setColor(palette.Light,             c(100, 100, 100))
    palette.setColor(palette.Button,            c(70,   70,  70))
    palette.setColor(palette.Dark,              c( 20, 20,  20))
    palette.setColor(palette.Shadow,            c( 0,  0,  0))
    palette.setColor(palette.Base,              c( 30,  30,  30))
    palette.setColor(palette.AlternateBase,     c( 66,  66,  66))
    palette.setColor(palette.Highlight,         c( 42, 130, 218))
    palette.setColor(palette.BrightText,        c(255,   0,   0))
    palette.setColor(palette.Link,              c( 42, 130, 218))
    palette.setColor(palette.Highlight,         c( 42, 130, 218))
    palette.setColor(palette.Disabled, palette.WindowText,      c(127, 127, 127))
    palette.setColor(palette.Disabled, palette.Text,            c(127, 127, 127))
    palette.setColor(palette.Disabled, palette.ButtonText,      c(127, 127, 127))
    palette.setColor(palette.Disabled, palette.HighlightedText, c(127, 127, 127))

    return palette


def view_volume(vol=None, args=None, window_name=None):
    if window_name is None:
        window_name = APP_NAME

    if args is None:
        args = sys.argv

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)

    app = QApplication(args)

    # start = time.time()
    # QtGui.QFontDatabase.addApplicationFont(os.path.join(os.path.split(__file__)[0], "fonts/Inter-Regular.ttf"))
    # print(time.time() - start)

    app.setStyle('Fusion')
    app.setPalette(generateDarkPalette())
    app.setStyleSheet(f'''
        QWidget {{
            font-size: 12px;
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
    window = VolumetricViewer(window_name=window_name)
    if vol is not None:
        window.openData(vol)
    app.setWindowIcon(QtGui.QIcon(QtGui.QPixmap(os.path.join(ICON_DIR, 'muvi_logo.png'))))
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
