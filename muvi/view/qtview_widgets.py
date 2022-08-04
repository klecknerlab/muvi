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
This module contains the Qt widgets used to create a 3D viewer, including
controls for changing parameters.  The actual viewer is in the qtview module.

Note that this module (and only this module) uses CamelCase naming for
variables, methods, etc., to conform with Qt standards.  The rest of the muvi
library uses underscore style naming to conform with Python standards.
'''

# import sys
from PyQt5 import QtCore, QtGui
Qt = QtCore.Qt
from PyQt5.QtWidgets import QWidget, QSpinBox, QVBoxLayout, QHBoxLayout, \
    QDoubleSpinBox, QSlider, QLabel, QTabWidget, QGroupBox, QCheckBox, QFrame, \
    QComboBox, QColorDialog, QPushButton, QLineEdit, QOpenGLWidget, QListWidget, \
    QListWidgetItem, QAction, QMenu
from PyQt5.QtGui import QColor, QSurfaceFormat
# from OpenGL import GL
# from .ooopengl import GL_SRGB8_ALPHA8
# from  import view
import math
import traceback
import time
import glob
import os
import numpy as np
from .view import View

GAMMA_CORRECT = 2.2

class UnwheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()

class UnwheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class UnwheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()

class ParamControl(QWidget):
    paramChanged = QtCore.pyqtSignal(str, object)

    def __init__(self, title="Value:", param=None, tooltip=None, w=None,
        slider=False, sideSlider=False):
        super().__init__()
        self.title = QLabel(title)
        self.param = param
        if tooltip is not None:
            self.setToolTip(tooltip)

        self.hbox = QHBoxLayout()
        self.hbox.setContentsMargins(0, 0, 0, 0)
        self.hbox.addWidget(self.title)
        if isinstance(w, (list, tuple)):
            for ww in w:
                self.hbox.addWidget(ww)
        else:
            self.hbox.addWidget(w)

        if slider:
            self.slider = UnwheelSlider(QtCore.Qt.Horizontal)
            self.slider.setTickPosition(QSlider.TicksAbove | QSlider.TicksBelow)

            if sideSlider:
                self.hbox.addWidget(self.slider, 1)
                self.setLayout(self.hbox)
            else:
                self.vbox = QVBoxLayout()
                self.vbox.setContentsMargins(0, 0, 0, 0)
                self.vbox.addLayout(self.hbox)
                self.vbox.addWidget(self.slider)
                self.setLayout(self.vbox)
        else:
            self.setLayout(self.hbox)

        self.silent = False

    def _paramChanged(self, value):
        if not self.silent:
            self.paramChanged.emit(self.param, value)


class PlaybackControl(ParamControl):
    playingChanged = QtCore.pyqtSignal(bool)

    def __init__(self, title="Value:", default=50, minVal=0, maxVal=100,
            step=1, tooltip=None, param=None):
        self.spinBox = QSpinBox()
        self.frameCount = QLabel()
        self.playButton = QPushButton('Play')
        self.isPlaying = False
        self.playButton.clicked.connect(self.togglePlay)

        super().__init__(title, param, tooltip, w=(self.spinBox, self.frameCount),
            slider=True, sideSlider=True)

        self.hbox.insertWidget(0, self.playButton, 0)
        self.hbox.setContentsMargins(10, 10, 10, 10)

        self.setRange(minVal, maxVal)
        self.setValue(default)

        if step is None:
            step = max(1, maxVal - minVal // 10)
        self.setSingleStep(step)

        self.slider.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.slider.setValue)

        self.spinBox.valueChanged.connect(self._paramChanged)

    def togglePlay(self):
        self.isPlaying = not self.isPlaying

        if self.isPlaying:
            self.playButton.setText('Pause')
        else:
            self.playButton.setText('Play')

        # print(self.isPlaying)
        self.playingChanged.emit(self.isPlaying)

    def setRange(self, minVal, maxVal):
        self.spinBox.setRange(minVal, maxVal)
        self.slider.setRange(minVal, maxVal)
        self.frameCount.setText(f'/ {maxVal}')

    def setValue(self, value):
        self.spinBox.setValue(value)
        self.slider.setValue(value)

    def setSilent(self, value):
        silent = self.silent
        self.silent = True
        self.setValue(value)
        self.silent = silent

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)
        self.slider.setSingleStep(step)
        self.slider.setTickInterval(10)


class BoolControl(ParamControl):
    def __init__(self, title="Value:", default=True, tooltip=None, param=None):
        self.checkBox = QCheckBox()
        super().__init__(title, param, tooltip, self.checkBox)
        self.checkBox.setChecked(default)
        self.checkBox.stateChanged.connect(lambda s: self._paramChanged(bool(s)))

    def setValue(self, value):
        self.checkBox.setChecked(value)


class IntControl(ParamControl):
    def __init__(self, title="Value:", default=50, minVal=0, maxVal=100,
            step=None, tooltip=None, param=None):
        self.spinBox = UnwheelSpinBox()

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        self.setRange(minVal, maxVal)
        self.setValue(default)

        if step is None:
            step = max(1, maxVal - minVal // 10)
        self.setSingleStep(step)

        self.slider.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.slider.setValue)

        self.spinBox.valueChanged.connect(self._paramChanged)

        self.value = self.spinBox.value

    def setRange(self, minVal, maxVal):
        self.spinBox.setRange(minVal, maxVal)
        self.slider.setRange(minVal, maxVal)

    def setValue(self, value, silent=False):
        self.spinBox.setValue(value)
        self.slider.setValue(value)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)
        self.slider.setSingleStep(step)
        self.slider.setTickInterval(step)


class LinearControl(ParamControl):
    def __init__(self, title="Value:", default=50, minVal=0, maxVal=100,
            step=None, decimals=None, tooltip=None, param=None, subdiv=5):

        self.spinBox = UnwheelDoubleSpinBox()

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        self.subdiv = subdiv
        self.step = step

        self.decimals = decimals
        self.setRange(minVal, maxVal)
        self.setValue(default)

        self.slider.valueChanged.connect(self.sliderChanged)
        self.spinBox.valueChanged.connect(self.spinChanged)

        self.spinBox.valueChanged.connect(self._paramChanged)

        self.value = self.spinBox.value

    def setRange(self, minVal, maxVal):
        step = (maxVal - minVal) / 10 if self.step is None else self.step
        self.currentStep = step

        self.steps = math.ceil((maxVal-minVal) / step)
        self.sliderSteps = self.steps * self.subdiv
        self.minVal = minVal
        self.maxVal = maxVal
        self.ratio = self.sliderSteps / (self.maxVal - self.minVal)
        self.spinBox.setRange(minVal, maxVal)
        self.slider.setRange(0, self.sliderSteps)
        self.spinBox.setSingleStep(step)
        self.slider.setSingleStep(self.subdiv)
        self.slider.setTickInterval(self.subdiv * 2 if self.sliderSteps//self.subdiv > 10 else self.subdiv)

        if self.decimals is None:
            self.spinBox.setDecimals(math.ceil(1.5-math.log10(self.currentStep)))
        else:
            self.spinBox.setDecimals(self.decimals)

    def setValue(self, value, silent=False):
        self.spinBox.setValue(value)
        self.slider.setValue(int((value - self.minVal) * self.ratio + 0.5))

    def setSilent(self, value):
        silent = self.silent
        self.silent = True
        self.setValue(value)
        self.silent = silent

    def spinChanged(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int((value - self.minVal) * self.ratio + 0.5))
        self.slider.blockSignals(False)

    def sliderChanged(self, value):
        self.spinBox.setValue(value / self.ratio + self.minVal)


class LogSpinBox(UnwheelDoubleSpinBox):
    def stepBy(self, step):
        sm = self.singleStep()
        i = math.log(self.value(), sm) + step
        self.setValue(sm**round(i))


class LogControl(ParamControl):
    def __init__(self, title="Value:", default=50, minVal=0, maxVal=100,
            step=None, decimals=None, tooltip=None, param=None, subdiv=5):
        self.spinBox = LogSpinBox()

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        if step is None:
            step = (maxVal - minVal) / 10

        if decimals is None:
            decimals = math.ceil(0.5-math.log10(minVal))

        self.spinBox.setDecimals(decimals)

        self.subdiv = subdiv
        self.step = step

        self.setRange(minVal, maxVal)

        # self.setRange(minVal, maxVal, math.ceil((math.log10(maxVal) - math.log10(minVal)) / math.log10(step) - 1E-6) * subdiv)
        # self.setSingleStep(step)
        self.setValue(default)

        self.slider.valueChanged.connect(self.sliderChanged)
        self.spinBox.valueChanged.connect(self.spinChanged)

        self.spinBox.valueChanged.connect(self._paramChanged)

    def setRange(self, minVal, maxVal):
        self.logMin = math.log10(minVal)
        self.logMax = math.log10(maxVal)
        self.logStep = math.log10(self.step)
        self.steps = math.ceil((self.logMax-self.logMin) / self.logStep)
        self.sliderSteps = self.steps * self.subdiv
        self.ratio = self.sliderSteps / (self.logMax - self.logMin)
        self.spinBox.setRange(minVal, maxVal)
        self.slider.setRange(0, self.sliderSteps)
        self.spinBox.setSingleStep(self.step)
        self.slider.setSingleStep(self.subdiv)
        self.slider.setTickInterval(self.subdiv * 2 if self.sliderSteps//self.subdiv > 10 else self.subdiv)

    def setValue(self, value, silent=False):
        self.spinBox.setValue(value)
        self.slider.setValue(int((math.log10(value) - self.logMin) * self.ratio + 0.5))

    def spinChanged(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int((math.log10(value) - self.logMin) * self.ratio + 0.5))
        self.slider.blockSignals(False)

    def sliderChanged(self, value):
        self.spinBox.setValue(10**(value / self.ratio + self.logMin))


class OptionsControl(ParamControl):
    def __init__(self, title="Value:", default=None, options={}, param=None,
        tooltip=None):
        self.comboBox = QComboBox()

        super().__init__(title, param, tooltip, w=self.comboBox)

        self.setOptions(options, default)

    def setOptions(self, options, default=None):
        silent = self.silent
        self.silent = True
        if isinstance(options, dict):
            self.option_values = tuple(options.keys())
            self.option_names = tuple(options.values())
        else:
            self.option_values = tuple(options)
            self.option_names = option_values

        self.comboBox.clear()

        for name in self.option_names:
            self.comboBox.addItem(name)

        if default is not None:
            self.setValue(default)

        self.silent = silent

        self.comboBox.currentIndexChanged.connect(lambda i: self._paramChanged(self.option_values[i]))

    def setValue(self, value):
        if isinstance(value, int):
            self.comboBox.setCurrentIndex(value)
        else:
            self.comboBox.setCurrentIndex(self.option_values.index(value))

    def value(self):
        return self.option_values[self.comboBox.currentIndex()]

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


class ColorSelector(QPushButton):
    valueChanged = QtCore.pyqtSignal(QColor)

    def __init__(self, default, hasAlpha=False, parent=None):
        self.hasAlpha = bool(hasAlpha)
        super().__init__(parent=parent)
        self.setObjectName('colorSelector')
        self.setValue(default)

    def setValue(self, color):
        self.value = color
        self.setStyleSheet(f"QWidget {{ background-color: {self.value.name()} }}")
        self.valueChanged.emit(self.value)

    def mousePressEvent(self, event):
        if self.hasAlpha:
            self.setValue(QColorDialog.getColor(self.value, options=QColorDialog.ShowAlphaChannel))
        else:
            self.setValue(QColorDialog.getColor(self.value))


class ColorControl(ParamControl):
    def __init__(self, title="Color:", default=None, param=None, tooltip=None):
        self.hasAlpha = len(default) == 4
        self.colorSelector = ColorSelector(toQColor(default), self.hasAlpha)

        super().__init__(title, param, tooltip, w=self.colorSelector)

        self.colorSelector.valueChanged.connect(lambda c: self._paramChanged(fromQColor(c, self.hasAlpha)))

    def setValue(self, value):
        self.colorSelector.setValue(toQColor(value))


class ListSpinBox(UnwheelSpinBox):
    def __init__(self, default, values, minVal=None, maxVal=None, parent=None):
        super().__init__(parent)
        self.values = tuple(sorted(values))

        if minVal is None:
            minVal = self.values[0]
        if maxVal is None:
            maxVal = self.values[-1]

        self.setRange(minVal, maxVal)
        self.setValue(default)

    def getIndex(self, value):
        for i in reversed(range(len(self.values))):
            if value >= self.values[i]:
                return i

        return 0

    def stepBy(self, step):
        val = self.value()
        i = self.getIndex(val)

        if step < 0 and val > self.values[i]:
            i += 1

        i = max(min(i + step, len(self.values) - 1), 0)
        self.setValue(self.values[i])


class ListControl(ParamControl):
    def __init__(self, title="Value:", default=None, values=[0, 1], tooltip=None,
            minVal=None, maxVal=None, param=None):

        if default is None:
            default = values[0]

        self.spinBox = ListSpinBox(default, values, minVal, maxVal)

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        self.slider.setRange(0, len(self.spinBox.values) - 1)

        self.spinChanged(default)

        self.slider.valueChanged.connect(self.sliderChanged)
        self.spinBox.valueChanged.connect(self.spinChanged)
        self.spinBox.valueChanged.connect(self._paramChanged)
        self.value = self.spinBox.value

    def setValue(self, value):
        self.spinBox.setValue(value)
        # self.slider.blockSignals(True)
        # self.slider.setValue(self.spinBox.getIndex(self.spinBox.value()))
        # self.slider.blockSignals(False)

    def spinChanged(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(self.spinBox.getIndex(self.spinBox.value()))
        self.slider.blockSignals(False)

    def sliderChanged(self, value):
        self.spinBox.setValue(self.spinBox.values[value])


class VectorControl(QWidget):
    paramChanged = QtCore.pyqtSignal(str, object)

    def __init__(self, title="Vector:", default=np.zeros(3), minVal=np.zeros(3),
            maxVal=np.ones(3), step=None, param=None, tooltip=None, subdiv=5,
            labels=['X:', 'Y:', 'Z:', 'W:']):
        super().__init__()


        self.vbox = QVBoxLayout()

        self.frame = QFrame()
        self.frame.setObjectName('Border')

        self.vbox.addWidget(QLabel(title))
        self.vbox.addWidget(self.frame)
        self.vbox.setContentsMargins(0, 0, 0, 0)


        self.step = step
        self.sub_vbox = QVBoxLayout()
        self.setLayout(self.vbox)
        self.param = param


        self.frame.setLayout(self.sub_vbox)

        self.controls = []
        self.value = default.copy()

        for i in range(len(default)):
            control = LinearControl(labels[i], default[i], minVal[i], maxVal[i],
                self.step, param=str(i), subdiv=subdiv, tooltip=tooltip)
            self.sub_vbox.addWidget(control)
            control.paramChanged.connect(self._paramChanged)
            self.controls.append(control)

        self.silent = False

    def setValue(self, value):
        self.value = value.copy()
        for control, val in zip(self.controls, self.value):
            control.blockSignals(True)
            control.setValue(val)
            control.blockSignals(False)

    def setSilent(self, value):
        silent = self.silent
        self.silent = True
        self.setValue(value)
        self.silent = silent

    def setRange(self, minVal, maxVal):
        for control, minv, maxv in zip(self.controls, minVal, maxVal):
            control.setRange(minv, maxv)

    def _paramChanged(self, p, v):
        self.value[int(p)] = v
        self.paramChanged.emit(self.param, self.value)



# class QHLine(QFrame):
#     def __init__(self):
#         super(QHLine, self).__init__()
#         self.setFrameShape(QFrame.HLine)
#         # self.setFrameShadow(QFrame.Sunken)
#         # self.setLineWidth(3)


PARAM_FIELDS = {
    'default':'default',
    'tooltip':'tooltip',
    'min':'minVal',
    'max':'maxVal',
    'step':'step',
    'logstep':'step',
    'options':'options',
}

def controlFromParam(param, view=None, prefix="", defaults={}):
    if hasattr(param, 'action'):
        button = QPushButton(param.display_name)
        func = getattr(view, param.action, None)
        if func is not None:
            button.clicked.connect(lambda: func(*param.args, **param.kw))
        return button

    full_name = prefix + param.name

    kwargs = dict(
        title = param.display_name + ":",
        param = full_name
    )

    for p, k in PARAM_FIELDS.items():
        v = getattr(param, p, None)
        if v is not None:
            kwargs[k] = v

    if full_name in defaults:
        kwargs['default'] = defaults[full_name]
    elif param in defaults:
        kwargs['default'] = defaults[param]

    if param.type == bool:
        return BoolControl(**kwargs)
    elif param.type == int:
        return IntControl(**kwargs)
    elif param.type == 'playback':
        return PlaybackControl(**kwargs)
    elif param.type == float:
        if hasattr(param, 'logstep'):
            return LogControl(**kwargs)
        else:
            return LinearControl(**kwargs)
    elif param.type == 'options':
        return OptionsControl(**kwargs)
    elif param.type == 'color':
        return ColorControl(**kwargs)
    elif param.type == np.ndarray:
        return VectorControl(**kwargs)
    else:
        return None

def paramListToVBox(params, vbox, view=None, prefix="", defaults={}):
    param_controls = {}

    sub_vbox = vbox

    for param in params:
        if isinstance(param, dict):
            # We have subcategories -- make tabs!
            tabs = QTabWidget()
            # tabs.setContentsMargins(5, 5, 5, 5)
            tabs.setStyleSheet('QTabWidget {background-color: palette(window);}')
            for cat, param_list in param.items():
                tab_vbox = QVBoxLayout()
                param_controls.update(paramListToVBox(param_list, tab_vbox, prefix=prefix, defaults=defaults))
                widget = QWidget()
                widget.setLayout(tab_vbox)
                tabs.addTab(widget, cat)

            vbox.addWidget(tabs)

        elif isinstance(param, str):
            sub_vbox = QVBoxLayout()
            frame = QFrame()
            vbox.addWidget(QLabel(param))
            frame.setObjectName('Border')
            frame.setLayout(sub_vbox)
            vbox.addWidget(frame)

        else:
            # This is just an item
            control = controlFromParam(param, view, prefix=prefix, defaults=defaults)
            if control is not None:
                sub_vbox.addWidget(control)
                if hasattr(param, 'name'):
                    param_controls[prefix + param.name] = control

    return param_controls


class StaticViewWidget(QOpenGLWidget):
    def __init__(self, parent, uiScale=1.0):
        self.parent = parent
        self.uiScale = uiScale

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
        # self.view['fontSize'] = self.font().pointSize()
        # print(self.font().pointSize())

    def updateParam(self, k, v, callback=False):
        self.view.__setitem__(k, v, callback=callback)
        self.hasUpdate = True
        self.update()

    def updateParams(self, params, callback=False):
        self.view.update(params, callback=callback)
        self.hasUpdate = True
        self.update()

    def paintGL(self):
        if not getattr(self, "_enabled", True):
            return

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
        self.view['display_scaling'] = self.dpr * self.uiScale

    def close(self):
        self.view.cleanup()

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    def resetView(self):
        self.view.resetView()
        self.update()


class ViewWidget(StaticViewWidget):
    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update)

        self._isPlaying = False
        self.hasUpdate = True

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
        self.setFlags(self.flags() | Qt.ItemIsUserCheckable ) #
        self.prefix = f'#{self.id}_'
        self.setCheckState(Qt.Checked if asset.visible else Qt.Unchecked)
        self.setToolTip('\n'.join(asset.get_info()))
        self.asset = asset
        self.shader = asset.shader

        self.tab = mainWindow.buildParamTab(asset.paramList(), prefix=self.prefix, defaults=asset.allParams())
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
            }, callback=True)

    def assetChanged(self, asset):
        checked = asset.checkState() == Qt.Checked

        # Check if it's fully set up first...
        if not hasattr(asset, 'prefix'):
            return

        params = {asset.prefix + "visible": checked}

        # If we are activating a volume, deactivate all others!
        if asset.shader == 'volume' and checked:
            self.blockSignals(True)

            # Turn off all other volumes, but block signals to prevent this
            #   from getting called again
            for asset2 in self.assets.values():
                if asset.id == asset2.id or (asset2.shader != "volume"):
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
