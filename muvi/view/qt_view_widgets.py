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
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QSpinBox, QVBoxLayout, QHBoxLayout, \
    QDoubleSpinBox, QSlider, QLabel, QTabWidget, QGroupBox, QCheckBox, QFrame, \
    QComboBox, QColorDialog, QPushButton, QLineEdit, QOpenGLWidget
from PyQt5.QtGui import QColor, QSurfaceFormat
# from OpenGL import GL
from .ooopengl import GL_SRGB8_ALPHA8
from . import view
import math
import traceback
import time
import glob
import os

GAMMA_CORRECT = 2.2

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
        self.hbox.addWidget(self.title)
        if isinstance(w, (list, tuple)):
            for ww in w:
                self.hbox.addWidget(ww)
        else:
            self.hbox.addWidget(w)

        if slider:
            self.slider = QSlider(QtCore.Qt.Horizontal)
            self.slider.setTickPosition(QSlider.TicksAbove | QSlider.TicksBelow)

            if sideSlider:
                self.hbox.addWidget(self.slider, 1)
                self.setLayout(self.hbox)
            else:
                self.vbox = QVBoxLayout()
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
        self.checkBox.stateChanged.connect(lambda s: self._paramChanged(bool(s)))

    def setValue(self, value):
        self.checkBox.setChecked(value)


class IntControl(ParamControl):
    def __init__(self, title="Value:", default=50, minVal=0, maxVal=100,
            step=None, tooltip=None, param=None):
        self.spinBox = QSpinBox()

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        self.setRange(minVal, maxVal)
        self.setValue(default)

        if step is None:
            step = max(1, maxVal - minVal // 10)
        self.setSingleStep(step)

        self.slider.valueChanged.connect(self.spinBox.setValue)
        self.spinBox.valueChanged.connect(self.slider.setValue)

        self.spinBox.valueChanged.connect(self._paramChanged)

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

        self.spinBox = QDoubleSpinBox()

        super().__init__(title, param, tooltip, w=self.spinBox, slider=True)

        if step is None:
            step = (maxVal - minVal) / 10

        if decimals is None:
            decimals = math.ceil(1.5-math.log10(step))
        self.spinBox.setDecimals(decimals)

        self.subdiv = subdiv
        self.step = step

        self.setRange(minVal, maxVal)
        self.setValue(default)

        self.slider.valueChanged.connect(self.sliderChanged)
        self.spinBox.valueChanged.connect(self.spinChanged)

        self.spinBox.valueChanged.connect(self._paramChanged)

    def setRange(self, minVal, maxVal):
        self.steps = math.ceil((maxVal-minVal) / self.step)
        self.sliderSteps = self.steps * self.subdiv
        self.minVal = minVal
        self.maxVal = maxVal
        self.ratio = self.sliderSteps / (self.maxVal - self.minVal)
        self.spinBox.setRange(minVal, maxVal)
        self.slider.setRange(0, self.sliderSteps)
        self.spinBox.setSingleStep(self.step)
        self.slider.setSingleStep(self.subdiv)
        self.slider.setTickInterval(self.subdiv * 2 if self.sliderSteps//self.subdiv > 10 else self.subdiv)

    def setValue(self, value, silent=False):
        self.spinBox.setValue(value)
        self.slider.setValue(int((value - self.minVal) * self.ratio + 0.5))

    def spinChanged(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int((value - self.minVal) * self.ratio + 0.5))
        self.slider.blockSignals(False)

    def sliderChanged(self, value):
        self.spinBox.setValue(value / self.ratio + self.minVal)


class LogSpinBox(QDoubleSpinBox):
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


class ListSpinBox(QSpinBox):
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

def control_from_param(param):
    kwargs = dict(
        title = param.display_name + ":",
        param = param.name
    )

    for p, k in PARAM_FIELDS.items():
        v = getattr(param, p, None)
        if v is not None:
            kwargs[k] = v

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
    else:
        return None

def param_list_to_vbox(params, vbox):
    param_controls = {}

    sub_vbox = None

    for param in params:
        if isinstance(param, dict):
            # We have subcategories -- make tabs!
            tabs = QTabWidget()
            # tabs.setContentsMargins(5, 5, 5, 5)
            for cat, param_list in param.items():
                tab_vbox = QVBoxLayout()
                tab_vbox.setContentsMargins(5, 5, 5, 5)
                tab_vbox.setSpacing(0)
                param_controls.update(param_list_to_vbox(param_list, tab_vbox))
                widget = QWidget()
                widget.setLayout(tab_vbox)
                tabs.addTab(widget, cat)

                # tabs.setContentsMargins(0, 0, 0, 0)
            vbox.addWidget(tabs)

        elif isinstance(param, str):
            label = QLabel(param)
            label.setObjectName('SectionLabel')
            label.setAlignment(QtCore.Qt.AlignCenter)
            # vbox.addWidget(label)
            # vbox.addWidget(QHLine())

            sub_vbox = QVBoxLayout()
            frame = QFrame()
            frame.setLayout(sub_vbox)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setFrameShadow(QFrame.Sunken)
            sub_vbox.setContentsMargins(0, 0, 0, 0)
            frame.setContentsMargins(2, 2, 2, 2)
            vbox.addWidget(frame)
            sub_vbox.addWidget(label)
            # sub_vbox.addWidget(QHLine())

        else:
            # This is just an item
            # print(param.name)
            control = control_from_param(param)
            if control is not None:
                if sub_vbox is None:
                    vbox.addWidget(control)
                else:
                    sub_vbox.addWidget(control)
                param_controls[param.name] = control

    return param_controls


class VolumetricView(QOpenGLWidget):
    frameChanged = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, volume=None):
        fmt = QSurfaceFormat()
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)

        super().__init__(parent)

        self.setUpdateBehavior(1)
        self.last_t = time.time()

        self.parent = parent
        self.lastPos = QtCore.QPoint()

        self.timer = QtCore.QTimer()
        self.timer.setInterval(0)
        self.timer.timeout.connect(self.update)
        # self.timer.start()

        self._isPlaying = False
        self.hasUpdate = True

        self.makeCurrent()
        self.view = view.View()
        self.doneCurrent()

        self._updates = {}
        # if volume is not None:
            # self.attachVolume(volume)

    def setPlaying(self, isPlaying):
        if isPlaying:
            self.play()
        else:
            self.pause()

    def play(self):
        if hasattr(self.view, 'volume'):
            self._isPlaying = True
            self._tStart = time.time()
            self.timer.start()

    def pause(self):
        self._isPlaying = False
        self.timer.stop()

    def attachVolume(self, volume):
        self.makeCurrent()
        self.volume = volume
        self.view.attach_volume(volume)
        self.view.draw()
        self.doneCurrent()

    def minimumSizeHint(self):
        return QtCore.QSize(300, 300)

    def sizeHint(self):
        return QtCore.QSize(800, 600)

    def initializeGL(self):
        try:
            # This sets the output to be sRGB -- unfortunately doesn't work with old versions of Qt!
            # As a result, the opengl renderer must create an offscreen buffer
            #   to acheive the same result. ):
            # self.setTextureFormat(GL_SRGB8_ALPHA8)
            self.dpr = self.devicePixelRatio()
            self.view.resize(width=self.width()*self.dpr,
                height=self.height()*self.dpr)
        except:
            traceback.print_exc()
            self.parent.close()

    def paintGL(self):
        try:
            if self._isPlaying:
                t = time.time()
                fps = self.view.params['framerate']
                advance = int((t - self._tStart) * fps)
                if advance:
                    frame = (self.view.params['frame'] + advance) % len(self.view.volume)
                    self.updateParam('frame', frame)
                    self._tStart += advance / fps
                    self.frameChanged.emit(frame)

            # print(t - self.last_t)
            # dt = t - self.last_t
            # self.last_t = t

            if self.hasUpdate:
                self._updateParams()
                self.view.draw()

        except:
            traceback.print_exc()
            self.parent.close()

    def resizeGL(self, width, height):
        self.view.resize(width*self.dpr, height*self.dpr)
        self.hasUpdate = True

    def mousePressEvent(self, event):
        self.lastPos = event.pos()
        self.view.buttons_pressed = int(event.buttons())
        self.hasUpdate = True
        self.update()

    def mouseReleaseEvent(self, event):
        self.view.buttons_pressed = int(event.buttons())
        self.hasUpdate = True
        self.update()

    def mouseMoveEvent(self, event):
        x = event.x()
        y = event.y()
        dx = x - self.lastPos.x()
        dy = y - self.lastPos.y()
        self.lastPos = event.pos()
        self.view.buttons_pressed = int(event.buttons())
        if dx or dy:
            self.view.mouse_move(x*self.dpr, y*self.dpr, dx*self.dpr, dy*self.dpr)
            self.hasUpdate = True
            self.update()

    def wheelEvent(self, event):
        self.view.params['scale'] *= 1.25**(event.angleDelta().y()/120)
        self.hasUpdate = True
        self.update()

    def updateParams(self, **kwargs):
        self._updates.update(**kwargs)
        self.hasUpdate = True
        self.update()

    def updateHiddenParams(self, **kwargs):
        self._updates.update(**kwargs)
        self.hasUpdate = True

    def updateParam(self, p, v):
        self.updateParams(**{p:v})

    def updateHiddenParam(self, p, v):
        self.updateHiddenParams(**{p:v})

    def _updateParams(self):
        if self.hasUpdate:
            self.view.update_params(**self._updates)
            self._updates = {}
            self.hasUpdate = False

    def previewImage(self):
        self.makeCurrent()
        self._updateParams()
        img = self.view.draw(offscreen=True, return_image=True)
        self.doneCurrent()

        return img

    def saveImage(self, fn=None, dir=None):
        if dir is None:
            dir = os.getcwd()
        if fn is None:
            fns = glob.glob(os.path.join(dir, 'muvi_screenshot_*.png'))
            for i in range(10**4):
                fn = os.path.join(dir, 'muvi_screenshot_%08d.png' % i)
                if fn not in fns:
                    break

        self.makeCurrent()
        self._updateParams()
        img = self.view.draw(offscreen=True, save_image=fn)
        self.doneCurrent()

        return fn, img
