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

from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAction, QMenu, \
    QFileDialog, QVBoxLayout, QWidget, QMessageBox
from .qtview_widgets import paramListToVBox
from .params import PARAMS, PARAM_CATEGORIES
import os
Qt = QtCore.Qt
import ast
import numpy as np
import json
import re
import traceback
from scipy.spatial.transform import Rotation, Slerp
from .ogl import cameraMatrix, mag, norm


def unArray(d):
    return {k:(v.tolist() if hasattr(v, 'tolist') else v) for k, v in d.items()}

def reArray(d):
    return {k:(np.array(v) if isinstance(v, (list, tuple)) else v) for k, v in d.items()}

def arrayEquiv(x, y):
    result = x == y
    if isinstance(result, np.ndarray):
        return result.all()
    return result

class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return f'!@<@!{obj.tolist()}!@>@!'
        return super().default(obj)


_assetRe = re.compile('\#([0-9]+)_(.*)')
_copyRe = re.compile('(.*)\s+Copy(\s+[0-9]+)?')


def smoothStep(x):
    x = np.clip(x, 0, 1)
    return 3*x**2 - 2*x**3

def smootherStep(x):
    x = np.clip(x, 0, 1)
    return 6*x**5 - 15*x**4 + 10*x**3


def rotAxis(X, ax, phi):
    a1 = (ax + 1) % 3
    a2 = (ax + 2) % 3

    Xr = X.copy()
    s, c = np.sin(phi), np.cos(phi)

    Xr[..., a1] = c * X[..., a1] - s * X[..., a2]
    Xr[..., a2] = s * X[..., a1] + c * X[..., a2]

    return Xr


class Keyframe(QListWidgetItem):
    def __init__(self, params, label=None, parent=None):
        if label is None:
            if "_label" in params:
                label = params.pop('_label')
            else:
                label = f'Keyframe'
        super().__init__(label, parent)
        self.setData(Qt.UserRole, params)
        self.setFlags(self.flags() | Qt.ItemIsEditable | Qt.ItemIsDragEnabled)

    def relabelAssets(self, relabel):
        newData = {}
        for param, val in self.data(Qt.UserRole).items():
            m = _assetRe.match(param)
            if m:
                id = int(m.group(1))
                if id in relabel:
                    newData[f'#{relabel[id]}_{m.group(2)}'] = val
                else:
                    newData[param] = val
            else:
                newData[param] = val
        self.setData(Qt.UserRole, newData)


class KeyframeEditor(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.mainWindow = parent

        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(5, 5, 5, 5)
        self.keyframeList = KeyframeList(self)
        self.vbox.addWidget(self.keyframeList)

        self.controls = paramListToVBox(PARAM_CATEGORIES['Keyframe'], self.vbox)
        for param, control in self.controls.items():
            control.paramChanged.connect(self.updateKeyframe)

        self.setLayout(self.vbox)
        self.isSaved = True

    def addKeyframe(self, event=None, data=None, label=None):
        if data is None:
            data = self.mainWindow.allParams()
            for param, control in self.controls.items():
                data[param] = control.value()

        if label is None:
            label = f'Keyframe #{self.keyframeList.count()+1}'

        row = self.keyframeList.currentRow() + 1
        self.keyframeList.insertItem(row, Keyframe(data, label))
        self.keyframeList.setCurrentRow(row)

        self.isSaved = False

    def updateKeyframe(self, param, value):
        item = self.keyframeList.currentItem()
        if item is not None:
            data = item.data(Qt.UserRole)
            data[param] = value
            item.setData(Qt.UserRole, data)
        self.isSaved = False

    def selectRow(self, row):
        if row < 0:
            return

        keyframe = self.keyframeList.item(row)
        params = keyframe.data(Qt.UserRole)
        for param, control in self.controls.items():
            if param in params:
                control.setValue(params[param])

    def openScript(self, fn):
        if not self.isSaved:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Question)
            msg.setText("The current script has unsaved data!  Save before opening a new one?")
            msg.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel);
            msg.setDefaultButton(QMessageBox.Save);
            ret = msg.exec()
            if ret == QMessageBox.Cancel:
                return
            elif ret == QMessageBox.Save:
                self.saveScript()

        with open(fn, 'rt') as f:
            data = json.load(f)

        # Do as much as we can before clearing the old data out
        # This way if it throws an error it doesn't corrupt existing
        #  stuff!
        setup = data['setup']
        frames = data['frames']

        bdir, bfn = os.path.split(fn)
        # print(bdir, bfn)
        assets = {int(k):os.path.abspath(os.path.join(bdir, v)) for k, v in setup['assets'].items()}
        # assets = {int(k):os.path.join(bdir, v) for k, v in setup['assets'].items()}
        # print(assets)

        # Parse all the frame data (a list of params) *without* actually
        #   changing anything.  We will create a list of items but not
        #   actually
        keyframes = []
        unknownParams = set()

        data = {}
        for params in frames:
            data['_label'] = f'Keyframe #{len(keyframes)+1}'
            for param, val in params.items():
                if isinstance(val, (tuple, list)):
                    val = np.array(val)
                m = _assetRe.match(param)
                if m:
                    name = m.group(2)
                    id = int(m.group(1))
                    if id not in assets:
                        unknownParams.add(param)
                    elif name in PARAMS:
                        data[param] = val
                    else:
                        unknownParams.add(name)
                else:
                    if param in PARAMS:
                        data[param] = val
                    else:
                        unknownParams.add(param)
            keyframes.append(Keyframe(data.copy()))

        if unknownParams:
            print(unknownParams)

        relabel = self.mainWindow.openAssets(assets)
        relabel = {old:new for old, new in relabel.items() if old != new}

        self.keyframeList.clear()

        for item in keyframes:
            if relabel:
                item.relabelAssets(relabel)
            self.keyframeList.addItem(item)

        if keyframes:
            self.keyframeList.setCurrentRow(0)

        self.isSaved = True

        # Exceptions handled in main class... no need to do it here!
        #  (in fact, if you do it screws things up there!)
        # except Exception as e:
        #     ec = e.__class__.__name__
        #     msg = QMessageBox()
        #     msg.setIcon(QMessageBox.Critical)
        #     msg.setWindowTitle(str(ec))
        #     msg.setText(str(ec) + ": " + str(e))
        #     msg.setDetailedText(traceback.format_exc())
        #     msg.setStyleSheet("QTextEdit {font-family: Courier; min-width: 600px;}")
        #     msg.setStandardButtons(QMessageBox.Cancel)
        #     msg.exec_()


    def saveScript(self, event=None, fn=None):
        if fn is None:
            fn, ext = QFileDialog.getSaveFileName(self, 'Save MUVI Script File',
                os.getcwd(), "MUVI script (*.muvi_script)")
            if not fn:
                return

        last = None
        frames = []
        for row in range(self.keyframeList.count()):
            item = self.keyframeList.item(row)
            params = {
                '_label':item.text(),
            }
            params.update(item.data(Qt.UserRole))

            if last is None:
                last = params.copy()
                frames.append(params)
            else:
                new = {k:v for k, v in params.items() if not arrayEquiv(last[k], v)}
                frames.append(new)
                last.update(new)

        bdir, bfn = os.path.split(os.path.abspath(fn))

        setup = dict(assets={
            k: os.path.relpath(v, bdir)
            for k, v in self.mainWindow.display.view.assetSpec().items()
            if v is not None
        })
        if hasattr(self.mainWindow, 'getExportSettings'):
            setup.update(self.mainWindow.getExportSettings())

        data = {
            "setup": setup,
            "frames": list(map(unArray, frames)),
        }

        with open(fn, 'wt') as f:
            f.write(re.sub('"!@<@!(.*)!@>@!"', lambda m: m.group(1),
                json.dumps(data, indent=2, cls=JSONEncoder))
            )

        self.isSaved = True

    def frames(self):
        frames = []
        params = self.mainWindow.allParams()
        interp = 'smooth'
        camera = 'object'
        spin = 'none'

        for row in range(self.keyframeList.count()):
            num = -1
            updates = {}

            item = self.keyframeList.item(row)
            for param, val in item.data(Qt.UserRole).items():
                if param.startswith("_"):
                    if param == '_frames':
                        num = val
                    elif param == '_interp':
                        interp = val
                    elif param == '_camera':
                        camera = val
                    elif param == '_spin':
                        spin = val
                elif not arrayEquiv(val, params.get(param, None)):
                    if isinstance(val, (list, tuple)):
                        val = np.array(val)
                    updates[param] = val

            if num == -1:
                if not frames:
                    num = 1
                elif 'frame' in updates:
                    num = abs(updates['frame'] - params['frame'])
                else:
                    num = 30

            # print(num, updates)

            if num == 1:
                frames.append(updates)
            elif num > 1:
                f = [{} for n in range(num)]

                if interp == "smooth":
                    interp_func = smoothStep
                elif interp == "smoother":
                    interp_func = smootherStep
                else:
                    interp_func = lambda x: x

                # If the camera is moving, handle that specially!
                interp_vals = interp_func(np.arange(1, num+1, dtype='d')/num)

                if 'camera_pos' in updates or 'look_at' in updates or 'up' in updates:
                    def get_param(var):
                        return params[var], updates.get(var, params[var])

                    c0, c1 = get_param('camera_pos')
                    l0, l1 = get_param('look_at')
                    u0, u1 = get_param('up')
                    d0 = mag(c0 - l0)
                    d1 = mag(c1 - l1)


                    if camera == 'object':
                        R = np.zeros((2, 3, 3), dtype='d')
                        R[0] = cameraMatrix(c0, l0, u0)[:3, :3]
                        R[1] = cameraMatrix(c1, l1, u1)[:3, :3]
                        R = Rotation.from_matrix(R)
                        slerp = Slerp([0, 1], R)
                        for i, x in enumerate(interp_vals):
                            Ri = slerp(x).as_matrix()
                            f[i]['look_at'] = l0 * (1-x) + l1 * x
                            d = d0 * (1-x) + d1 * x
                            f[i]['camera_pos'] = f[i]['look_at'] + d * Ri[:, 2]
                            f[i]['up'] = Ri[:, 1]

                    else: # Direct interpolation mode
                        for i, x in enumerate(interp_vals):
                            f[i]['camera_pos'] = c0 * (1-x) + c1 * x
                            f[i]['up'] = u0 * (1-x) + u1 * x
                            f[i]['look_at'] = l0 * (1-x) + l1 * x

                for param, val1 in updates.items():
                    val0 = params[param]

                    if param == 'frame':
                        for i in range(num):
                            x = (i+1)/num # Always has linear interpolation!
                            f[i][param] = int(val0 * (1-x) + val1 * x + 0.5)
                    elif param in {'camera_pos', 'up', 'look_at'}:
                        # Note that look_at is handled normally!
                        continue
                    elif isinstance(val0, (float, np.ndarray)):
                        for i, x in enumerate(interp_vals):
                            f[i][param] = val0 * (1-x) + val1 * x
                    else:
                        for i in range(num):
                            f[i][param] = val1

                if spin != 'none':
                    cp = params['camera_pos']
                    la = params['look_at']
                    up = params['up']

                    dir = -1 if spin.startswith('+') else +1
                    if spin[1] == 'x':
                        ax = 0
                    elif spin[1] == 'y':
                        ax = 1
                    else:
                        ax = 2

                    for i, x in enumerate(interp_vals):
                        phi = dir * x * 2 * np.pi
                        cf = f[i]
                        cp = cf.get('camera_pos', cp)
                        la = cf.get('look_at', la)
                        up = cf.get('up', up)

                        cf['camera_pos'] = rotAxis(cp - la, ax, phi) + la
                        cf['up'] = rotAxis(up, ax, phi)

                frames += f


            params.update(updates)

        return frames

class KeyframeList(QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setDragDropMode(self.InternalMove)

        self.currentRowChanged.connect(parent.selectRow)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.contextMenu)
        self.addOption = QAction('New Keyframe (from current view)')
        self.addOption.triggered.connect(parent.addKeyframe)
        self.deleteOption = QAction('Delete')
        self.duplicateOption = QAction('Duplicate')
        self.showOption = QAction('Show')
        self.copyOption = QAction('Copy Parameters to Clipboard')


    def contextMenu(self, point):
        item = self.itemAt(point)

        menu = QMenu()
        menu.addAction(self.addOption)

        if item is not None:
            menu.addAction(self.deleteOption)
            menu.addAction(self.duplicateOption)
            menu.addAction(self.showOption)
            menu.addAction(self.copyOption)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == self.showOption:
            self.parent().mainWindow.display.updateParams(item.data(Qt.UserRole), callback=True)
        elif action == self.copyOption:
            clipboard = self.parent().mainWindow.clipboard
            if clipboard is not None:
                clipboard.setText(re.sub('"!@<@!(.*)!@>@!"', lambda m: m.group(1),
                    json.dumps(item.data(Qt.UserRole), indent=2, cls=JSONEncoder)))
        elif action == self.deleteOption:
            self.takeItem(self.row(item))
        elif action == self.duplicateOption:
            params = item.data(Qt.UserRole)
            self.parent().addKeyframe(params)
