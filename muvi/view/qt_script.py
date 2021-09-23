from PyQt5 import QtCore, QtGui
from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QAction, QMenu, \
    QFileDialog, QVBoxLayout, QWidget
import os
Qt = QtCore.Qt
import ast
import numpy as np
import json
import re


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


class Keyframe(QListWidgetItem):
    def __init__(self, params, label=None, parent=None):
        if label is None:
            label = f'Keyframe #{parent.count()+1}'
        super().__init__(label, parent)
        self.setData(Qt.UserRole, params)
        self.setFlags(self.flags() | Qt.ItemIsEditable | Qt.ItemIsDragEnabled)


class KeyframeEditor(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.mainWindow = parent

        self.vbox = QVBoxLayout()
        self.vbox.setContentsMargins(5, 5, 5, 5)
        self.keyframeList = KeyframeList(self)
        self.vbox.addWidget(self.keyframeList)
        self.setLayout(self.vbox)

    def addKeyframe(self, event=None, data=None, label=None):
        if data is None:
            data = self.mainWindow.allParams()
        Keyframe(data, label, self.keyframeList)

    def saveScript(self, event=None, fn=None):
        if fn is None:
            fn, ext = QFileDialog.getSaveFileName(self, 'Save MUVI Script File',
                os.getcwd(), "MUVI script (*.muvi_script)")

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

        assets = self.mainWindow.display.view.assetSpec()

        setup = dict(assets=assets)
        if hasattr(self.mainWindow, 'getExportSettings'):
            setup.update(self.mainWindow.getExportSettings())

        data = {
            "setup": setup,
            "frames": frames,
        }

        with open(fn, 'wt') as f:
            f.write(re.sub('"!@<@!(.*)!@>@!"', lambda m: m.group(1),
                json.dumps(data, indent=2, cls=JSONEncoder))
            )

class KeyframeList(QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setDragDropMode(self.InternalMove)

        # self.currentRowChanged.connect(self.testPrint)

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.contextMenu)
        self.addOption = QAction('New Keyframe (from current view)')
        self.addOption.triggered.connect(self.parent.addKeyframe)
        self.showOption = QAction('Show')

    def contextMenu(self, point):
        item = self.itemAt(point)

        menu = QMenu()
        menu.addAction(self.addOption)

        if item is not None:
            menu.addAction(self.showOption)

        action = menu.exec_(QtGui.QCursor.pos())

        if action == self.showOption:
            self.parent.mainWindow.display.updateParams(item.data(Qt.UserRole), callback=True)
