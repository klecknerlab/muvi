Module muvi.view.qtview
=======================

Functions
---------

    
`fromQColor(qc, has_alpha)`
:   

    
`toQColor(t)`
:   

    
`view_volume(vol, window_name='Volumetric Viewer')`
:   

Classes
-------

`BoolViewSetting(gl_display, varname, default, parent=None)`
:   QCheckBox(parent: QWidget = None)
    QCheckBox(str, parent: QWidget = None)

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QCheckBox
    * PyQt5.QtWidgets.QAbstractButton
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

`CollapsableVBox(parent=None, title='', isOpen=False)`
:   QWidget(parent: QWidget = None, flags: Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags())

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `addWidget(self, *args, **kwargs)`
    :

    `add_row(self, *args)`
    :

    `toggle(self, checked)`
    :

`ColorViewSetting(gl_display, varname, default, parent=None)`
:   QFrame(parent: QWidget = None, flags: Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags())

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QFrame
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, QMouseEvent)

    `setValue(self, color)`
    :

`IntViewSetting(gl_display, varname, default, minval, maxval, rollover=True, parent=None, force_update=True)`
:   QSpinBox(parent: QWidget = None)

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QSpinBox
    * PyQt5.QtWidgets.QAbstractSpinBox
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Descendants

    * muvi.view.qtview.ListViewSetting

    ### Methods

    `advance(self)`
    :

`LinearViewSetting(gl_display, varname, default, minval, maxval, step, decimals=None, parent=None)`
:   QDoubleSpinBox(parent: QWidget = None)

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QDoubleSpinBox
    * PyQt5.QtWidgets.QAbstractSpinBox
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Descendants

    * muvi.view.qtview.LogViewSetting

    ### Methods

    `set_from_slider(self, val, slider_max)`
    :

`ListViewSetting(gl_display, varname, default, vals, parent=None, force_update=True)`
:   QSpinBox(parent: QWidget = None)

    ### Ancestors (in MRO)

    * muvi.view.qtview.IntViewSetting
    * PyQt5.QtWidgets.QSpinBox
    * PyQt5.QtWidgets.QAbstractSpinBox
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `stepBy(self, step)`
    :   stepBy(self, int)

`LogViewSetting(gl_display, varname, default, minval, maxval, step, decimals=None, change_func=None, parent=None)`
:   QDoubleSpinBox(parent: QWidget = None)

    ### Ancestors (in MRO)

    * muvi.view.qtview.LinearViewSetting
    * PyQt5.QtWidgets.QDoubleSpinBox
    * PyQt5.QtWidgets.QAbstractSpinBox
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `stepBy(self, step)`
    :   stepBy(self, int)

`PlayButton(timer, labels=['Play', 'Pause'], parent=None)`
:   QPushButton(parent: QWidget = None)
    QPushButton(str, parent: QWidget = None)
    QPushButton(QIcon, str, parent: QWidget = None)

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QPushButton
    * PyQt5.QtWidgets.QAbstractButton
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `toggle(self)`
    :   toggle(self)

`ViewWidget(parent=None, volume=None)`
:   QOpenGLWidget(parent: QWidget = None, flags: Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags())

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QOpenGLWidget
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `initializeGL(self)`
    :   initializeGL(self)

    `minimumSizeHint(self)`
    :   minimumSizeHint(self) -> QSize

    `mouseMoveEvent(self, event)`
    :   mouseMoveEvent(self, QMouseEvent)

    `mousePressEvent(self, event)`
    :   mousePressEvent(self, QMouseEvent)

    `mouseReleaseEvent(self, event)`
    :   mouseReleaseEvent(self, QMouseEvent)

    `paintGL(self)`
    :   paintGL(self)

    `resizeGL(self, width, height)`
    :   resizeGL(self, int, int)

    `save_image(self)`
    :

    `sizeHint(self)`
    :   sizeHint(self) -> QSize

    `update_view_settings(self, force_update=True, **kwargs)`
    :

    `wheelEvent(self, event)`
    :   wheelEvent(self, QWheelEvent)

    `xRotationChanged(...)`
    :

    `yRotationChanged(...)`
    :

    `zRotationChanged(...)`
    :

`ViewerApp(window_name='Volumetric Viewer', volume=None)`
:   QMainWindow(parent: QWidget = None, flags: Union[Qt.WindowFlags, Qt.WindowType] = Qt.WindowFlags())

    ### Ancestors (in MRO)

    * PyQt5.QtWidgets.QMainWindow
    * PyQt5.QtWidgets.QWidget
    * PyQt5.QtCore.QObject
    * sip.wrapper
    * PyQt5.QtGui.QPaintDevice
    * sip.simplewrapper

    ### Methods

    `keyPressEvent(self, event)`
    :   Close application from escape key.
        
        results in QMessageBox dialog from closeEvent, good but how/why?

    `toggle_view(self)`
    :