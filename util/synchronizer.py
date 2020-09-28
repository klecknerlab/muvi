import matplotlib as mpl
mpl.use('Qt5Agg')
import numpy as np
# from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QGridLayout,
        QMainWindow, QVBoxLayout, QLabel, QProgressBar, QPushButton, QWidget,
        QSpinBox, QDoubleSpinBox, qApp, QAction)
from PyQt5.QtGui import (QIcon)
import textwrap
import sys
from muvi.drivers import Synchronizer, reset_fpga

# Set the name in the menubar.
if sys.platform.startswith('darwin'):
    try:
        from Foundation import NSBundle
        bundle = NSBundle.mainBundle()
        if bundle:
            # app_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
            app_info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
            if app_info:
                app_info['CFBundleName'] = 'MUVI Scan Synchronizer'
    except ImportError:
        pass

class Plot(FigureCanvas):
    def __init__(self, parent=None, width=4, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_axes((0.1, 0.15, 0.8, 0.8))
        super().__init__(fig)


class MainWindow(QMainWindow):
    TICKS = 48000000
    SUBDIV = 1
    PULSE_TICKS = 200

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.setWindowTitle('MUVI Scan Synchronizer')
        self.setWindowIcon(QIcon('test_icon.png'))

        self.statusBar().showMessage('Synchronizer not connected.')

        # quit = QAction(QIcon('exit.png'), '&Exit', self)
        # quit.setShortcut('Ctrl+Q')
        # quit.setStatusTip('Exit application')
        # quit.triggered.connect(qApp.quit)

        save_xml = QAction('&Save XML', self)
        save_xml.setShortcut('Ctrl+S')
        save_xml.triggered.connect(self.save_xml)

        connect = QAction('&Connect Synchronizer', self)
        connect.setShortcut('Ctrl+C')
        connect.triggered.connect(self.check_sync)


        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        file_menu.addAction(save_xml)
        file_menu.addAction(connect)


        self.central_widget = QWidget()
        self.grid = QGridLayout()
        self.gridloc = 0

        self.grid2 = QGridLayout()
        self.vbox = QVBoxLayout()
        self.vbox.addLayout(self.grid)
        self.vbox.addLayout(self.grid2)

        self.plot = Plot()
        self.vbox.addWidget(self.plot, 1)

        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.vbox)

        self.control_frame_rate = self.add_control('Frame Rate (kHz):', 75, min=0.1, max=500, step=1, decimals=3,
            tip='The camera frame rate; usually the maximum framerate for a given resolution on your camera.  The real triggered frame rate will be slightly less.')
        self.control_galvo_t1 = self.add_control('Galvo Return Time (ms):', 1, min=0, max=100, step=0.25, decimals=2,
            tip='The time required for the galvo to return to the beginning of the scan.')
        self.control_galvo_t2 = self.add_control('Galvo Stabilization Time (ms):', 0.25, min=0, max=100, step=0.25, decimals=2,
            tip='The time to wait after the start of the scan to start taking frames; used to allow the galvo motion to settle down.')
        self.control_fpv = self.add_control('Frames per Volume:', 512, min=0, max=2048, step=128, decimals=False,
            tip='The number of frames in the depth direction of your desired volume.')

        self.display_vps = self.add_display('Volume Rate:',
            tip='The number of volumes captured per second.')
        self.display_real_frame_rate = self.add_display('Real Frame Rate:',
            tip='The actual triggered framerate of the camera; will be slightly less than the requested frame rate to match the electronics time base.')
        self.display_scanf = self.add_display('Frames per Scan:',
            tip='The number of frames in the scan, including non-active frames (when the sheet returns to the start of the scan).')
        self.display_fgf = self.add_display('Func. Gen. Frequency:',
            tip='The frequency which should be set on the function generator.')
        self.display_fgs = self.add_display('Func. Gen. Symmetry:',
            tip='The "symmetry" of the triangle wave which should be set on the function generator.')

        self.button_cdf = self.add_button('Capture Dead Frames', (0, 0), toggle=True, func=self.update_output,
            tip='If toggled, capture image frames at all times, rather than just during the scan.')
        self.button_am = self.add_button('Alignment Mode', (0, 1), toggle=True, func=self.update_output,
            tip='If toggled, only output laser pulses at the beginning/end of the scan.')
        self.button_tc = self.add_button('Two Color', (0, 2), toggle=True, func=self.update_output,
            tip='If toggled, enter two color mode')
        self.button_start = self.add_button('Start', (1, 0, 1, 3), func=self.start_clicked,
            tip='Start/stop the scan output.')


        self.retry_timer = QtCore.QTimer(self)

        self.sync = None
        self.running = False
        self.check_sync()

        self.update_output()



    def save_xml(self):
        print('Not yet implemented!')


    def add_control(self, label, value=1, min=0, max=100, step=1, decimals=2, update=True, tip=None):
        if decimals:
            box = QDoubleSpinBox()
        else:
            box = QSpinBox()

        box.setRange(min, max)
        box.setSingleStep(step)
        if decimals: box.setDecimals(decimals)
        box.setValue(value)

        if update:
            box.valueChanged.connect(self.update_output)

        label = QLabel(label)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)

        self.grid.addWidget(label, self.gridloc, 0)
        self.grid.addWidget(box, self.gridloc, 1)
        self.gridloc += 1

        # tip = text_wrap(tip)
        # print(tip)

        if tip is not None:
            tip = "<FONT COLOR=black>" + tip + "<\FONT>"
            box.setToolTip(tip)
            label.setToolTip(tip)

        return box

    def add_display(self, label, default='-', tip=None):
        label = QLabel(label)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.grid.addWidget(label, self.gridloc, 0)

        display = QLabel('-')
        self.grid.addWidget(display, self.gridloc, 1)
        self.gridloc += 1

        if tip is not None:
            tip = "<FONT COLOR=black>" + tip + "<\FONT>"
            display.setToolTip(tip)
            label.setToolTip(tip)

        return display

    def add_button(self, label, gridloc, func=None, toggle=False, pressed=False, tip=None):
        button = QPushButton(label)

        if tip is not None:
            tip = "<FONT COLOR=black>" + tip + "<\FONT>"
            button.setToolTip(tip)

        if toggle:
            button.setCheckable(True)
            button.setChecked(pressed)
            button.setStyleSheet("QPushButton::checked{background-color:#66F;}")

        if func is not None:
            if toggle:
                button.toggled.connect(func)
            else:
                button.clicked.connect(func)

        self.grid2.addWidget(button, *gridloc)

        return button

    def start_clicked(self):
        if self.running:
            self.stop_output()
        else:
            self.start_output()

    def stop_output(self):
        self.button_start.setText('Start Output')
        self.running = False
        if self.sync is not None:
            try:
                self.sync.stop()
                self.statusBar().showMessage('Synchronizer connected, but not running.')
            except:
                self.sync = None
                self.statusBar().showMessage('Synchronizer disconnected! (File \u2192 Connect Synchronizer)')


    def start_output(self):
        if self.sync is not None:
            try:
                self.sync.start()
                self.running = True
                self.button_start.setText('Stop Output')
                self.statusBar().showMessage('Synchronizer connected and running.')
            except:
                self.sync = None
                self.statusBar().showMessage('Synchronizer disconnected! (File \u2192 Connect Synchronizer)')


    def check_sync(self):
        if self.sync is None:
            try:
                self.sync = Synchronizer(reset=False)
                self.stop_output()
                self.initialize_output()
                self.update_sync_output()

            except:
                self.sync = None
                if reset_fpga():
                    self.statusBar().showMessage('Resetting synchronizer, trying to reconnect...')
                    self.retry_timer.singleShot(1000, self.check_sync)
                else:
                    self.statusBar().showMessage('Synchronizer not found. (File \u2192 Connect Synchronizer)')


    def initialize_output(self):
        if self.sync is not None:
            for n in range(4):
                self.sync.digital_setup(n, active=True, inverted=(n in (2, 3)))


    def update_sync_output(self):
        if self.sync is not None and hasattr(self, '_channels'):
            if not hasattr(self, 'pn'):
                self.pn = 0

            for i, setup in enumerate(self._channels):
                self.sync.pulse_setup(i, self.pn, *setup)
                # print(i, self.pn, *setup)

            self.sync.select_program(self.pn)
            self.sync.cycle_setup(self.ticks_per_cycle)
            self.sync.update()

            self.pn = (self.pn + 1) % 2


    def update_output(self):
        fps = self.control_frame_rate.value() * 1E3
        t1 = self.control_galvo_t1.value() * 1E-3
        t2 = self.control_galvo_t2.value() * 1E-3
        fpv = self.control_fpv.value()

        ticks = int(np.ceil(self.TICKS / (self.SUBDIV * fps))) * self.SUBDIV
        rfps = self.TICKS / ticks
        self.display_real_frame_rate.setText(f'{rfps/1E3:0.3f} kHz')

        t1f = int(np.ceil(t1 * rfps))
        t2f = int(np.ceil(t2 * rfps))

        two_color = self.button_tc.isChecked()
        num_colors = 2 if two_color else 1

        # Frames per scan: fpv + galvo recovery times + 1 extra at the end
        scanf = fpv*num_colors + t1f + t2f + 1

        self.display_scanf.setText(f'{scanf:d}')
        self.display_vps.setText(f'{rfps/scanf:0.3f} Hz')
        self.display_fgf.setText(f'{rfps/(scanf-1):0.3f} Hz')
        self.display_fgs.setText(f'{100*(scanf-t1f-1)/(scanf-1):0.3f} %')

        # print(ticks, t1f, t2f)

        capture_dead = self.button_cdf.isChecked()
        alignment_mode = self.button_am.isChecked()

        # hi = self.PULSE_TICKS
        hi = ticks//2
        lo = ticks-hi


        self._channels = []
        self._channels.append((1, 0, hi, lo))
        if capture_dead:
            self._channels.append((scanf, 0, hi, lo))
        else:
            self._channels.append((fpv*num_colors, t2f*ticks, hi, lo))
        if alignment_mode:
            if two_color:
                self._channels.append((2, t2f*ticks, hi, lo + ticks*(2*(fpv-2) + 1)))
                self._channels.append((2, (t2f+1)*ticks, hi, lo + ticks*(2*(fpv-2) + 1)))
            else:
                self._channels.append((2, t2f*ticks, hi, lo + ticks*(fpv-2)))
                self._channels.append((0, 0, 0, 0))
        else:
            if two_color:
                self._channels.append((fpv, t2f*ticks, hi, lo+ticks))
                self._channels.append((fpv, (t2f+1)*ticks, hi, lo+ticks))
            else:
                self._channels.append((fpv, t2f*ticks, hi, lo))
                self._channels.append((0, 0, 0, 0))
        # self._channels.append((t2f*ticks, 1, hi, lo))

        self.ticks_per_cycle = ticks * scanf

        nc = len(self._channels)

        colors=['r', 'y', 'g', 'b', 'm']

        axes = self.plot.axes

        axes.cla()

        for i, (cycles, delay, hi, lo) in enumerate(self._channels):
            dt = (hi + lo)/ticks
            hif = hi / ticks
            d1 = delay/ticks

            y0 = nc - 1 - i

            if cycles == 2:
                d2 = d1 + dt
                x = np.array([0, d1, d1, d1+1, d1+1, d2, d2, d2+1, d2+1, scanf], 'd')
                y = np.array([0, 0,  1,  1,    0,    0,    1,    1,  0,  0])*0.7
            elif cycles:
                d2 = d1 + dt*cycles
                x = np.array([0, d1, d1, d2, d2, scanf], 'd')
                y = np.array([0, 0,  1,  1,  0,  0])*0.7
            else:
                x = np.array([0, scanf], 'd')
                y = np.zeros(2)

            x *= 1000/fps
            c = colors[i]

            axes.plot(x, y + y0, color=c)
            axes.fill_between(x, y0, y+y0, color=c, alpha=0.5)



        y0 = nc
        x = np.array([0, t2f+fpv*num_colors, scanf-1, scanf], 'd') * 1000/fps
        y = np.array([0, 1, 0, 0]) * 2
        c = 'k'
        axes.plot(x, y + y0, color=c)
        axes.fill_between(x, y0, y+y0, color=c, alpha=0.5)

        axes.set_xlabel('Time (ms)')
        ticks = np.arange(nc + 1) + 0.35
        ticks[-1] += 0.65
        axes.set_yticks(ticks)
        axes.set_yticklabels([f'D{nc-i}' for i in range(nc)] + ['FG'])

        axes.set_xlim(0, scanf * 1000/fps)

        self.plot.draw()

        self.update_sync_output()





if __name__ == '__main__':
    app = QApplication([])
    app.setStyle('Fusion')
    win = MainWindow()
    win.show()
    app.exec_()
