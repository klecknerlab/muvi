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

'''This modules contains a driver for the FPGA-based signal synchronizer.
'''

import struct
import platform
import time

if platform.system() == 'Windows':
    USE_SERIAL = True
    import serial
    from serial.tools.list_ports import comports

    def find_usb_ports(vid, pid):
        return [dev for dev in comports() if (dev.vid == vid and dev.pid == pid)]

else:
    USE_SERIAL = False
    import usb.core

    def find_usb_ports(vid, pid):
        return(list(usb.core.find(find_all=True, idVendor=vid, idProduct=pid)))



def reset_fpga(ids=((0x1d50, 0x6130), (0x1209, 0x2100))):
    '''Try putting TinyFPGA boards into user mode.

    Keywords
    --------
    ids: iterable of tuples of 2 integeres
        The vendor and product ids of the USB devices.  THe defaults are usually
        what you want.

    Returns
    -------
    num_reset : int
        The number of devices detected and reset.  Will return 0 if none found.
    '''
    reset = 0

    for (vid, pid) in ids:
        for dev in find_usb_ports(vid, pid):
            if USE_SERIAL:
                with serial.Serial(dev.device) as p:
                    p.write(b'\x00')
            else:
                dev.configurations()[0].interfaces()[1].endpoints()[0].write('\x00')
            reset += 1

    return reset


class SynchronizerError(Exception):
    pass


class Synchronizer:
    ticks_per_s = 48000000
    analog_max = 4

    '''An FPGA based pulse generator, developed by the Kleckner Lab at UC
    Merced.

    In general, the output is organized into `cycles', the length of which is
    controlled with `cycle_count'.  All time units are specified in "ticks", of
    which there are exactly 48,000,000 per second (the FPGA clock rate is 48 MHz).
    For example, for a cycle which updates at 1 kHz, `cycles=48000'. All of the
    channels share the same cycle, and are synchronized (that's the point!).

    The output of *each* digital channel works as follows:
        - Reset at the beginning of the cycle, and wait for `delay' ticks.  If
            `delay=0' go high immediately.
        - Go high for `hi' ticks.
        - Go low for `lo' ticks.
        - Repeat, producing a total of `pulses' ticks.  If `pulses=0', produce as
            many pulses as possible.
        - Stay low until the next cycle begin.  Note: if the specified number of
            pulses are not finished, it doesn't matter: when a new cycle is
            started everything is reset!

    Each channel also has a couple of useful options:
        - Invert: if inverted, "high" becomes 0 V, and "low" is 3.3 V
        - Gate: if True, the signal is always low unless the gate is active.
    '''

    USB_VID = 0x1209
    USB_PID = 0x6131

    ticks_per_s = 48000000
    analog_max = 4

    def __init__(self, device=0, reset=True):
        devs = find_usb_ports(self.USB_VID, self.USB_PID)

        if not devs or device >= len(devs):
            if reset and reset_fpga():
                    print('Found FPGA in bootloader mode; resetting and trying again.')
                    time.sleep(1)
                    devs = find_usb_ports(self.USB_VID, self.USB_PID)

        if device >= len(devs):
            raise ValueError('Requested index (%d) is more than available devices (found %d)' % (device, len(devs)))

        dev = devs[device]

        if USE_SERIAL:
            try:
                self._serial_port = serial.Serial(dev.device)
            except:
                print('Serial port opening error; waiting 5 seconds and trying again...')
                time.sleep(5)
                self._serial_port = serial.Serial(dev.device)

            self._read = self._serial_port.read
            self._write = self._serial_port.write
        else:
            port = dev.configurations()[0]
            self._out, self._in = port.interfaces()[1].endpoints()
            self._read = self._in.read
            self._write = self._out.write


    def _cmd(self, cmd1, cmd2, data):
        # print(cmd1, cmd2, data) #debug
        self._write(struct.pack(">cBI", cmd1, cmd2, data))
        return self._read_packet()

    def _read_packet(self):
        rdat = self._read(6)
        if rdat[0] & 32:
            raise SynchronizerError("Synchronizer returned error: '%s'" % rdat)
        return rdat


    def close(self):
        '''Manually close the communications port; normally not needed.'''
        if hasattr(self, '_serial_port'):
            self._serial_port.close()


    def pulse_setup(self, channel, program=0, pulses=0, delay=0, hi=1, lo=1):
        '''Set up a pulse program.

        Parameters
        ----------
        channel : int (0-16)
            The channel number to write to.

        Keywords
        --------
        program : int (0-16)
            The program number to write to.
        delay : int (0-4,294,967,295)
            The number of ticks to delay the first pulse by.
        hi : int (0-4,294,967,295)
            The number of ticks each pulse is high for.
        lo : int (0-4,294,967,295)
            The number of ticks each pulse is low for.
        pulses : int (0-4,294,967,295)
            The number of pulses per cycle

        '''

        self._cmd(b'D', 0b00000001, program)
        self._cmd(b'D', 0b11000000 + (channel<<2), pulses)
        self._cmd(b'D', 0b11000001 + (channel<<2), delay)
        self._cmd(b'D', 0b11000010 + (channel<<2), hi)
        self._cmd(b'D', 0b11000011 + (channel<<2), lo)


    def select_program(self, program):
        self._cmd(b'D', 0b00000000, program)

    def digital_setup(self, channel, active=True, inverted=False, gated=False):
        '''Set up a digital output channel.

        Parameters
        ----------
        channel : int (0-16)
            The channel number
        active : bool (default: True)
            Is this channel active?
        inverted : bool (default: False)
            If True, output is inverted ("high" -> 0 V, "low" -> 5 V)
        gated: bool (default: False)
            If True, output is low unless the trigger is active
        '''
        self._cmd(b'D', 0b01000000 + (channel<<2), (1 if active else 0) + (2 if inverted else 0) + (4 if gated else 0))


    def cycle_setup(self, ticks=0, write=True, execute=False):
        '''Change the number of ticks per cycle.  Change delayed until an
        "update" command is called.

        Keywords
        --------
        ticks : int (0-4,294,967,295)

        '''
        self._cmd(b'C', 1, ticks)


    def update(self):
        self._write(b'UPDATE')
        self._read_packet()


    def trigger(self, cycles, external=False):
        '''Trigger the gated channels.

        Keywords
        --------
        cycles : int (0-4,294,967,295)
            The number of cycles the gate should be active for.
        external : bool (default: False)
            If True, sets up trigger for external input via the trigger pin.
            Otherwise begins the trigger immediately.
        '''

        self._cmd(b'T', 0 if external else 1, cycles)


    def start(self):
        self._write(b'RUN!  ')
        self._read_packet()


    def stop(self, immediate=False):
        if immediate:
            self._write(b'STOP! ')
        else:
            self._write(b'STOP. ')
        self._read_packet()


    def running(self):
        self._write(b'RUN?  ')
        reply = self._read_packet()
        if reply == b"R>YES ":
            return True
        elif reply == b"R>NO  ":
            return False
        else:
            raise SynchronizerError("Synchronizer returned unexpected respose: '%s'" % reply)


    def analog_write(self, addr, data):
        '''Write analog data.

        parameters
        ----------
            addr : int (0-4095)
                The first address to write
            data : iterable of ints or floats
                The input data.  Should either be integers < 16 bits,
                or floats (-analog_max - analog_max).  The former will be directly written to the
                output, while the latter directly specifies output voltage.
        '''

        if (addr < 0) or (addr > 4095) or (not isinstance(addr, int)):
            raise SynchronizerError("Analog write address must and interger in the range (0, 4095)")
        if (len(data) + addr) > 4096:
            raise SynchronizerError("Tried to write to much data\n(final address must be < 4096)")

        for i, dat in enumerate(data):
            if isinstance(dat, float):
                # dat = int(65535 * (self.analog_max - dat) / (2 * self.analog_max) + 0.5)
                dat = int(65535 * (dat + self.analog_max) / (2 * self.analog_max) + 0.5)
                # print(dat)

            if not isinstance(dat, int) or (dat < 0) or (dat > 65535):
                raise SynchronizerError("Analog write values must be ints in the range (0, 65535)\n(if values specified as float, must have magnitude < %s V)" % self.analog_max)

            self._cmd(b'A', 3, ((addr+i)<<16) + dat)


    def analog_setup(self, channel1=True, channel2=False, addr=0, ticks=48):
        '''Change analog setup.

        keywords
        --------
        channel1, channel2 : bool (default: True, False)
            Determines if each output channel is active.
        addr : int (0-4095, default: 0)
            The address to start analog reading at.
        ticks : int (default: 48)
            The number of clock ticks per analog update.

        Note: If two channels are selected, it will take two updates to change
        the outputs.  Additionally, channel 1 is always update from even
        addresses and channel 2 from odd addresses.  Finally, the output values
        are always update on channel 2 write.

        (If one of the two channels are selected, updates happen every "ticks"
        cycles)
        '''

        self._cmd(b'A', 0, ((1<<16) if channel1 else 0) + ((1<<17) if channel2 else 0) + addr)
        self._cmd(b'A', 1, ticks)
