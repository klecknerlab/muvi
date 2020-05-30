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

import numpy as np
import struct
import os
from threading import Lock

uint8_t = 'B'
uint16_t = 'H'
uint32_t = 'I'
uint64_t = 'Q'
int16_t = 'h'
int32_t = 'l'
bool32_t = 'i'
float_t = 'f'
double_t = 'd'

class Seq:
    _HEADER_FIELDS = (
        ('magic', uint32_t),
        ('name', '24s'),
        ('version', int32_t),
        ('header_size', int32_t),
        ('description', '512s'),
        ('width', uint32_t),
        ('height', uint32_t),
        ('bit_depth', uint32_t),
        ('bit_depth_real', uint32_t),
        ('image_size_bytes', uint32_t),
        ('image_format', uint32_t),
        ('allocated_frames', uint32_t),
        ('origin', uint32_t),
        ('true_image_size', uint32_t),
        ('suggested_frame_rate', double_t),
        ('description_format', int32_t),
        ('reference_frame', uint32_t),
        ('fixed_size', uint32_t),
        ('flags', uint32_t),
        ('bayer_pattern', int32_t),
        ('time_offset_us', int32_t),
        ('extended_header_size', int32_t),
        ('compression_format', uint32_t),
        ('reference_time_s', int32_t),
        ('reference_time_ms', uint16_t),
        ('reference_time_us', uint16_t),
        # More header values not implemented
    )

    def __init__(self, filename, output_bits=8, black_level=64,
                 white_level=4064, gamma=1.0, dark_clip=0.0):
        '''Open a Norpix SEQ file.

        This reader does not return the raw values in the file, but rather
        converts it to normalized values in a user-defined way.

        Parameters
        ----------
        filename : str

        Keywords
        --------
        output_bits : int (default: 8)
            The number of bits in the output array.  Should be 8 or 16
        black_level : int (default: 64)
            The level in the decoded cine file which is considered "black";
            this value will be converted to 0 in the output.  This value is
            standard for Phantom cameras, and should not need to be changed.
        white_level : int (default: 4064)
            The level in the decoded cine file which is considered "white";
            this value will be converted to 2^bits - 1 in the output.  This
            value is standard for Phantom cameras, and should not need to be
            changed.
        gamma : float (default: 1.0)
            The gamma value of the extracted images.  The actual intensity of
            the pixel is proportional to (output value)^(gamma)
        dark_clip : float (default: 0.0)
            Relative brightnesses below this value are converted to 0 in the
            output.  In other words, the clip value in the raw file is given
            by (black_level * (1 - dark_clip) + white_level * dark_clip).
            Note that this clip is applied *before* gamma correction!
        '''
        self._file = open(filename, 'rb')
        self.filename = filename

        # Read the headers
        self.header = self._read_header(self._HEADER_FIELDS)

        # Do some basic checks
        if self.header['magic'] != 0xFEED:
            raise IOError('magic number incorrect; is this an SEQ file?')
        if self.header['compression_format'] != 0:
            raise IOError('only uncompressed images are supported in .seq files')
        if self.header['image_format'] != 100:
            raise IOError('only monochrome images are supported')

        # For convenience
        self.width = self.header['width']
        self.height = self.header['height']
        self.internal_bit_depth = self.header['bit_depth_real']

        if self.internal_bit_depth == 8:
            self.dtype = 'u1'
        elif self.internal_bit_depth == 16:
            self.dtype = 'u2'
        else:
            raise ValueError('bit depth should be 8 or 16 (found %s)' % self.internal_bit_depth)

        self.pixel_count = self.width * self.height
        self.image_bytes = self.pixel_count  * np.dtype(self.dtype).itemsize

        # An image block also includes timestamp metadata, so it's a bit bigger
        #   than the raw image data.  We can get the total size from the
        #   header.
        self.image_block_size = self.header['true_image_size']


        # Some things differe depending on the software version
        if self.header['version'] >= 5:  # StreamPix version 6
            self.image_offset = 8192
            self.timestamp_micro = True
        else:
            self.image_offset = 1024
            self.timestamp_micro = False

        # We need to compute this manually... sort of annoying!
        self.image_count = (os.stat(self.filename).st_size -
                            self.image_offset) // self.image_block_size

        # Allows Seq object to be accessed from multiple threads
        self.file_lock = Lock()

    def _unpack(self, format):
        s = struct.Struct('<' + format)
        vals = s.unpack(self._file.read(s.size))
        return vals[0] if len(vals) == 1 else vals

    def _read_header(self, fields, offset=None):
        return {name: self._unpack(format) for name, format in fields}

    def __getitem__(self, key):
        if type(key) == slice:
            return [self.get_frame(i) for i in range(len(self))[key] ]
        else:
            if key < 0:
                key = len(self) + key
            return self.get_frame(key)

    def __len__(self):
        return self.image_count

    def close(self):
        if hasattr(self, 'f'):
            self._file.close()
        del self._file

    def __del__(self):
        self._file.close()

    def get_frame(self, i):
        self.file_lock.acquire()

        self._file.seek(self.image_offset + self.image_block_size * i)
        frame = np.fromfile(self._file, self.dtype, self.pixel_count)

        self.file_lock.release()

        return frame.reshape(self.height, self.width)

    def get_time(self, i):
        self._file.seek(self.image_offset + self.image_block_size * i + self.image_bytes)

        ts, tms = self._unpack('<LH', self._file.read(6))
        if self.timestamp_micro:
            tus = self._unpack('<H', self._file.read(2))
        else:
            tus = 0

        return ts + tms * 1E-3 + tus * 1E-6


if __name__ == "__main__":
    import sys

    seq = Seq(sys.argv[1])


    for header, name in (('header', 'Header'), ):
        print('-'*79)
        print(name)
        print('-'*79)

        for k, v in getattr(seq, header).items():
            v = str(v)
            if len(v) > 59:
                v = v[:56] + '...'
            print('%18s: %s' % (k, v))

    Nx = 768
    Ox = 1024
    Ny = 256
    Oy = 25
    Nz = 256
    Oz = 20000

    vol = np.empty((Nz, Ny, Nx), dtype='u1')

    for z in range(Nz):
        vol[z] = seq[Oz+z][Oy:Oy+Ny, Ox:Ox+Nx]



    from muvi import VolumetricMovie
    from muvi.view.qtview import view_volume

    vm = VolumetricMovie([vol])

    vm.save('seq_test.vti')
    view_volume(vm)

#     import matplotlib.pyplot as plt
# #
#     frame = seq[len(seq)//2]
# #
#     plt.imshow(frame)
#     plt.colorbar()
# # #
# #     plt.hist(frame[..., :1000].reshape(-1), bins=np.arange(256))
# #     plt.yscale('log')
# #
#     plt.show()
