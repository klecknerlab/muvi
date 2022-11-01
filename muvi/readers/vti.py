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
import numba
from .. import VolumetricMovie, VolumeProperties
from xml.etree import ElementTree
import re
import warnings
import struct


# A simple homegrown LZ4 decompresser, which uses JIT through numba.
# It's really fast!  Typically performance is several times that of the LZ4
#   library!  A few GB/s of decompression speed is easily acheivable on
#   commodity hardware, primarily due to excuting in parallel.
@numba.jit(nopython=True, parallel=True, cache=True)
def numba_decompress_blocks(input, block_size, last_block_size, block_ends, output):
    num_blocks = len(block_ends)

    for p in numba.prange(num_blocks):
        if p == 0:
            i = numba.uint64(0)
        else:
            i = numba.uint64(block_ends[p - numba.uint(1)])

        block_end = numba.uint64(block_ends[p])
        j = numba.uint64(block_size * p)

        if (p == (num_blocks - numba.uint8(1))):
            end = j + numba.uint64(last_block_size)
        else:
            end = j + numba.uint64(block_size)

        while ((j < end) and (i < block_end)):
            t1 = numba.uint16((input[i] & 0xF0) >> 4)
            t2 = numba.uint16((input[i] & 0x0F) + 4)
            i += numba.uint8(1)

            if (t1 == 15):
                while input[i] == 255:
                    t1 += numba.uint8(input[i])
                    i += numba.uint8(1)

                t1 += numba.uint8(input[i])
                i += numba.uint8(1)

            for n in range(t1):
                output[j] = input[i]
                i += numba.uint8(1)
                j += numba.uint8(1)

            if (j >= end): break

            off = numba.uint16(input[i]) + (numba.uint16(input[i+1]) << 8)
            i += numba.uint8(2)

            if (t2 == 19):
                while input[i] == 255:
                    t2 += numba.uint8(input[i])
                    i += numba.uint8(1)

                t2 += numba.uint8(input[i])
                i += numba.uint8(1)

            for n in range(t2):
                output[j] = output[j - off]
                j += numba.uint8(1)


class VTIMovie(VolumetricMovie):
    def __init__(self, fn, **kwargs):
        '''Create a VolumetricMovie from a VTI file.

        Parameters
        ----------
        fn : str
            Filename

        Any additional keywords are passed to the info directory, overriding
        any in the original file.  (Can be used, for example, to modify the
        perspective correction or other parameters.)

        Presently this library has only been tested on VTI files generated by
        the library itself.  In principle, outside files are supported, but
        may require debugging.  (The library is also somewhat picky about
        which types of encodings, compression, etc. are supported.  This could
        be upgraded if there is a use case for it.)
        '''

        self.filename = fn
        self.info = VolumeProperties()

        offsets = []

        self.f = open(fn, 'rb')

        found_main_tag = False

        for event, elem in ElementTree.iterparse(self.f, events=['start', 'end']):
            # if event == 'start':
            #     print(elem.tag)

            # Check to make sure the main tag is what we expect, and
            #    extract some properties from it.
            if not found_main_tag:
                if elem.tag != 'VTKFile':
                    raise ValueError("expected first tag in '%s' to be 'VTKFile', instead found '%s'" % (self.filename, elem.tag))
                else:
                    found_main_tag = True

                    filetype = elem.get('type', None)
                    if filetype.lower() != 'imagedata':
                        raise ValueError("expected VTKFile tag in '%s' to have type 'ImageData', found '%s' instead" % (self.filename, filetype))

                    # If byte_order not specified, assume little endian.
                    byte_order = elem.get('byte_order', 'littleendian')
                    if byte_order.lower() == 'littleendian':
                        self.header_byte_order = '<'
                    elif byte_order.lower() == 'bigendian':
                        self.header_byte_order = '>'
                        warnings.warn("VTK file '%s' has big endian headers -- support for this file type is preliminary and untested." % (self.filename))
                    else:
                        raise ValueError("VTK file '%s' has unknown byte_order: '%s'" % (self.filename, byte_order))

                    # If header_type not specified, assume UInt32
                    header_type = elem.get('header_type', 'UInt32')
                    if header_type.lower() == 'uint32':
                        long_header = False
                    elif header_type.lower() == 'uint64':
                        long_header = True
                    else:
                        raise ValueError("VTK file '%s' has unknown header_type: '%s'" % (self.filename, header_type))

                    compression = elem.get('compressor', None)
                    if compression.lower() != 'vtklz4datacompressor':
                        raise ValueError("This reader only supports LZ4 compressed data files.\n   (File '%s' specified '%s' compressor, should be 'vtkLZ4DataCompressor')" % (self.filename, compression))
                        self.compression = 'LZ4'

            # Read until we hit the AppendedData section.  Since this may
            #  contain raw data, it may no longer be valid XML after this point!
            # Also, you don't want to read in GB of data!
            if event == 'start' and elem.tag == 'AppendedData':
                encoding = elem.get('encoding', None)
                if encoding != 'raw':
                    raise ValueError("Only raw encoded VTI files supported at this time.\n  (Found encoding='%s' in AppendedData of '%s')" % (encoding, self.filename))
                break

            # Check for the completion of some important tags!
            if event == 'end':

                # These are tags created by the MUVI metadata writer.  They
                #   may not be present, which is ok too!  (This will be true
                #   if the file was generated by anything other than the MUVI
                #   software, in which case it may still be ok!)
                if elem.tag == 'VolumeProperties':
                    self.info.update(elem)

                # This is the main tag for the image data
                elif elem.tag == 'ImageData':
                    # There should only be one; we can check if we've already
                    #   populated the frame offsets, and if so issue a warning.
                    if len(offsets):
                        warnings.warn("Warning: found multiple ImageData objects; ignoring all those after the first.")
                        continue

                    else:
                        # Read the extent and spacing info from the ImageData tag
                        try:
                            extents = list(map(int, elem.get('WholeExtent').split()))
                            # print(type(extents[1] - extents[0]))
                            Nx = extents[1] - extents[0]
                            Ny = extents[3] - extents[2]
                            Nz = extents[5] - extents[4]
                            for i, key in enumerate(('Nx', 'Ny', 'Nz')):
                                N = extents[i*2+1] - extents[i*2]
                                if key in self.info:
                                    if self.info[key] != N:
                                        raise ValueError('Size of array in VolumeProperties does not match data - invalid file.')
                                else:
                                    self.info[key] = N

                        except:
                            raise ValueError("Invalid or missing 'WholeExtent' field in ImageData of VTK File '%s'" % (self.filename))

                        if len(extents) != 6:
                            raise ValueError("'WholeExtent' field in ImageData of VTK File '%s' has wrong number of elements" % (self.filename))

                        # We don't use spacing, so why bother checking it?
                        # if 'spacing' in elem.keys():
                        #     try:
                        #         spacing = np.map(float, elem.get('Spacing').split())
                        #     except:
                        #         raise ValueError("Invalid 'Spacing' field in ImageData of VTK File '%s'" % (self.filename))
                        #
                        #     if len(spacing) != 3:
                        #         raise ValueError("'Spacing' field in ImageData of VTK File '%s' has wrong number of elements" % (self.filename))
                        # else:
                        #     spacing = np.ones(3)

                        # Find the Piece tags inside the ImageData
                        pieces = [e for e in elem if e.tag == 'Piece']

                        # The reader only (currently) supports a single Piece
                        #   tag -- i.e. the data should be in one chunk
                        if len(pieces) != 1:
                            raise ValueError("'ImageData' tag in VTK file '%s' should contain exactly 1 piece (found %d).\n(Multipiece files not supported at this time.)" % (self.filename, len(pieces)))

                        pieces = [e for e in pieces[0] if e.tag == 'CellData']
                        if len(pieces) != 1:
                            raise ValueError("'Piece' tag in VTK file '%s' should contain exactly 1 'CellData' (found %d).\n(Multiimage not supported at this time.)" % (self.filename, len(pieces)))

                        # Get the actual data tags inside the CellData
                        for data in pieces[0]:
                            if data.tag == 'DataArray':
                                if data.get('format').lower() != "appended":
                                    raise ValueError('VTK Reader only supports VTI files with appended data.')
                                if data.get('TimeStep') != str(len(offsets)):
                                    warnings.warn('Frames appear to be out of order in file; may display incorrectly.')
                                channels = int(data.get('NumberOfComponents', 1))
                                if 'channels' in self.info:
                                    if channels != self.info['channels']:
                                        raise ValueError('Number of channels in VolumeProperties does not match data - invalid file.')
                                elif channels > 1:
                                    self.info['channels'] = channels

                                offsets.append(int(data.get('offset')))

        end = self.f.tell()

        if long_header:
            self.header_numpy_type = np.dtype(self.header_byte_order + "u8")
            self.header_struct_type = 'Q'
        else:
            self.header_numpy_type = np.dtype(self.header_byte_order + "u4")
            self.header_struct_type = 'I'

        self.header_size = self.header_numpy_type.itemsize

        self.f.seek(0)
        header = self.f.read(end + 256)
        m = re.search(b'<\s*AppendedData.*?>.*?_', header, flags=re.DOTALL)
        if not m:
            raise ValueError('Failed to find AppendedData section in "%s".  Is this a valid file?' % self.filename)
        self.binary_start = m.end()
        self.frame_offsets = [offset + self.binary_start for offset in offsets]

        # Find the frame shape -- this should never change!
        if 'channels' in self.info:
            self.shape = (self.info['Nz'], self.info['Ny'], self.info['Nx'], self.info['channels'])
        else:
            self.shape = (self.info['Nz'], self.info['Ny'], self.info['Nx'])

        # Update info fields if the user directly specified anything
        self.info.update(kwargs)
        self.validate_info()

    def get_volume(self, i):
        # Read header for this data block
        self.f.seek(self.frame_offsets[i])
        num_blocks, block_size, last_block_size = struct.unpack(self.header_byte_order + '3' + self.header_struct_type, self.f.read(3 * self.header_size))

        # Find the compressed block sizes and ends
        block_sizes = np.frombuffer(self.f.read(num_blocks * self.header_size), dtype=self.header_numpy_type)
        block_ends = np.cumsum(block_sizes)

        # Decompress output into numpy array
        output = np.empty((num_blocks - 1) * block_size + last_block_size, dtype='u1')
        numba_decompress_blocks(np.frombuffer(self.f.read(block_ends[-1]), dtype='u1'), block_size, last_block_size, block_ends, output)

        # Reshape, retype, and return.
        return output.view(self.info['dtype']).reshape(self.shape)


    def close(self):
        self.f.close()
        del self.f
