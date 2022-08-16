#!/usr/bin/python3
#
# Copyright 2022 Dustin Kleckner
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

'''This module contains functions for reading / writing to VTK files with
compressed appeneded data.

It isn't meant to be used directly; rather if should be called through the
interfaces defined by various data types.
'''

from xml.sax.saxutils import quoteattr, escape
from xml.etree import ElementTree
import lz4.block
import concurrent.futures
import numpy as np
import numba
import re
import struct

OFFSET_DIGITS = 20 #Long enough to store 64 bit numbers

# Convert Numpy types to VTK Data types
VTK_DATA_TYPES = {
    'b': "Int8",
    'B': "UInt8",
    'h': "Int16",
    'H': "UInt16",
    'i': "Int32",
    'I': "UInt32",
    'l': "Int64",
    'L': "UInt64",
    'f': "Float32",
    'd': "Float64"
}

class VTKWriter:
    def __init__(self, fn, vtk_type, block_size=2**16, compression=-1):
        self.fn = fn
        self.closed = False
        self.vtk_type = vtk_type
        self.data = {}

        self.block_size = block_size

        if compression > 0:
            self._write_comp = dict(mode='high_compression', compression=compression, store_size=False)
        else:
            self._write_comp = dict(mode='fast', acceleration=1-compression, store_size=False)


    def _check_open(self):
        if hasattr(self, '_f'):
            return True
        if self.closed:
            raise ValueError("Can't write to a closed VTKFile")
        elif not hasattr(self, "_f"):
            self._f = open(self.fn, 'wb')
            self._f.write(rf'''<?xml version="1.0"?>
<VTKFile type="{self.vtk_type}" version="0.1" byte_order="LittleEndian" header_type="UInt64" compressor="vtkLZ4DataCompressor">
'''.encode('UTF-8'))

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, type, value, tb):
        self.close()

    def write_tag(self, tag):
        self._check_open()
        s, data = tag.xml(indent=1)

        if data:
            start = self._f.tell()
            for offset, d in data.items():
                self.data[offset + start] = d
        self._f.write(s.encode('utf-8'))

    def close(self):
        self._f.write(b'  <AppendedData encoding="raw">\n_')
        self._f.flush()
        append_start = self._f.tell()

        offsets = {}
        for offset, data in self.data.items():
            offsets[offset] = self._f.tell() - append_start
            self._write_data(data)

        self._f.write(b'\n  </AppendedData>\n</VTKFile>')
        self._f.flush()

        for o1, o2 in offsets.items():
            self._f.seek(o1)
            self._f.write(f'{o2}"'.encode('utf-8'))

        self._f.close()
        delattr(self, '_f')
        self.closed = True

    def _write_data(self, data):
        if hasattr(data, "tobytes"):
            data = data.tobytes()

        num_blocks = (len(data) + self.block_size - 1) // self.block_size
        mv = memoryview(data)
        blocks = [mv[i*self.block_size:(i+1)*self.block_size] for i in range(num_blocks)]
        # blocks = [data[i*self.block_size:(i+1)*self.block_size] for i in range(num_blocks)]

        with concurrent.futures.ThreadPoolExecutor() as exec:
            dat_comp = list(exec.map(lambda block: lz4.block.compress(block, **self._write_comp), blocks))

        header = struct.pack('<QQQ', num_blocks, self.block_size, len(blocks[-1]))
        header += struct.pack('<%dQ' % num_blocks, *map(len, dat_comp))

        self._f.write(header)
        for chunk in dat_comp:
            self._f.write(chunk)

        self._f.flush()


class DataProxy:
    def __init__(self, dtype, data, *args, **kwargs):
        '''Object to hold a function to acquire data for writing to a VTKFile.

        Used to allow the creation of VTKTag's without loading the data, which
        could be too big to fit in memory.

        Recreates some attributes of numpy arrays to allow transparent handling.

        Parameters
        ----------
        dtype : a numpy data type string
        data : function or numpy array
            The function to call to get the data (should return a numpy array
            or raw bytes)

        Any additional arguments or keywords are passed to the data function
        when it is called
        '''
        self.dtype = np.dtype(dtype)
        self.data = data
        self.args = args
        self.kwargs = kwargs

    def tobytes(self):
        data = self.data

        if hasattr(data, '__call__'):
            data = data(*self.args, **self.kwargs)

        if hasattr(data, 'tobytes'):
            data = data.tobytes()

        return data


class VTKTag:
    def __init__(self, tag, contents=None, **attr):
        '''Class to hold a VTK Tag

        parameters
        ----------
        tag : string
            The tag name
        contents : string, iterable, or None
            The contents of this tag.  If an iterable, assumed to be contain
            strings or more VTKTag objects

        All keyword attributes get converted to tag attributes using the `str`
        command -- if special formatting is required do this before passing
        them as keywords.
        '''

        self.tag = tag

        self.contents = contents
        if isinstance(contents, (np.ndarray, DataProxy)):
            attr['type'] = VTK_DATA_TYPES[contents.dtype.char]
            attr['format'] = 'appended'
            self.appended = True
        else:
            if isinstance(self.contents, VTKTag):
                self.contents = [self.contents]
            self.appended = False

        self.attr = ' '.join( f'{k}={quoteattr(str(v))}' for k, v in attr.items())

    def xml(self, indent=1):
        s = f"{'  '*indent}<{self.tag} {self.attr}"
        data = {}

        if self.appended: # This tag contains appended binary data
            s += ' offset="'
            data[len(s)] = self.contents
            s += '0"' + ' '*(OFFSET_DIGITS-1) + "/>\n"
        elif self.contents is None: # This tag contains nothing
            s += "/>\n"
        elif isinstance(self.contents, str): # This tag contains a string
            s += ">" + escape(self.contents) + f"</{self.tag}>\n"
        elif hasattr(self.contents, "__iter__"): # This tag contains more tags
            s += ">\n"
            for child in self.contents:
                if isinstance(child, VTKTag):
                    cs, cd = child.xml(indent+1)

                    # Copy child data, updating offsets
                    for offset, d in cd.items():
                        data[len(s) + offset] = d

                    s += cs

                elif isinstance(child, str):
                    s += child + "\n"
                else:
                    raise TypeError(f'can not put object of type {child.type} in VTK contents')
            s += f"{'  '*indent}</{self.tag}>\n"

        else:
            raise TypeError('contents of VTKTag should be data, string, or iterable')

        return s, data

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


class VTKReader:
    def __init__(self, fn):
        '''Open and read a VTK file.

        Not intended to be used directly by the end user, but used internally
        by several user facing classes.

        Arguments
        ---------
        fn : string
            File name
        '''

        self.filename = fn
        self.filetype = None

        self.f = open(fn, 'rb')
        self.root = None
        # Read in the tree, but stop once we get to appended data!
        for event, elem in ElementTree.iterparse(self.f, events=['start', 'end']):
            if self.root is None:
                self.root = elem
            if event == 'start' and elem.tag == 'AppendedData':
                self.has_data = True
                break

        if (not self.root) or self.root.tag != 'VTKFile':
            raise ValueError(f"File '{self.filename}' is not a valid VTK file")

        # We should have an AppendedData section -- find the file offset to
        #   the start of the compressed binary data
        if self.has_data:
            end = self.f.tell()
            self.f.seek(0)
            header = self.f.read(end + 256)
            m = re.search(b'<\s*AppendedData.*?>.*?_', header, flags=re.DOTALL)
            if not m:
                self.has_data = False
            else:
                self.data_start = m.end()

        # Read in the type from the root tag
        if 'type' not in self.root.attrib:
            raise ValueError(f"VTK file '{self.filename}' does not have a defined 'type' in the root tag -- this is not a valid VTK document")
        self.vtk_type = self.root.attrib['type']

        # Get header byte order, and make sure it's valid
        self.header_byte_order = "<" if self.root.attrib.get('') else ">"
        hbo = self.root.attrib.get('header_byte_order', "LittleEndian").lower()
        if hbo == 'littleendian':
            self.header_byte_order = "<"
        elif hbo == 'bigendian':
            slef.header_byte_order = ">"
        else:
            raise ValueError(f'Invalid header byte order: "{hbo}"')

        # Get header data type and make sure it's valid
        self.header_numpy_type = np.dtype(self.root.attrib.get('header_type', 'UInt32').lower())
        if self.header_numpy_type == 'u8':
            self.header_struct_type = 'Q'
        elif header_type == 'u4':
            self.header_struct_type = 'L'
        else:
            raise ValueError(f'Header data type is "{self.header_numpy_type.str}", should be uint32 or uint64')
        self.header_size = self.header_numpy_type.itemsize

        # Check that the compression is what we expect
        # In the future, it would be nice to support uncompressed files!
        compression = self.root.attrib.get('compressor', None)
        if compression.lower() != 'vtklz4datacompressor':
            raise ValueError("This reader only supports LZ4 compressed data files.\n   (File '%s' specified '%s' compressor, should be 'vtkLZ4DataCompressor')" % (self.filename, compression))
            self.compression = 'LZ4'

        self.main = self.root.find(self.vtk_type)
        if not self.main:
            raise ValueError(f"File 'self.filename' missing main tag (<'{self.vtk_type}'>)")

        pieces = self.main.findall('Piece')
        if len(pieces) != 1:
            raise ValueError(f"Expected VTK file with 1 'Piece' tag, found {len(pieces)}")
        self.contents = pieces[0]


    def get_data_from_tag(self, tag):
        '''Read a data array from a DataArray tag'''
        offset = int(tag.attrib['offset'])
        dtype = np.dtype(tag.attrib['type'].lower())

        data = self.get_raw_data(offset, dtype)

        # If it's a vector, make it so!
        if 'NumberOfComponents' in tag.attrib:
            data = data.reshape(-1, int(tag.attrib['NumberOfComponents']))

        return data


    def get_raw_data(self, offset, dtype):
        # Read header for this data block
        self.f.seek(self.data_start + offset)
        num_blocks, block_size, last_block_size = struct.unpack(self.header_byte_order + '3' + self.header_struct_type, self.f.read(3 * self.header_size))

        # Find the compressed block sizes and ends
        block_sizes = np.frombuffer(self.f.read(num_blocks * self.header_size), dtype=self.header_numpy_type)
        block_ends = np.cumsum(block_sizes)

        # Decompress output into numpy array
        output = np.empty((num_blocks - 1) * block_size + last_block_size, dtype='u1')
        numba_decompress_blocks(np.frombuffer(self.f.read(block_ends[-1]), dtype='u1'), block_size, last_block_size, block_ends, output)

        # Reshape, retype, and return.
        return output.view(dtype)


    def close(self):
        self.f.close()
        del self.f
