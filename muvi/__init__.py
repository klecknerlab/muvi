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
# import json
import struct
import warnings
import os
import sys
# import blosc
# blosc.set_nthreads(1)
import lz4.block
# import base64
# import struct
import concurrent.futures
# from muvi import status_range
from xml.etree import ElementTree as ET
# import re
# import numba


class VolumetricMovieError(Exception):
    pass


def xml_indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            xml_indent(elem, level+1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


class VolumeProperties:
    _defaults = {
        # 'dtype': 'u1',
        # 'channels': 1,
        # 'interleaved': True,
        # 'offset': 0,
        # 'clip': 80,
        # 'gamma2': True,
    }

    _alternates = {
        'Lx': 'Nx',
        'Ly': 'Ny',
        'Lz': 'Nz',
        'Ns': 'Nz',
    }

    _param_types = {
        'Nx': int,
        'Ny': int,
        'Nz': int,
        'Ns': int,
        'offset': int,
        'Nt': int,
        'channels': int,
        'interleaved': bool,
        'Lx': float,
        'Ly': float,
        'Lz': float,
        'dx': float,
        'dy': float,
        'dz': float,
        't0': float,
        'dt': float,
        'dtype': str,
        'creation_time': float,
        'units': str,
        'clip': int,
        'top': int,
        'gamma2': bool,
    }

    __properties_TYPES = [str, int, float, bool]
    __properties_TYPES_STR = dict((t.__name__, t) for t in __properties_TYPES)

    def __init__(self, *args, **kwargs):
        '''
        Generic class for handling metadata associated with volumetric movies.

        In general, behaves like a dictionary which only takes basic data types
        (by default: string, int, float, bool).  Designed to export/import
        from XML.

        Initiated with positional and/or keyword arguments.  Positional
        arguments can be strings (interpreted as XML input), ElementTrees,
        dictionaries, or other VolumeProperties objects.  (See update method for
        details)

        Also contains default and alternate values specific to Volumetric
        Images.

        Known Keywords
        --------------
        Nx, Ny, Nz : int
            Number of pixels on each axis
        Nt : int
            Number of time steps
        Ns : int
            The number of 2D frames per complete 3D scan period.  If the scan is
            not interleaved, this is the time to scan a single color channel.
            Generally, should be more than Nt, since there are (almost always)
            dead frames in between scans.
        dt : float
            The time step (1 / volume rate).
        channels : int
            Number of color channels.  If specified, this will be the size
            of the last axis of a single volume.  If it is not specified, the
            returned volumes will be 3D.
        interleaved : bool (default: True)
            Indicates that different color channels are stacked in alternating
            2D frames (e.g. RGRGRGRGRG--RGRG...).  If False, different color
            channels are in separate scans (e.g. RRRRR--GGGGG--RR...).
        offset : int
            The number of 2D frames to ignore at the beginning of the movie.
            (default: 0; only for 2D movies).
        Lx, Ly, Lz : float
            Physical size of the volume on each axis.  For distortion corrected
            volumes, this is the size of the volume at the Axis center.
        dx, dy, dz : float
            Used for perspective correction; effective distance along each axis
            to the pivot point.  `dz` should be the effective distance from the
            volume center to the camera nodal point, and should positive.
            Only one of `dy` or `dz` should be specified, depending on where
            the camera is located.  Sign depends on the location of the scanner;
            for example if the scanner is in the -y direction, you would
            specify a negative `dy`.
        units : str
            The physical units for all lengths.
        creation_time : float
            The creation time of the volume, as a Unix time float.
        clip : int (default: 80)
            When converting 2D movies, the value below which to clip and count
            as zero.
        top : int
            When converting 2D movies, the maximum value used to rescale data.
        gamma2 : bool (default: True)
            If true, set a gamma=2.0; in other words the stored value is the
            square root of the actual intensity.  This method of storage
            optimizes noise performance.
        '''
        self._d = {}

        for arg in args:
            self.update(arg, raise_errors=True)

        for k, v in kwargs.items():
            self[k] = v


    def update(self, input, warn=False, raise_errors=False):
        '''
        Update internal data with external source.

        Parameters
        ----------
        input: string, xml.etree.ElementTree.Element, or dictionary-like

        Keywords
        --------
        warn: bool (default: False)
            If True, warns when it finds items in the input it can't parse.
        raise_errors: bool (default: False)
            If True, raises errors when it finds items it can't parse.

        If input is another VolumeProperties or dictionary, elements are copied over.
        If input is a string, it is assumed to be XML which is parsed and
        then iterated over.  The default behavior is to ignore items it can't
        parse, which can be changed (see above).

        If input is an XML Element, it is assumed that it contains sub-elements
        which are the items to be updated from.

        If either a string or XML Element is specified, it is assumed all
        data objects are embedded in an out tag (the type of which is ignored):
        ```
        <VolumeProperties>
            <float name='Lx'>1.5</float>
            <int name='Nx'>20</int>
            ...
        </VolumeProperties>
        ```
        '''
        if isinstance(input, (VolumeProperties, dict)):
            for k, v in input.items():
              self[k] = v
            return

        if isinstance(input, (str, bytes)):
            input = ET.fromstring(input)

        if isinstance(input, ET.Element):
            for item in input:
                t = self.__properties_TYPES_STR.get(item.tag, None)
                if t is None:
                    if warn:
                        print("Warning: could not interperet input tag '%s'" % item.tag)
                    if raise_errors:
                        raise ValueError("Could not interperet input tag '%s'" % item.tag)
                    continue

                if 'name' not in item.keys():
                    if warn:
                        print("Warning: input tag '%s' has no name" % item.tag)
                    if raise_errors:
                        raise ValueError("Input tag '%s' has no name" % item.tag)
                    continue

                self._d[item.get('name')] = t(item.text)

        else:
            raise TypeError('Could not interperet input: %s' % repr(input))



    def __setitem__(self, key, val):
        if type(val) not in self.__properties_TYPES:
            raise ValueError('Values for VolumeProperties should be one of: [%s]' % (', '.join(self.__properties_TYPES_STR.keys())))
        elif key not in self._param_types:
            if not key.startswith('user_'):
                raise ValueError("Invalid VolumeProperty '%s'" % key)
        else:
            val = self._param_types[key](val)

        self._d[key] = val


    def __getitem__(self, key):
        if key in self._d:
            return self._d[key]
        elif key in self._defaults:
            return self._defaults[key]
        elif key in self._alternates:
            return self[self._alternates[key]]
        else:
            raise KeyError(key)


    def get(self, key, default=None):
        try:
            return self['key']
        except KeyError:
            return default


    def items(self):
        return self._d.items()


    def xml(self, tag='VolumeProperties', encoding='UTF-8', indent=None):
        '''Encode the properties into an XML bytestring

        Arguments
        ---------
        tag: str (default: "VolumeProperties")
            The tag to use to encapsulate the whole properties object.
        encoding: str (default: "UTF-8")
        indent: int or None (default: None)
            The indentation level.  If None, the XML is compacted onto a single
            line.  For an indent of 0, the outer element is unindented, but the
            inner parts are indented with 2 spaces.
            (indent = 1 adds an extra two spaces to everything, and so on.)
        '''

        base = ET.Element(tag)

        for k, v in sorted(self._d.items()):
            item = ET.SubElement(base, type(v).__name__)
            item.set('name', k)
            item.text = str(v)

        if indent is not None:
            if not isinstance(indent, int):
                ident = 0
            xml_indent(base, indent)
            pre_indent = b'  ' * indent
        else:
            pre_indent = b''

        return (pre_indent + ET.tostring(base, encoding=encoding)).rstrip(b' ')


    def __str__(self):
        return "%s(%s) " % (self.__class__.__name__, ', '.join(
            "%s=%s" % (k, repr(v)) for k, v in sorted(self._d.items())
        ))


    def __contains__(self, key):
        # Return true only if key is defined; alternates or default values will
        #   not return True, even though they can be accessed!  (This is
        #   intentional!)
        return (key in self._d)


    def to_file(self, filename):
        '''Create a new XML file, and write the VolumeProperties to it.

        Note that if you want to write the XML to an already open file you can
        use the `xml` method instead.

        Parameters
        ----------
        filename : str
            Filename
        '''

        with open(filename, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write(self.xml(indent=0))


    def update_from_file(self, filename, tag='VolumeProperties', warn=False, raise_errors=True):
        '''Update object from an XML file.

        Parameters
        ----------
        filename : str
            Filename
        tag : str (default: `"VolumeProperties"`)
            Tag which contains the properties.  The reader will search for any
            instances of this tag and extract information from them.
        '''
        root = ET.parse(filename)
        self._search_tree(root, tag, warn=warn, raise_errors=raise_errors)


    def _search_tree(self, e, tag, warn=False, raise_errors=True):
        if e.tag == tag:
            self.update(e, warn=warn, raise_errors=raise_errors)
        else:
            for ee in e:
                self._search_tree(ee, tag, warn=warn, raise_errors=raise_errors)


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



class status_range:
    def __init__(self, start, stop=None, step=1, pre_message='', post_message='', length=40):
        '''Create a range-like objet which prints out a status message
        automatically.'''
        if stop is None:
            self.start = 0
            self.stop = start
        else:
            self.start = start
            self.stop = stop

        self.step = step
        self.length = length

        self.count = 0
        self.val = 0
        self.max_count = (self.stop - self.start + self.step - 1) // self.step
        self.fmt = "\r" + pre_message +  "[%-" + str(length) + "s] " + post_message + "%s"


    def __iter__(self):
        self.update()
        return self


    def __len__(self):
        return self.max_count


    def get_next(self):
        ret = self.val
        self.val += self.step
        return ret


    def __next__(self):
        if self.count >= self.max_count:
            self.update()
            sys.stdout.write('\n')
            raise StopIteration

        else:
            ret = self.get_next()
            self.count += 1

            self.update()
            return ret


    def update(self):
        '''Update the printout.

        It is not usually called directly, but rather automatically handled
        on each loop.'''

        cnt = '%d/%d' % (self.count, self.max_count)
        sys.stdout.write(self.fmt % ('=' * int(min(self.max_count, self.count)/self.max_count*self.length + 0.5), cnt))
        sys.stdout.flush()


class status_enumerate(status_range):
    '''Similar to `status_range` function, except acts like an `enumerate`
    iterator instead of a `range` iterator.
    '''

    def __init__(self, obj, **kwargs):
        super().__init__(len(obj), **kwargs)
        self.obj = obj


    def get_next(self):
        return (self.count, self.obj[self.count])

#
# _file_ext = {
#     # '.h5':'HDF5',
#     # '.hdf5':'HDF5',
#     '.s4d':'S4D',
#     '.cine':'CINE',
#     '.muv':'MUVI',
#     '.muvi':'MUVI',
# }


# MUVI_READERS = {
#     '.vti': vti.VTIReader,
#     '.cine': cine.Cine,
#     '.s3d': sparse.Sparse3D,
# }
#
# MUVI_WRITERS = {
#     '.vti': vti.VTIWriter,
# }

class VolumetricMovie:
    '''Base class for working with volumetric movies.

    Attributes
    ----------
    source : list/array-like object
        An object which supports indexing and returns a 3D or 4D numpy array.

    Any keywords are passed directly to the `info` attribute, which is a
    VolumeProperties object.

    If you would like to suppport a new data type, it is suggest you inheret
    from this class.  To make a new class which is supported by the viewer,
    you must meet the following requirements:
        1. Define a new `__init__` and `get_volume` method.
        2. The initialization must create an `info` attribute which is
            a `VolumeProperties` object, and populate it as needed.  (The
            `infer_properties` method can take care of many basic attributes.)
    '''

    def __init__(self, source, **kwargs):
        self.source = source
        self.info = VolumeProperties(**kwargs)
        self.info['Nt'] = len(source)
        self.infer_properties()


    def infer_properties(self, overwrite=False):
        '''
        Infer properties of volume from first frame.

        Keywords
        --------
        overwrite : bool (default: False)
            If True, existing values will be overwritten
        '''

        vol = self.get_volume(0)

        shape = vol.shape
        if len(shape) == 3:
            Nz, Ny, Nx = shape
            channels = None
        elif len(shape) == 4:
            Nz, Ny, Nx, channels = shape

        # dtype = vol.dtype.str.lstrip('<>|')
        dtype = vol.dtype.char

        for key, val in [
            ('Nx', Nx),
            ('Ny', Ny),
            ('Nz', Nz),
            ('channels', channels),
            ('dtype', dtype),
        ]:
            if val is not None and key not in self.info:
                self.info[key] = val


    def get_volume(self, index):
        return self.source[index]


    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.get_volume(i) for i in key]
        else:
            return self.get_volume(key)


    def __len__(self):
        return self.info['Nt']


    def save(self, fn, start=0, end=None, compression=0, block_size=2**16, print_status=True):
        '''Write volumetric movie to a VTI file.

        Parameters
        ----------
        fn : str
            Filename to write

        Keywords
        --------
        start : int (default: 0)
            The first frame to write.
        end : int
            The last frame to write; if not specificed, the last frame in the
            movie.
        compression : int (default: 0, ranges from -10 -- 12)
            The level of compression.  If > 0 corresponds to the "fast" mode
            in lz4.block, otherwise uses the "high_compression" option.
        block_size : int (default: 2^16)
            The size of the blocks in the compressed chunk
        print_status : bool (default: True)
            If specified, a running progress bar is printed as it is saved.
        '''

        num_frames = self.start_vti_write(fn, start, end, compression, block_size)

        if print_status:
            sfn = fn
            if len(fn) > 30: sfn = sfn[:27] + '...'
            enum = status_range(0, num_frames, post_message='%s:' % sfn)
        else:
            enum = range(0, num_frames)

        for i in enum:
            self.write_vti_frame()


    def start_vti_write(self, fn, start=0, end=None, compression=0, block_size=2**16):
        '''Start writing a VTI file.  Used internally; see the `save` method
        for a simple user interface.

        Parameters
        ----------
        fn : str
            Filename to write

        Keywords
        --------
        start : int (default: 0)
            The first frame to write.
        end : int
            The last frame to write; if not specificed, the last frame in the
            movie.
        compression : int (default: 0, ranges from -10 -- 12)
            The level of compression.  If > 0 corresponds to the "fast" mode
            in lz4.block, otherwise uses the "high_compression" option.
        block_size : int (default: 2^16)
            The size of the blocks in the compressed chunk

        Returns
        -------
        frames_remaining : int
            The number of frames which need to be written.
        '''

        if hasattr(self, '_write_file'):
            raise ValueError("Tried to start writing a volume that is already being written to another file!")

        if end is None:
            end = len(self)
        elif end < 0:
            end = len(self) + end

        if compression > 0:
            self._write_comp = dict(mode='high_compression', compression=compression, store_size=False)
        else:
            self._write_comp = dict(mode='fast', acceleration=1-compression, store_size=False)

        self._write_header_offsets = []
        self._write_data_offsets = []
        self._write_start = start
        self._write_end = end
        self._write_current = start
        self._write_block_size = block_size

        t = np.arange(end - start) * self.info.get('dt', 1)

        vol_info = dict(
            x0 = 0,
            y0 = 0,
            z0 = 0,
            x1 = self.info['Nx'],
            y1 = self.info['Ny'],
            z1 = self.info['Nz'],
            dx = 1,
            dy = 1,
            dz = 1,
            time_values = ' '.join(map(str, t)),
            num_time_steps = end-start,
            encoding = "raw",
            # encoding = "base64" if b64_encode else "raw",
            header_type = "UInt64",
            # header_type = "UInt64" if long_header else "UInt32",
            arr_dtype = VTK_DATA_TYPES[self.info['dtype']],
        )

        self._write_file = open(fn, 'wb')
        self._write_file.write(r'''<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="{header_type}" compressor="vtkLZ4DataCompressor">
<ImageData WholeExtent="{x0} {x1} {y0} {y1} {z0} {z1}" Origin="{x0} {y0} {z0}" Spacing="{dx} {dy} {dz}" TimeValues="{time_values}">
<Piece Extent="{x0} {x1} {y0} {y1} {z0} {z1}">
<CellData name="ImageScalars">
'''.format(**vol_info).encode('UTF-8'))

        if 'channels' in self.info:
            self._write_data_str = b'        <DataArray type="%s" Name="ImageScalars" format="appended" NumberOfComponents="%d" %%-60s\n' % (vol_info["arr_dtype"].encode("UTF-8"), self.info['channels'])
        else:
            self._write_data_str = b'        <DataArray type="%s" Name="ImageScalars" format="appended" %%-60s\n' % (vol_info["arr_dtype"].encode("UTF-8"))
        self._write_attrib_str = b'TimeStep="%d" offset="%d"/>'

        for n in range(start, end):
            self._write_file.flush()
            self._write_header_offsets.append(self._write_file.tell())
            self._write_file.write(self._write_data_str % (self._write_attrib_str % (n-start, 0)))

        self._write_file.write(r'''      </CellData>
    </Piece>
  </ImageData>
  <AppendedData encoding="{encoding}">
_'''.format(**vol_info).encode('UTF-8'))

        return self._write_end - self._write_current


    def write_vti_frame(self):
        '''Write a single frame to a VTI file.  If all frames are written,
        will also finish up writing the file and close it.

        The file must be openeded with `start_vti_write` method before calling
        this method.

        Returns
        -------
        frames_remaining : int
            The number of frames which still need to be written.  If it returns
            a 0, the writing is done and the file is closed.
        '''

        if not hasattr(self, '_write_file'):
            raise ValueError("No file open for writing; call start_vti_write first.")

        dat = self[self._write_current].tostring()
        self._write_current += 1

        num_blocks = (len(dat) + self._write_block_size - 1) // self._write_block_size
        dat_split = [dat[i*self._write_block_size:(i+1)*self._write_block_size] for i in range(num_blocks)]

        with concurrent.futures.ThreadPoolExecutor() as exec:
            dat_comp = list(exec.map(lambda x: lz4.block.compress(x, **self._write_comp), dat_split))

        # if long_header:
        header = struct.pack('<QQQ', num_blocks, self._write_block_size, len(dat_split[-1]))
        header += struct.pack('<%dQ' % num_blocks, *map(len, dat_comp))
        # else:
        #     header = struct.pack('<III', num_blocks, block_size, len(dat_split[-1]))
        #     header += struct.pack('<%dI' % num_blocks, *map(len, dat_comp))

        # if b64_encode:
            # return base64.b64encode(header) + base64.b64encode(b''.join(dat_comp))
        # else:
        # dat_comp.insert(0, header)
        # return b''.join(dat_comp)

        self._write_data_offsets.append(self._write_file.tell())

        self._write_file.write(header)
        for chunk in dat_comp:
            self._write_file.write(chunk)

        remaining = self._write_end - self._write_current

        if remaining <= 0:
            self._write_file.write(b'  </AppendedData>\n</VTKFile>')

            for n, (hoff, doff) in enumerate(zip(self._write_header_offsets, self._write_data_offsets)):
                doff = doff - self._write_data_offsets[0]
                self._write_file.seek(hoff)
                self._write_file.write(self._write_data_str % (self._write_attrib_str % (n, doff)))

            self._write_file.close()
            del self._write_file

            return 0

        else:
            return remaining



# class VolumetricMovie(object):
#     '''Generic class for working with volumetric movies.
#
#     Attributes
#     ----------
#     volumes : an iterable which contains the volumetric data.  Should support
#         ``len``, be addressable by index, and return a numpy array
#     info : a dictionary of metadata parameters, including perspective distortion
#         and scale information; see documentation of recognized parameters below
#     computed_info: a dictionary of metadata paremeters which are computed from
#         the other parameters.  These parameters are not saved if the volume
#         is written to disk
#     metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
#         object if the volume is written to disk
#     name : a string used to identify the volume; if loaded from a disk this
#         should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
#
#     Members of info, computed_info, and metadata will be accesible as attributes
#     of the class.  To alter these attibutes, however, the underlying
#     dictionaries should be altered, rather than modifying the attributes of the
#     class itself (which will not have the desired effect with regards to
#     saving the volume).
#
#     Each volume should be either 3 or 4 dimensional, where in the later case the
#     fourth axis is the color dimension, typically specifying 1--4 planes.
#
#     Length scales in the volume are specified in terms of an abitrary physical
#     unit, whose scale is specified via "Lunit".
#
#     Time scales are always specified in terms of seconds.
#
#     The valid info parameters are documented below.  Each parameter must be
#     expressable as a floating point number.
#
#     General Info Parameters
#     -----------------------
#     - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
#         physical unit is mm (default), or 25.4E-3 if the unit is inches)
#     - VPS : volumes per second
#     - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
#         this assumes no dead time in the scan, which is unlikely!)
#     - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
#         distorted volumes (or non-distorted axes)*
#     - shape : tuple of ints.  The shape of each volume, automatically determined
#         from the first volume.  Has 3-4 elemeters, of the form:
#         (depth, height, width [, channels])
#     - dtype : numpy data type
#
#     Info Parameters for Scanning Slope-Distortion
#     ---------------------------------------------
#     - Dz : Displacement of camera from center of the volume.  Usually negative,
#         since the camera should always be in the negative-z direction relative
#         to the volume
#     - Dx, Dy: Displacement of the scanning sheet axis from the center of the
#         volume.  Only one should be specified, depending on the relevant axis
#     - m1x : the slope of the ray leading the right edge of the volume.  Can be
#         computed as -Lx / (2*Dz).
#     - m1z : the slop of the ray leading to the back edge of the volume.  Can be
#         computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
#         the camera axis (as is normally the case).
#     - m0z : the slope of the ray leading to the front edge of the volume.  If
#         not specified, assumed to be -m1z.  *This should only be directly
#         specified for scanning at an oblique angle*
#
#     The additional parameters m0x, m0y, and m1y are determined automatically
#     from the volume size.  These will be made available in the ``computed_info``
#     attributes, or as attributes of the main class itself.
#
#     '''
#
#     # metadata = {}
#
#     def __init__(self, volumes, info={}, name='[VOLUME IN MEMORY]'):
#         self.volumes = volumes
#         self.info = info
#         # self.metadata = metadata
#         vol0 = volumes[0]
#         self.shape = vol0.shape
#         self.dtype = vol0.dtype
#         self.validate_info()
#         self.name = name
#
#
#     def validate_info(self):
#         if not hasattr(self, 'shape'):
#             self.shape = self[0].shape
#
#         xy_ar = self.shape[2] / self.shape[1]
#         xz_ar = self.shape[2] / self.shape[0]
#         self.computed_info = {}
#
#         if 'Lx' in self.info:
#             if 'Dz' in self.info:
#                 raise VolumetricMovieError('%s: Invalid specification for perspective correction on x/y axis\nBoth camera distance and scale directly specified; should be one or the other' % self.name)
#             elif 'Ly' not in self.info:
#                 self.computed_info['Ly'] = self.info['Lx'] / xy_ar
#         elif 'Dz' in self.info:
#             if 'm1x' in self.info:
#                 self.computed_info['m0x'] = -self.m1x
#                 self.comptued_info['m1y'] = self.m1x / xy_ar
#                 self.computed_info['m0y'] = -self.m1y
#             else:
#                 raise VolumetricMovieError('%s: Camera distance (Dz) specified, but slope angle (m1x) not defined' % self.name)
#         else:
#             raise VolumetricMovieError('%s: Scaling info for X/Y requires either Lx or Dz/m1x entries in info; neither found' % self.name)
#
#         if 'Lz' in self.info:
#             if 'Dx' in self.info or 'Dy' in self.info:
#                 raise VolumetricMovieError('%s: Invalid specification for perspective correction on z axis\nBoth scanner distance and scale directly specified; should be one or the other' % self.name)
#         elif 'Dx' in self.info:
#             if 'm1z' not in self.info:
#                 raise VolumetricMovieError('%s: Scanner distance (Dx) specified, but slope angle (m1z) not defined' % self.name)
#             if 'm0z' not in self.info:
#                 self.computed_info['m1z'] = -self.m0z
#             self.computed_info['Lz'] = abs((self.m1z - self.m0z) * self.Dx)
#         elif 'Dy' in self.info:
#             if 'm1z' not in self.info:
#                 raise VolumetricMovieError('%s: Scanner distance (Dy) specified, but slope angle (m1z) not defined' % self.name)
#             if 'm0z' not in self.info:
#                 self.computed_info['m1z'] = -self.m0z
#                 self.computed_info['Lz'] = abs((self.m1z - self.m0z) * self.Dy)
#         else:
#             self.computed_info['Lz'] = self.Lz / xz_ar
#
#
#     def get_info(self):
#         info = self.computed_info.copy()
#         info.udpate(self.info)
#         return info
#
#
#     # def __getattr__(self, name):
#     #     if name in self.info:
#     #         return self.info[name]
#     #     elif name in self.computed_info:
#     #         return self.computed_info[name]
#     #     elif name in self.metadata:
#     #         return self.metadata[name]
#     #     else:
#     #         raise AttributeError("'%s' object has no attribute '%s'" % self.__class__.__name__, name)
#
#
#     def __len__(self):
#         return len(self.volumes)
#
#
#     def __getitem__(self, i):
#         return self.get_volume(i)
#
#
#     def get_volume(self, i):
#         return self.volumes[i]
#
#
#     def __iter__(self):
#         self._iter_index = 0;
#         return self
#
#
#     def __next__(self):
#         if self._iter_index < len(self):
#             vol = self[self._iter_index]
#             self._iter_index += 1
#             # print('iter ->', self._iter_index)
#             return vol
#         else:
#             raise StopIteration
#
#
#     def save(self, fn, file_type=None, print_status=False, start=0, end=None, skip=1, **kwargs):
#         if file_type is None:
#             ext = os.path.splitext(fn)[1].lower()
#             if ext in _file_ext:
#                 file_type = _file_ext[ext]
#             else:
#                 raise ValueError('Filename does not contain one of the recognized extensions: %s\n(Found: "%s")' % (str(sorted(self._file_ext.keys())), ext))
#
#         blosc_options = dict(cname='lz4', shuffle=blosc.NOSHUFFLE, typesize=1)
#         blosc_options.update(kwargs)
#
#         if not end: end = len(self)
#
#         if print_status:
#             sfn = fn
#             if len(fn) > 30: sfn = sfn[:27] + '...'
#             enum = status_range(start, end, skip, post_message='%s:' % sfn)
#         else:
#             enum = range(start, end, skip)
#
#         if file_type == "HDF5":
#             with tables.open_file(fn, mode='w') as f:
#                 comp_filter = tables.Filters(**blosc_options)
#
#                 g = f.create_group('/', 'VolumetricMovie')
#
#                 for n in enum:
#                     f.create_carray(g, 'frame_%08d' % n, obj=self[n], filters=comp_filter)
#
#                 for k, v in self.info.items():
#                     f.set_node_attr(g, k, v)
#
#                 f.flush()
#
#         elif file_type == "MUVI":
#
#             info = dict((k, numpy_to_python_val(v)) for (k, v) in self.info.items())
#
#             # Set up header before opening file.
#             json_bytes = bytes('\n' + json.dumps(info, ensure_ascii=False) + '\n', encoding='utf8')
#             header_length = len(json_bytes)
#             offset = max(28 + header_length, 8192)
#             spacer_bytes = offset - (28 + header_length)
#
#             # Get volume info
#             num_volumes = len(enum)
#             vol0 = self[0]
#
#             # Basic volume info
#             shape = vol0.shape
#             dt = vol0.dtype.type
#             if dt not in _NUMPY_TYPES:
#                 raise ValueError('Data type %s not supported in MUV file!' % repr(dt))
#             dt = _NUMPY_TYPES[dt]
#
#             # Lets pad the header and align with 1024 bit boundaries
#             bin_header_size = align_bdy(offset + 5*8 + 2 + num_volumes*8, 1024) + 1024
#             next_volume_offset = offset + bin_header_size
#             uc_size = None
#
#             volume_offsets = []
#
#             # Check that this is actually a volume
#             if len(shape) == 3:
#                 (depth, height, width) = shape
#                 channels = 1
#             elif len(shape) == 4:
#                 (depth, height, width, channels) = shape
#             else:
#                 raise ValueError('Volumes should have 3 or 4 dimensions, found %d' % (len(shape)))
#
#             if channels not in (1, 2, 3, 4):
#                 raise ValueError('Volume should have 1-4 channels only! (found %d)' % channels)
#
#             # Open file
#             with open(fn, 'wb') as f:
#
#                 # Write header
#                 f.write(b'MUVI')
#                 f.write(struct.pack('<QQQ', 1, offset, header_length))
#                 f.write(json_bytes)
#
#                 # Write preliminary header data; we'll go back later and insert the
#                 #   volume offsets.
#                 # f.write(bytes(offset - f.tell()))
#                 f.seek(offset)
#                 f.write(struct.pack('<QQQQQ', num_volumes, depth, height, width, channels))
#                 f.write(dt)
#                 f.write(struct.pack('<Q', next_volume_offset))
#
#                 # Iterate over volumes, writing each
#                 for i in enum:
#                     # Convert to bytes
#                     vol = self[i].tostring()
#
#                     # Get uncompressed size of a single volume
#                     if uc_size is None:
#                         uc_size = len(vol)
#                         if uc_size > 2**31:
#                             raise ValueError('The uncompressed size of a single volume is > 2GB; this is currently unsupported by the file writer.')
#
#                     # Check size
#                     if uc_size != len(vol):
#                         raise ValueError('Volume %d (0-indexed) does not have same size as first volume!' % i)
#
#                     # Compress volume
#                     cvol = blosc.compress(vol, **blosc_options)
#
#                     # Jump to volume offset
#                     # f.write(bytes(next_volume_offset - f.tell()))
#                     f.seek(next_volume_offset)
#                     nb = len(cvol)
#                     f.write(struct.pack('<Q', nb))
#                     f.write(cvol)
#
#                     # Keep track of current offset, find next one.
#                     volume_offsets.append(next_volume_offset)
#                     next_volume_offset = align_bdy(next_volume_offset + 8 + nb, 1024)
#
#                 # Go back and write offsets
#                 f.seek(offset + 5*8 + 2)
#                 for n in volume_offsets:
#                     f.write(struct.pack('<Q', n))
#
#                 f.close()
#
#         else:
#             raise ValueError("Unrecognized file type.  Valid options: ['HDF5', 'MUVI']")
#
#
# class SparseMovie(VolumetricMovie):
#     def __init__(self, fn, group='/VolumetricMovie'):
#         from . import sparse
#
#         self._sp = sparse.Sparse4D(fn)
#         v = self[0]
#
#         self.info = {
#             'Lx': v.shape[2],
#             'Ly': v.shape[1],
#             'Lz': v.shape[0],
#         }
#
#         self.name = fn
#
#         self.validate_info()
#
#
#     def __len__(self):
#         return len(self._sp)
#
#
#     def get_volume(self, i):
#         return self._sp[i]
#
#
# class CineMovie(VolumetricMovie):
#     def __init__(self, fn, fpv=512, offset=0, fps=None, info=None, clip=80, top=100, gamma=False):
#         from .cine import Cine
#
#         self._f = Cine(fn)
#
#         if fps is None: fps = fpv
#         self.fps = fps
#         self.fpv = fpv
#         self.offset = offset
#         self.clip = clip
#         self.top = top
#         self.gamma = gamma
#
#         self.info = {
#             'Lx': self._f.width,
#             'Ly': self._f.height,
#             'Lz': fps,
#             'gamma': 2 if gamma else 1
#         }
#
#         self.info.update()
#         self.validate_info()
#         self.name = fn
#
#
#     def __len__(self):
#         return len(self._f)//self.fps
#
#     def get_volume(self, i):
#         i0 = self.offset + i * self.fps
#
#         vol = np.empty((self.fps, self._f.height, self._f.width), dtype='u1')
#
#         for i in range(self.fpv):
#             frame = (np.clip(self._f[i+i0], self.clip, self.clip + self.top) - self.clip).astype('f')
#             frame /= self.top
#             if self.gamma: frame = np.sqrt(frame)
#             vol[i] = (frame*255).astype('u1')
#
#         return vol
#
#
#
#     def close(self):
#         self._f.close()
#
#
# _MUV_TYPES = {
#     b'u1': np.uint8,
#     b'u2': np.uint16,
#     b'u4': np.uint32,
#     b's1': np.int8,
#     b's2': np.int16,
#     b's4': np.int32,
#     b'hf': np.float16,
#     b'ff': np.float32,
#     b'df': np.float64
# }
#
# _NUMPY_TYPES = dict((v, k) for k, v in _MUV_TYPES.items())
#
# def ceil_div(a, b): return -(-a//b)
# def align_bdy(a, b): return ceil_div(a, b) * b
#
#
# def numpy_to_python_val(obj):
#     if np.issubdtype(type(obj), np.integer):
#         return int(obj)
#     elif np.issubdtype(type(obj), np.floating):
#         return float(obj)
#     elif np.issubdtype(type(obj), np.complexfloating):
#         return complex(obj)
#     else:
#         return obj


# if __name__ == '__main__':
#     example = VolumeProperties(dx=100/256, Nz=256, Ns=300, dt=0.01, units='mm')
#     example.to_file('volume_properties.xml')

# class MuviMovie(VolumetricMovie):
#     def __init__(self, fn, info={}):
#         self._f = open(fn, 'rb')
#         if self._f.read(4) != b'MUVI':
#             raise ValueError('%s does not appear to be a movie file (first 4 bytes are not "MUVI")' % fn)
#
#         (self.version, ) = struct.unpack('<Q', self._f.read(8))
#         if self.version != 1:
#             raise ValueError('Version number of MUVI file is %d; only 1 is supported' % self.version)
#
#         (self.bin_header_offset, self.header_length) = struct.unpack('<QQ', self._f.read(16))
#         header = self._f.read(self.header_length)
#         # print(repr(header))
#         self.info = json.loads(header)
#         self.info.update(info)
#
#         self._f.seek(self.bin_header_offset)
#         self.num_volumes, self.depth, self.height, self.width, self.channels = \
#             struct.unpack('<QQQQQ', self._f.read(40))
#         dt = self._f.read(2)
#         if dt not in _MUV_TYPES:
#             raise ValueError('MUVI file (%s) had data type "%s", which is not supported.' % (fn, dt))
#         self.dt = _MUV_TYPES[dt]
#
#         self.shape = (self.depth, self.height, self.width, self.channels)
#         self.volume_offsets = struct.unpack('<%dQ' % self.num_volumes, self._f.read(8 * self.num_volumes))
#
#         self.validate_info()
#         self.name = fn
#
#
#     def __len__(self):
#         return self.num_volumes
#
#
#     def get_volume(self, i):
#         self._f.seek(self.volume_offsets[i])
#         (nbytes, ) = struct.unpack('<Q', self._f.read(8))
#
#         raw = self._f.read(nbytes)
#         (bbytes, ) = struct.unpack('<L', raw[12:16])
#         if bbytes != nbytes:
#             raise ValueError('Error loading volume %d in "%s"; blosc size does not match expected.\n(May be caused by >2GB volumes, which are not supported by this reader.)' % (i, self.name))
#
#         return np.frombuffer(blosc.decompress(raw), dtype=self.dt).reshape(self.shape)
#
#
#     def close(self):
#         self._f.close()
#
#
# class HDF5Movie(VolumetricMovie):
#     def __init__(self, fn, group='/VolumetricMovie'):
#         self._f = tables.open_file(fn, 'r')
#         self.movie_node = self._f.get_node('/VolumetricMovie')
#         self.frames = sorted(filter(lambda n: n.startswith('frame_'), dir(self.movie_node)))
#         self.info = {n:self.movie_node._v_attrs[n] for n in self.movie_node._v_attrs._v_attrnamesuser}
#
#         self.validate_info()
#         self.name = fn
#
#
#     def __len__(self):
#         return len(self.frames)
#
#
#     def get_volume(self, i):
#         return np.asarray(self.movie_node[self.frames[i]])
#
#
#     def close(self):
#         self._f.close()
#
#
# # _file_types = sorted(set(_file_ext.values()))
# _file_types = {
#     'HDF5': HDF5Movie,
#     'S4D': SparseMovie,
#     'CINE': CineMovie,
#     'MUVI': MuviMovie
# }
#
#



# def open_4D_movie(fn, file_type=None, *args, **kwargs):
#     if file_type is None:
#         ext = os.path.splitext(fn)[1].lower()
#         if ext in _file_ext:
#             file_type = _file_ext[ext]
#
#     if file_type not in _file_types:
#         raise ValueError('File extension %s not supported.' % ext)
#
#     return _file_types[file_type](fn, *args, **kwargs)


# class HDF5Movie(object):
#     '''Class for storing Volumetric Movies as HDF5 files.
#     Can be used for reading or writing.
#
#     Parameters
#     ----------
#     fn : string
#         filename
#     mode: string
#         read-write mode (default: "r"), passed directly to PyTables
#     compression : bool or string
#         If true, when writing data will be saved in (default: True, or determined from file)
#         If a string is specified it should correspond to a PyTables compression method.
#     compression_level : int
#         If compression is specified, this is the compression level passed to
#         PyTables (default: 5)
#     distortion : dict
#         If specified, should correspond to parameter (string): float values.
#     info : dict
#         Arbitrary parameters to be saved in file as JSON object.
#     group : string (or bytes)
#         The name of the group in the HDF file where the movie is stored.
#         (default: VolumetricMovie)
#
#     Attributes
#     ----------
#     info : dictionary.  Arbitrary user data.  Can be modified by user, but
#         not written to file unless a frame is added or ``write_header`` is
#         called.
#     shape : length 3 tuple.  The shape of each stored volume.
#         Will be undefined for a new volume until the first frame is written.
#     dtype : numpy data type.  Will be undefined for a new volume until the
#         first frame is written.
#     fn : filename
#     distortion : distortion model as a dict.
#     frame_size : integer; raw frame size in bytes.
#
#     None of the attributes other than info should ever be modified by the user!
#     '''
#
#     def __init__(self, fn, mode="r", compressed=True, compression_level=5, distortion={}, info={}, group='/VolumetricMovie'):
#         self._f = tables.open_file(fn, mode=mode)
#
#         if self.mode == 'a':
#             if os.path.exists(fn):
#                 mode = 'r+'
#             else:
#                 mode = 'w'
#
#         self.mode = mode
#         self.fn = fn
#         self.distortion = distortion
#         self.info = info.copy()
#         self.group = ('' if group.startswith('/') else '/') + group
#
#         if self.mode in ('r', 'r+'):
#             if self.group in self._f:
#
#             else:
#
#         if self.mode == "w":
#                 pass
#
#     # def write_info(self):
#     #
#     #
#     # def read_info(self):
#
#     def add_frame(self, arr):
#         """Append a frame to the volumetric movie.
#
#         Parameters
#         ----------
#         arr : array to write; shape should match that of volume unless
#             movie is empty; in this case the shape will be defined accordingly.
#         """
#         arr = np.asarray(arr)
#         if arr.ndim not in (3, 4):
#             raise ValueError("arr should be a 3D or 4D array!")
#
#         if self.header is None:
#             self.dtype = arr.dtype
#             self.header = {
#                 "_required_":["shape", "frame_offsets", "compression", "dtype", "distortion"],
#                 "shape": arr.shape,
#                 "frame_offsets":[],
#                 "compression": "lz4" if self.compressed else "raw",
#                 "dtype": self.dtype.name,
#                 "distortion": self.distortion,
#             }
#             self.shape = arr.shape
#             self.count = np.prod(self.shape)
#             self.frame_size = self.count * self.dtype.itemsize
#             self.cache_frames = self.cache_size // self.frame_size
#             self._frame_offsets = []
#
#         if self.shape != arr.shape:
#             raise ValueError("Shape of input array must match rest of volumetric movie!")
#
#         if self.dtype != arr.dtype:
#             raise ValueError("Type of input array must match rest of volumetric movie!")
#
#         if not hasattr(self, "_write_at"):
#             if self.compressed:
#                 self._f.seek(self._frame_offsets[-1])
#                 self._write_at = self._frame_offsets[-1] + 8 + struct.unpack("<Q", self._f.read(8))[0]
#             else:
#                 self._write_at = self._frame_offsets[-1] + self.frame_size
#
#         self._frame_offsets.append(self._write_at)
#         self.write_header()
#
#         b = arr.tobytes()
#         self._f.seek(self._write_at)
#         if self.compressed:
#             b = lz4framed.compress(b)
#             self._write_at += 8 + len(b)
#             self._f.write(struct.pack("<Q", len(b)))
#             self._f.write(b)
#         else:
#             self._write_at += len(b)
#             self._f.write(b)
#
#     def __len__(self):
#         return len(self._frame_offsets)
#
#
#     def frame(self, n):
#         if n in self._cache_i:
#             return self._cache[self._cache_i.index(n)]
#
#         else:
#             #If needed, remove items in cache.
#             while len(self._cache) > self.cache_frames - 1:
#                 self._cache.pop(0)
#                 self._cache_i.pop(0)
#
#             self._f.seek(self._frame_offsets[n])
#             if self.compressed:
#                 nb = struct.unpack("<Q", self._f.read(8))[0]
#                 b = self._f.read(nb)
#                 arr = np.frombuffer(lz4framed.decompress(b), dtype=self.dtype, count=self.count).reshape(self.shape)
#             else:
#                 arr = np.fromfile(self._f, dtype=self.dtype, count=self.count).reshape(self.shape)
#             self._cache_i.append(n)
#             self._cache.append(arr)
#             return arr
#
#
#     def close(self):
#         self._f.close()
#
#     def __getitem__(self, n):
#         return self.frame(n)
#
#     def __iter__(self):
#         self._iter_n = 0
#         return self
#
#     def __next__(self):
#         if self._iter_n >= (len(self)):
#             self._iter_n = 0
#             raise StopIteration
#
#         else:
#             self._iter_n += 1
#             return self[self._iter_n-1]
