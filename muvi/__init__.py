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
import warnings
import os
import sys
import lz4.block
import concurrent.futures
from xml.etree import ElementTree as ET
import time
from .distortion import get_distortion_map

class VolumetricMovieError(Exception):
    pass

# Create a unique object for use below...
_not_a_default = object()

def _xml_indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            _xml_indent(elem, level+1)
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
        # 'gamma': 1.0,
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
        'top': int,
        'gamma': float,
        'black_level': int,
        'white_level': int,
        'dark_clip': float,
        'flip_y': bool,
        'source_filename': str,
        'setup_filename': str,
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
        top : int
            When converting 2D movies, the maximum value used to rescale data.
        gamma : float (default: 1.0)
            The gamma correction applied to the raw data.  The actual intensity
            recorded by the camera is (output_value)^(gamma).
        black_level : int
            The raw pixel value from the camera which corresponds to 0
            intensity.  Default depends on camera model (for Phantom camera,
            this is 64 for a 12 bit readout).  This will be mapped to 0 in the
            output data.
        white_level : int
            The raw pixel value which is considered maximum intensity.  Will be
            mapped to the maximum value in the output (255 for 'u1' data type).
            Default depends on camera model (for Phantom cameras, this is 4096
            for a 12 bit readout).
        dark_clip : float
            The *relative* intensity which will be clipped to 0.
        flip_y : bool (default: True)
            If true, flip the y-axis from the raw image data.  Generally
            speaking, y in 2D images increases as you move down on the screen.
            In the 3D volumes, y is "up", and so usually requires flipping.
        source_filename : str
            The filename of the source data.  Usually defined only for movies
            derived from 2D streams.
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

    def copy(self):
        return VolumeProperties(self)

    def __getitem__(self, key):
        if key in self._d:
            return self._d[key]
        elif key in self._defaults:
            return self._defaults[key]
        elif key in self._alternates:
            return self[self._alternates[key]]
        else:
            raise KeyError(key)

    def get(self, key, default=_not_a_default):
        try:
            return self[key]
        except KeyError:
            if default is _not_a_default:
                raise KeyError(key)
            else:
                return default

    def get_list(self, *keys):
        return [self[key] for key in keys]

    def items(self):
        return self._d.items()

    def keys(self):
        return self._d.keys()

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
            _xml_indent(base, indent)
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


    def update_from_file(self, filename, tag='VolumeProperties'):
        '''Update object from an XML file.

        Parameters
        ----------
        filename : str
            Filename
        tag : str (default: `"VolumeProperties"`)
            Tag which contains the properties.  The reader will search for any
            instances of this tag and extract information from them.
        '''
        root = ET.parse(filename).getroot()
        if root.tag != tag:
            raise VolumetricMovieError("'%s' is not a VolumeProperties file\n  (root tag was '%s'; should be '%s')" % (filename, root.tag, tag))
        else:
            self.update(root)

        # self._search_tree(root, tag, warn=warn, raise_errors=raise_errors)


    # def _search_tree(self, e, tag, warn=False, raise_errors=True):
    #     if e.tag == tag:
    #         self.update(e, warn=warn, raise_errors=raise_errors)
    #     else:
    #         for ee in e:
    #             self._search_tree(ee, tag, warn=warn, raise_errors=raise_errors)


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



class _status_range:
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


class _status_enumerate(_status_range):
    '''Similar to `_status_range` function, except acts like an `enumerate`
    iterator instead of a `range` iterator.
    '''

    def __init__(self, obj, **kwargs):
        super().__init__(len(obj), **kwargs)
        self.obj = obj


    def get_next(self):
        return (self.count, self.obj[self.count])


class VolumetricMovie:
    '''Base class for working with volumetric movies.

    Parameters
    ----------
    info : VolumeProperties object
        Storage for the volume metadata

    Methods
    -------
    validate_info()
        Auto determines properties of the volumes from the first frame.
    update_distortion()
        Generates the distortion map info from the volume info.
    get_volume(index)
        Retrieves a volume.  Note that volumes may also be accessed by treating
        the VolumetricMovie object like a list; this is preferable for most
        uses.  (This method is called by __item__; if implementing a derived
        class, you should *only* redefine `get_volume`.)
    close()
        Closes the input files (if any)
    save(fn)
        Saves the movie to a VTI file.
    start_vti_write(fn)
        Starts a write to a VTI file.  Used when an external utility wants to
        track the saving process over time.
    write_vti_frame()
        Writes a single frame, and returns the number of frames remaining.
        Write must be started with `start_vti_write` first.

    If you would like to suppport a new data type, it is suggest you inheret
    from this class.  To make a new class which is supported by the viewer,
    you must meet the following requirements:
        1. Define a new `__init__` and `get_volume` method.
        2. The initialization must create an `info` attribute which is
            a `VolumeProperties` object, and populate it as needed.  (The
            `validate_info` method can take care of many basic attributes.)
    '''

    def __init__(self, source, **kwargs):
        '''
        Parameters
        ----------
        source : list/array-like object
            An object which supports indexing and returns a 3D or 4D numpy array.

        Any keywords are passed directly to the `info` attribute, which is a
        VolumeProperties object.
        '''

        self.volumes = source
        self.info = VolumeProperties(**kwargs)
        self.info['Nt'] = len(source)
        self.validate_info()

    def validate_info(self):
        '''
        Infer properties of volume from first frame.
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
            if val is not None:
                if key not in self.info:
                    self.info[key] = val
                else:
                    if self.info[key] != val:
                        raise VolumetricMovieError("validation error in %s:\n    info property '%s' does not agree with value infered from first frame (%s != %s)" % (repr(self), key, self.info[key], val))

        if 'creation_time' not in self.info:
            self.info['creation_time'] = time.time()

        self.update_distortion()


    def update_distortion(self):
        '''Update distortion map.

        Should only need to be called diectly if volume info has changed
        since initilization (which shouldn't usually happen).
        '''
        self.distortion = distortion.get_distortion_map(self.info)


    def get_volume(self, index):
        '''
        '''
        return self.volumes[index]


    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.get_volume(i) for i in key]
        else:
            return self.get_volume(key)


    def __len__(self):
        return self.info['Nt']


    def close(self):
        pass


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
            enum = _status_range(0, num_frames, post_message='%s:' % sfn)
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
'''.format(**vol_info).encode('UTF-8'))
        info = self.info.copy()
        info['Nt'] = end - start
        self._write_file.write(info.xml(indent=1))
        self._write_file.write(r'''  <ImageData WholeExtent="{x0} {x1} {y0} {y1} {z0} {z1}" Origin="{x0} {y0} {z0}" Spacing="{dx} {dy} {dz}" TimeValues="{time_values}">
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


def open_3D_movie(fn, file_type=None, **kwargs):
    '''Open a 3D movie from a file on disk.

    Parameters
    ----------
    fn : str
        The input filename.  Currently supports 'vti' and 'cine' files.
    file_type : str or None
        If defined, treat the file as if it has this extension.

    Keyword arguments are passed directly to the volumetric movie object, and
    will in general be passed into the info attribute.  This can be used, for
    example, to adjust distortion parameters.
    '''
    if file_type is None:
        file_type = os.path.splitext(fn)[1]
    file_type = file_type.lower().lstrip('.')

    if file_type == 'vti':
        from .readers.vti import VTIMovie
        return VTIMovie(fn, **kwargs)
    elif file_type == 'cine':
        from .readers.cine import Cine
        return VolumetricMovieFrom2D(fn, Cine, **kwargs)
    elif file_type == 'seq':
        import pims
        return VolumetricMovieFrom2D(fn, pims.open, **kwargs)
    else:
        raise ValueError("3D Movie file with extension '%s' not supported" % file_type)

# Added for backward compatibility.  Really, this is a 3D movie, which is 4D
#   data, not a 4D movie (which would be 5D data).
open_4D_movie = open_3D_movie

class VolumetricMovieFrom2D(VolumetricMovie):
    '''
    A class used to create 3D movies from 2D movies.

    Methods
    -------
    locate_info()
        Automatically locate `.xml` file with VolumeProperties info.
    validate_2D()
        Determine VolumeProperties from first frame (should be called *after*
        `locate_info`).
    get_frame(index)
        Retrieve a single frame from the 2D movie.

    In general, it should not be nessesary to create descendents from this
    class.  If you would like to implement a reader for a new type of 2D movie,
    the class can be passed to the `reader` method of initialization.  All
    other functions should be handled automatically.
    '''

    # If True, the frames already have tone mapping applied.  False in general,
    #  but may be true for derived classes (e.g. Cine)
    _FRAMES_TONE_MAPPED = False
    _DEFAULT_INFO = {
        'gamma': 1.0,
        'flip_y': True,
        'dtype': 'B',
    }

    _TONE_MAP_KEYS = ('black_level', 'white_level', 'gamma', 'dark_clip')

    def __init__(self, filename, reader, setup_xml=None, **kwargs):
        '''
        Parameters
        ----------
        filename : str
        reader : class
            This class should take the filename as the first argument, and
            return an object which behaves like a list of 2D arrays.  See
            `readers/cine.py` for an example.
        setup_xml : str or None
            If defined, use this file to determine the VolumeProperties
            (i.e. the `info` attribute).  If not defined, will attempt to
            locate an `.xml` file with the same basename as the input, or
            a file called `muvi_setup.xml` in the same directory.
        '''
        self.info = VolumeProperties(self._DEFAULT_INFO)

        self.info['source_filename'] = filename

        if setup_xml is not None:
            self.info['setup_filename'] = setup_xml
            self.info.update(setup_xml)
        else:
            self.locate_info()

        self.info.update(kwargs)

        if getattr(reader, "_MUVI_SUPPORTS_TONE_MAP", False):
            self.needs_tone_map = False
            tone_map = {k:v for k, v in self.info.items() if k in self._TONE_MAP_KEYS}
            self.frames = reader(filename, **tone_map)
            for key in self._TONE_MAP_KEYS:
                val = getattr(self.frames, key, None)
                if val is not None:
                    self.info[key] = val
        else:
            self.needs_tone_map = True
            self.frames = reader(filename)

        self.validate_2D()

    def locate_info(self):
        '''
        Attempt to auto-locate an `.xml` file with the VolumeProperties info.
        '''
        bfn = os.path.splitext(self.info['source_filename'])[0]
        bdir = os.path.split(bfn)[0]

        fn1 = bfn + '.xml'
        fn2 = os.path.join(bdir, 'muvi_setup.xml')

        if os.path.exists(fn1):
            self.info.update_from_file(fn1)
            self.info['setup_filename'] = fn1
        elif os.path.exists(fn2):
            self.info.update_from_file(fn2)
            self.info['setup_filename'] = fn2
        else:
            warnings.warn("failed to find 3D setup in normal locations\n (checked '%s' and '%s')" % (fn1, fn2))


    def validate_2D(self):
        '''
        Auto-determine the Volume Properties from the first frame of the movie.
        '''
        if 'Nz' not in self.info:
            raise ValueError("if deriving volumes from 2D data, 'Nz' must be defined")
        if 'Ns' not in self.info:
            warnings.warn("'Ns' not defined for volumes derived from 2D data\n   Will default to Ns=Nz, but this is probably wrong!")
        if self.info['Ns'] < self.info['Nz']:
            raise ValueError("Ns must be >= Nz")

        dead_frames = self.info['Ns'] - self.info['Nz']
        if 'offset' not in self.info:
            self.info['offset'] = 0

        Nt = (len(self.frames) - self.info['offset'] + dead_frames) \
                // self.info['Ns']
        if 'Nt' in self.info:
            if Nt < self.info['Nt']:
                warnings.warn('specified number of volumes (Nt=%d) is more than stored in frames (%d)\n  (changing Nt to latter)' % (self.info['Nt'], Nt))
            self.info['Nt'] = Nt
        else:
            self.info['Nt'] = Nt

        if 'creation_time' not in self.info:
            self.info['creation_time'] = time.time()

        Ny, Nx = self.get_frame(0).shape
        self.info['Nx'] = Nx
        self.info['Ny'] = Ny

        if 'channels' in self.info:
            self.vol_shape = (self.info['Nz'], self.info['Ny'], self.info['Nx'], self.info['channels'])
        else:
            self.vol_shape = (self.info['Nz'], self.info['Ny'], self.info['Nx'])

        self.vol_dtype = self.info['dtype']

        self.update_distortion()

    def get_frame(self, i):
        return self.frames[i]

    def get_volume(self, i):
        vol = np.empty(self.vol_shape, dtype=self.vol_dtype)

        y_step = -1 if self.info.get('flip_y', False) else 1

        offset = self.info.get('offset', 0) + i * self.info['Ns']

        if len(self.vol_shape) == 4:
            raise ValueError('color reading not (yet) supported')
        else:
            for z in range(self.vol_shape[0]):
                vol[z, ::y_step] = self.get_frame(offset + z)

        return vol
        # return vol[128:384, 128:384, 128:384]
