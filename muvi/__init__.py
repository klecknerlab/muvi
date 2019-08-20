#!/usr/bin/python3
#
# Copyright 2018 Dustin Kleckner
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
import json
import struct
import warnings
import os


try:
    import tables
    _HAS_TABLES = True
except:
    warnings.warn("pytables module not found, HPF5 loading/saving not available.", RunTimeWarning)
    _HAS_TABLES = False


class VolumetricMovieError(Exception):
    pass


_file_ext = {
    '.h5':'HDF5',
    '.hdf5':'HDF5',
    '.s4d':'S4D',
}

class VolumetricMovie(object):
    '''Generic class for working with volumetric movies.

    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.

    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).

    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.

    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".

    Time scales are always specified in terms of seconds.

    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.

    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type

    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*

    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    '''

    # metadata = {}

    def __init__(self, volumes, info={}, name='[VOLUME IN MEMORY]'):
        self.volumes = volumes
        self.info = info
        # self.metadata = metadata
        vol0 = volumes[0]
        self.shape = vol0.shape
        self.dtype = vol0.dtype
        self.validate_info()
        self.name = name

    def validate_info(self):
        if not hasattr(self, 'shape'):
            self.shape = self[0].shape

        xy_ar = self.shape[2] / self.shape[1]
        xz_ar = self.shape[2] / self.shape[0]
        self.computed_info = {}

        if 'Lx' in self.info:
            if 'Dz' in self.info:
                raise VolumetricMovieError('%s: Invalid specification for perspective correction on x/y axis\nBoth camera distance and scale directly specified; should be one or the other' % self.name)
            elif 'Ly' not in self.info:
                self.computed_info['Ly'] = self.info['Lx'] / xy_ar
        elif 'Dz' in self.info:
            if 'm1x' in self.info:
                self.computed_info['m0x'] = -self.m1x
                self.comptued_info['m1y'] = self.m1x / xy_ar
                self.computed_info['m0y'] = -self.m1y
            else:
                raise VolumetricMovieError('%s: Camera distance (Dz) specified, but slope angle (m1x) not defined' % self.name)
        else:
            raise VolumetricMovieError('%s: Scaling info for X/Y requires either Lx or Dz/m1x entries in info; neither found' % self.name)

        if 'Lz' in self.info:
            if 'Dx' in self.info or 'Dy' in self.info:
                raise VolumetricMovieError('%s: Invalid specification for perspective correction on z axis\nBoth scanner distance and scale directly specified; should be one or the other' % self.name)
        elif 'Dx' in self.info:
            if 'm1z' not in self.info:
                raise VolumetricMovieError('%s: Scanner distance (Dx) specified, but slope angle (m1z) not defined' % self.name)
            if 'm0z' not in self.info:
                self.computed_info['m1z'] = -self.m0z
            self.computed_info['Lz'] = abs((self.m1z - self.m0z) * self.Dx)
        elif 'Dy' in self.info:
            if 'm1z' not in self.info:
                raise VolumetricMovieError('%s: Scanner distance (Dy) specified, but slope angle (m1z) not defined' % self.name)
            if 'm0z' not in self.info:
                self.computed_info['m1z'] = -self.m0z
                self.computed_info['Lz'] = abs((self.m1z - self.m0z) * self.Dy)
        else:
            self.computed_info['Lz'] = self.Lz / xz_ar


    def get_info(self):
        info = self.computed_info.copy()
        info.udpate(self.info)
        return info


    # def __getattr__(self, name):
    #     if name in self.info:
    #         return self.info[name]
    #     elif name in self.computed_info:
    #         return self.computed_info[name]
    #     elif name in self.metadata:
    #         return self.metadata[name]
    #     else:
    #         raise AttributeError("'%s' object has no attribute '%s'" % self.__class__.__name__, name)


    def __len__(self):
        return len(self.volumes)


    def __getitem__(self, i):
        return self.volumes[i]


    def __iter__(self):
        self._iter_index = 0;
        return self


    def __next__(self):
        if self._iter_index < len(self):
            vol = self[self._iter_index]
            self._iter_index += 1
            # print('iter ->', self._iter_index)
            return vol
        else:
            raise StopIteration


    def save(self, fn, file_type=None, complevel=5, complib='blosc:lz4', bitshuffle=False, shuffle=False):
        if file_type is None:
            ext = os.path.splitext(fn)[1].lower()
            if ext in _file_ext:
                file_type = _file_ext[ext]
            else:
                raise ValueError('Filename does not contain one of the recognized extensions: %s\n(Found: "%s")' % (str(sorted(self._file_ext.keys())), ext))

        if file_type == "HDF5":
            with tables.open_file(fn, mode='w') as f:
                comp_filter = tables.Filters(complevel=complevel, complib=complib, shuffle=shuffle, bitshuffle=bitshuffle)

                g = f.create_group('/', 'VolumetricMovie')

                for n, vol in enumerate(self):
                    # print('write ->', n)
                    f.create_carray(g, 'frame_%08d' % n, obj=vol, filters=comp_filter)

                for k, v in self.info.items():
                    f.set_node_attr(g, k, v)

                f.flush()

        else:
            raise ValueError("Unrecognized file type.  Valid options: ['HDF5']")


class SparseMovie(VolumetricMovie):
    def __init__(self, fn, group='/VolumetricMovie'):
        from . import sparse

        self._sp = sparse.Sparse4D(fn)
        v = self[0]

        self.info = {
            'Lx': v.shape[2],
            'Ly': v.shape[1],
            'Lz': v.shape[0],
        }

        self.name = fn

        self.validate_info()


    def __len__(self):
        return len(self._sp)


    def __getitem__(self, i):
        return self._sp[i]



class HDF5Movie(VolumetricMovie):
    def __init__(self, fn, group='/VolumetricMovie'):
        self._f = tables.open_file(fn, 'r')
        self.movie_node = self._f.get_node('/VolumetricMovie')
        self.frames = sorted(filter(lambda n: n.startswith('frame_'), dir(self.movie_node)))
        self.info = {n:self.movie_node._v_attrs[n] for n in self.movie_node._v_attrs._v_attrnamesuser}

        self.validate_info()
        self.name = fn


    def __len__(self):
        return len(self.frames)


    def __getitem__(self, i):
        return np.asarray(self.movie_node[self.frames[i]])


    def close(self):
        self._f.close()


# _file_types = sorted(set(_file_ext.values()))
_file_types = {
    'HDF5': HDF5Movie,
    'S4D': SparseMovie,
}

def open_4D_movie(fn, file_type=None, *args, **kwargs):
    if file_type is None:
        ext = os.path.splitext(fn)[1].lower()
        if ext in _file_ext:
            file_type = _file_ext[ext]

    if file_type not in _file_types:
        raise ValueError('File extension %s not supported.' % ext)

    return _file_types[file_type](fn, *args, **kwargs)

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
