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

import numpy as np
from ..filetypes.vtk import VTKTag, VTKWriter, VTKReader, DataDict
from .. import _status_enumerate, _status_range
import os


def vec_names(N, prefix=None):
    if prefix:
        prefix = prefix + '.'
    else:
        prefix = ''

    if N <= 4:
        return [prefix + 'x', prefix + 'y', prefix + 'z', prefix + 'w'][:N]
    else:
        return [prefix + str(n+1) for n in range(N)]


class PointItems:
    def __init__(self, points, force_floats=None, force_length=None):
        '''See `Points.items` method for more information.'''
        self.points = points
        self.ff = force_floats
        self.fl = force_length
        self.keys = set(self.points.keys())

    def __iter__(self):
        return self

    def __next__(self):
        if not self.keys:
            raise StopIteration

        k = self.keys.pop()
        return k, self.points.force(k, self.ff, self.fl)


class Points:
    def __init__(self, pos, display={}, metadata={}, **attr):
        '''A class for holding data on points in space.

        For the most part, behaves like a dictionary, with multiple attributes
        which can be attached to each point.  At the very least, the `pos`
        attribute is always defined.  Note that all the attributes must have
        the same length!

        Parameters
        ----------
        pos : (N, Nd) shaped array
            The point coordinates.  Should be a numpy array or castable to one

        Any additional keywords are attached as attributes.  Note that the
        length of each *must* match that of `pos`.

        Keywords
        --------
        display: Dictionary (default: {})
            A dictionary of attributes used by the Muvi software for displaying
            the points.  Can be used to get a default representation when the
            data is loaded.  This information will be saved in VTK files.
        metadata: Dictionary (default: {})
            A dictionary of arbitrary attributes, used for user metadata.
            This information will be saved in VTK files.

        Example
        -------
        ~~~
        # Create a new points array with mass and velocity attached attributes
        # Note that velocity is a vector, and mass is a scalar, but both have
        #   the same length as the position vectors!
        points = Points(
            [(0, 1, 2), (3, 4, 5],
            vel = [(1, 2, 3), (-1, -2, -3)],
            mass = [2, 3]
        )

        # Get mass and promote order of array
        m = points['mass'].reshape(-1, 1)

        # Set some new attributes
        points['p'] = points['vel'] * m
        points['moment'] = points['pos'] * m

        # Save to a VTP file
        points.save('test.vtp')
        ~~~
        '''

        self._d = {}
        self['pos'] = pos

        if not isinstance(display, dict):
            raise ValueError('display keyword must be dictionary')
        self.display = display

        if not isinstance(metadata, dict):
            raise ValueError('display keyword must be dictionary')
        self.metadata = metadata

        for k, v in attr.items():
            self[k] = v

    def __len__(self):
        return self._N

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, val):
        val = np.asarray(val)
        if key == 'pos' and 'pos' not in self._d:
            self._N = len(val)
        if len(val) != self._N:
            raise ValueError(f'Length of attribute ({len(val)}) does not match pos ({self._N})')
        self._d[key] = val

    def keys(self):
        '''Return an iterable of all the attributes'''
        return self._d.keys()

    def __contains__(self, key):
        return key in self._d

    def items(self, force_floats=None, force_length=None):
        '''Return an iterator class with the attribute names and values of the
        points.

        Keywords
        --------
        force_floats : None (default) or numpy data type
            If specified, all floating point attributes are forced to this
            type.
        force_length : None (default) or int
            If specified, data array is forced to this length.  If the length
            is shorter, the data is clipped, if the length is longer the extra
            values are filled with nan's.

        Returns
        -------
        items : PointItems
            An iterable which behaves like dict.items()
        '''

        return PointItems(self, force_floats=force_floats, force_length=force_length)

    def force(self, key, force_floats=None, force_length=None):
        '''Return a data array for one of the attributes, forcing the length
        and/or type.

        Arguments
        ---------
        key : string
            The attribute key

        Keywords
        --------
        force_floats : None (default) or numpy data type
            If specified, all floating point attributes are forced to this
            type.
        force_length : None (default) or int
            If specified, data array is forced to this length.  If the length
            is shorter, the data is clipped, if the length is longer the extra
            values are filled with nan's.

        Returns
        --------
        data : numpy array
            The data with specified type and size.  If no changes are required,
            the original array is returned, otherwise a copy is made.
        '''
        data = self[key]
        N = len(data)

        # Figure out output data type
        # If force float is enabled and we're a float, convert
        if force_floats and data.dtype in ('f', 'd'):
            dtype = force_floats
        # Otherwise leave it alone
        else:
            dtype = data.dtype

        # Are we forcing the length?  And if so, is it wrong?
        if force_length and N != force_length:
            shape = (force_length,) + data.shape[1:]
            if force_length > N:
                dr = np.full(shape, np.nan, dtype)
                dr[:N] = data
                # Here the type is already fixed, so we can return
                return dr
            else:
                data = data[:force_length]
                # Don't return... let the retyping happen below

        # Do we need to retype?
        if data.dtype != dtype:
            return data.astype(dtype)
        else:
            return data

    def save(self, fn, filetype=None, force_floats='f', display=None):
        '''Save the points into a file.

        Parameters
        ----------
        fn : string
            The filename to save to

        Keywords
        --------
        filetype : string ('vtp' or 'csv')
            The type of file to save.  If not specified, determined from the
            extension.
        force_floats : None or numpy data type (default: "f")
            If defined, force all floating point values to this type.  Usually
            used to convert to single precision, which is the default.  Set to
            `None` to leave the types as is.
        '''

        if filetype is None:
            filetype = os.path.splitext(fn)[1][1:].lower()

        if filetype == 'csv':
            headers = []
            data = []

            for k, d in self.items(force_floats=force_floats):
                if k == 'pos':
                    headers = vec_names(d.shape[1]) + headers
                    data.insert(0, d)
                elif d.ndim == 1:
                    headers.append(k)
                    data.append(d.reshape(-1, 1))
                elif d.ndim == 2:
                    headers += vec_names(d.shape[1], k)
                    data.append(d)
                else:
                    raise ValueError("can't save attributes which are not vectors or scalars")

            np.savetxt(fn, np.hstack(data), fmt='%s', delimiter=',', header=','.join(headers))

        elif filetype == 'vtp':
            point_data = []
            scalars = []
            vectors = []

            for k, d in self.items(force_floats=force_floats):
                if k == 'pos':
                    pos = VTKTag('DataArray', d, NumberOfComponents=d.shape[1])
                elif d.ndim == 1:
                    scalars.append(k)
                    point_data.append(VTKTag('DataArray', d, Name=k))
                elif d.ndim == 2:
                    vectors.append(k)
                    point_data.append(VTKTag('DataArray', d, Name=k, NumberOfComponents=d.shape[1]))
                else:
                    raise ValueError("can't save attributes which are not vectors or scalars")

            contents = [
                VTKTag('Points', pos),
                VTKTag('PointData', point_data, Scalars=' '.join(scalars), Vectors=' '.join(vectors))
            ]

            with VTKWriter(fn, 'PolyData') as f:
                if self.display:
                    f.write_tag(DataDict('MuviDisplay', self.display))
                if self.metadata:
                    f.write_tag(DataDict('UserData', self.metadata))

                f.write_tag(VTKTag('PolyData', [
                    VTKTag('Piece', contents, NumberOfPoints = len(self)),
                ]))
        else:
            raise ValueError('filetype not recognized!')


class PointSequenceItems:
    def __init__(self, seq):
        '''See `PointSequence.items` method for more information.'''
        self.seq = seq
        self.keys = sorted(self.seq.keys(), reverse=True)

    def __iter__(self):
        return self

    def __next__(self):
        if not self.keys:
            raise StopIteration

        k = self.keys.pop()
        return k, self.seq[k]


class PointSequence:
    def __init__(self, points, display=None, metadata=None):
        '''A parent class for sequences of points.  Behaves like a dictionary
        with *integer* keys containing Points objects.

        Parameters
        ----------
        points : iterable of Points objects
            If list-like, timesteps are assumed to be sequential and
            starting at 0.  Alternatively, if different numbering is required,
            pass a dictionary with *integer* keys and Points objects as the
            values

        Keywords
        --------
        display: dictionary or None (default: None)
            A dictionary of attributes used by the Muvi software for displaying
            the points.  Can be used to get a default representation when the
            data is loaded.  If None, taken from the first point in the
            sequence.  This information will be saved in VTK files.
        metadata: Dictionary or None (default: None)
            A dictionary of arbitrary attributes, used for user metadata.
            This information will be saved in VTK files.
        '''

        if isinstance(points, dict):
            time_steps = sorted(points.keys())
        else:
            time_steps = range(len(points))

        self._N = 0
        self._d = {}

        for i in time_steps:
            dat = points[i]

            if display is None and hasattr(dat, 'display'):
                display = dat.display
            if metadata is None and hasattr(dat, 'metadata'):
                metadata = dat.metadata

            self._d[i] = dat
            if len(dat) > self._N:
                self._N = len(dat)

        self.current = None

        if not isinstance(display, dict):
            raise ValueError('display keyword must be dictionary')
        self.display = display

        if not isinstance(metadata, dict):
            raise ValueError('metadata keyword must be dictionary')
        self.metadata = metadata


    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        if not isinstance(i, int):
            raise ValueError('index must be an integer')

        # Don't fetch new data if we are asking for the same point
        # Useful when loading from disk!
        if i != getattr(self, 'current', None):
            self.current_data = self._get(i)
            self.current = i

        if hasattr(self, '_computed'):
            for key, func in self._computed:
                self.current_data[key] = func(self.current_data)

        return self.current_data

    def add_computed_field(self, field, func):
        '''Added a field for each time frame which is automatically
        computed and added to the Points object when it is requested.

        This will also add the computed field to data which is subsequently
        saved.

        Parameters
        ----------
        field : str
            The name of the field to be created
        func : function with one argument
            A function which takes a Points object and returns an array of
            shape (N, ...)
        '''
        if not hasattr(self, '_computed'):
            self._computed = []

        self._computed.append((field, func))


    def _get(self, i):
        if i not in self._d:
            raise KeyError(f"Invalid timestep: {i}")
        return self._d[i]

    def items(self):
        '''Return an iterator over the timesteps and Points arrays.

        Behaves like dict.items(), but the order is always sorted.'''

        return PointSequenceItems(self)

    def keys(self):
        return self._d.keys()

    def __contains__(self, key):
        return key in self._d

    def save(self, fn, filetype='vtp', force_floats='f', print_status=False):
        '''Save the points into a file.

        Parameters
        ----------
        fn : string
            The filename to save to

        Keywords
        --------

        filetype : string or None
            The type of file to save.  If not specified, determined from the
            extension.  This value can only be 'vtp' (default), but left here
            for future compatibility.
        force_floats : None or numpy data type (default: "f")
            If defined, force all floating point values to this type.  Usually
            used to convert to single precision, which is the default.  Set to
            `None` to leave the types as is.
        '''

        # f0 : int or None
        #     If specified, the first frame to export
        # f1 : int or None
        #     If specified, the last frame to export

        if filetype is None:
            filetype = os.path.splitext(fn)[1][1:].lower()

        if filetype == 'vtp':
            points = []
            point_data = []
            scalars = set()
            vectors = set()

            if print_status:
                iterator = _status_enumerate(sorted(self.items()), pre_message='Building data: ')
            else:
                iterator = enumerate(sorted(self.items()))

            for i, (t, p) in iterator:
                for k, d in p.items(force_floats=force_floats, force_length=self._N):
                    if k == 'pos':
                        points.append(VTKTag('DataArray', d, NumberOfComponents=d.shape[1], TimeStep=i))
                    elif d.ndim == 1:
                        scalars.add(k)
                        point_data.append(VTKTag('DataArray', d, Name=k, TimeStep=i))
                    elif d.ndim == 2:
                        vectors.add(k)
                        point_data.append(VTKTag('DataArray', d, Name=k, NumberOfComponents=d.shape[1], TimeStep=i))
                    else:
                        raise ValueError("can't save attributes which are not vectors or scalars")


            contents = [
                VTKTag('Points', points),
                VTKTag('PointData', point_data, Scalars=' '.join(scalars), Vectors=' '.join(vectors))
            ]

            with VTKWriter(fn, 'PolyData', print_status=print_status) as f:
                if self.display:
                    f.write_tag(DataDict('MuviDisplay', self.display))
                if self.metadata:
                    f.write_tag(DataDict('UserData', self.metadata))

                f.write_tag(VTKTag('PolyData', [
                    VTKTag('Piece', contents, NumberOfPoints = self._N)
                ], TimeValues=' '.join(map(str, sorted(self.keys())))))
        else:
            raise ValueError('filetype not recognized!')


class PointsFromFile(Points):
    def __init__(self, f):
        '''A class for Points data loaded from a VTP file.

        Parameters
        ----------
        f : string or VTKReader class
            Input file
        '''
        if isinstance(f, VTKReader):
            self.vtk = f
        else:
            self.vtk = VTKReader(f)

        # Check that we have the right type of data in this file.
        if 'TimeValues' in self.vtk.main.attrib:
            raise ValueError(f"'{self.vtk.filename}' contains a PointsSequence object, not Points")
        if self.vtk.main.tag != 'PolyData':
            raise ValueError(f"Wrong VTK file type (found '{vtk.main.tag}', should be 'PolyData')")

        # Manually set number of points -- if this doesn't match data it will
        #   throw an error later.
        self._N = self.vtk.contents.attrib['NumberOfPoints']

        # Get point positions
        points = self.vtk.contents.find('Points/DataArray')
        if points is None:
            raise ValueError(f"VTK file '{self.vtk.filename}' is missing Points/DataArray tag")

        self._d = {}
        self['pos'] = self.vtk.get_data_from_tag(points)

        # Get point data
        point_data = self.vtk.contents.findall('PointData/DataArray')
        for tag in point_data:
            if 'Name' not in tag.attrib:
                raise ValueError(f"VTK file '{self.vtk.filename}' has PointData DataArray which is missing the 'Name' field")
            self[tag.attrib['Name']] = self.vtk.get_data_from_tag(tag)

        self.display = {}
        for tag in self.vtk.root.findall('MuviDisplay'):
            self.display.update(self.vtk.get_dict(tag))

        self.metadata = {}
        for tag in self.vtk.root.findall('UserData'):
            self.metadata.update(self.vtk.get_dict(tag))

class PointSequenceFromFile(PointSequence):
    def __init__(self, f):
        '''A class for PointsSequence data loaded from a VTP file.

        Parameters
        ----------
        f : string or VTKReader class
            Input file
        '''
        if isinstance(f, VTKReader):
            self.vtk = f
        else:
            self.vtk = VTKReader(f)

        # Check that we have the right type of data in this file.
        if 'TimeValues' not in self.vtk.main.attrib:
            raise ValueError(f"'{self.vtk.filename}' contains a Points object, not PointSequence")
        if self.vtk.main.tag != 'PolyData':
            raise ValueError(f"Wrong VTK file type (found '{vtk.main.tag}', should be 'PolyData')")

        # Get the mapping between 'TimeStep' and actual value
        self._ts = tuple(map(int, self.vtk.main.attrib['TimeValues'].split()))

        # Manually set number of points
        self._N = self.vtk.contents.attrib['NumberOfPoints']

        # Make empty data dictionaries
        self._d = {ts:{} for ts in self._ts}

        # Get point positions
        points = self.vtk.contents.findall('Points/DataArray')
        for tag in points:
            if 'TimeStep' in tag.attrib:
                ts = self._ts[int(tag.attrib['TimeStep'])]
                self._d[ts]['pos'] = tag
            else:
                for data in self._d.items():
                    data['pos'] = tag

        if points is None:
            raise ValueError(f"VTK file '{self.vtk.filename}' is missing Points/DataArray tag")

        # Get point data
        point_data = self.vtk.contents.findall('PointData/DataArray')
        for tag in point_data:
            if 'Name' not in tag.attrib:
                raise ValueError(f"VTK file '{self.vtk.filename}' has PointData DataArray which is missing the 'Name' field")
            key = tag.attrib['Name']

            if 'TimeStep' in tag.attrib:
                ts = self._ts[int(tag.attrib['TimeStep'])]
                self._d[ts][key] = tag
            else:
                for data in self._d.items():
                    data[key] = tag

        self.display = {}
        for tag in self.vtk.root.findall('MuviDisplay'):
            self.display.update(self.vtk.get_dict(tag))

        self.metadata = {}
        for tag in self.vtk.root.findall('UserData'):
            self.metadata.update(self.vtk.get_dict(tag))


    def _get(self, i):
        if i not in self._d:
            raise KeyError(f"Invalid timestep: {i}")

        data = {key:self.vtk.get_data_from_tag(tag) for key, tag in self._d[i].items()}
        pos = data.pop('pos')

        good = np.isfinite(pos).all(1)
        if good.any():
            cutoff = np.max(np.where(np.isfinite(pos).all(1))[0])+1
        else:
            cutoff = 0
        pos = pos[:cutoff]
        data = {key:data[:cutoff] for key, data in data.items()}

        return Points(pos, **data)
