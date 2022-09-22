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
from ..filetypes.vtk import VTKTag, VTKWriter, VTKReader
from .points import PointItems
from .. import _status_enumerate, _status_range
import os

def rectilinear_grid(*axes):
    '''Build the coordinates of a rectilinear grid.

    Basically works the same way as numpy's meshgrid, but returns a sensibly
    shaped single array with the coordinates.
    '''
    R = np.empty(tuple(len(a) for a in axes[::-1]) + (len(axes),))
    nd = len(axes)
    for i, a in enumerate(axes):
        shape = tuple(-1 if j == (nd-1-i) else 1 for j in range(nd))
        R[..., i] = a.reshape(shape)
    return R


class GridData:
    def __init__(self, pos, **attr):
        '''A class for holding data on a 3D grid.

        For the most part, behaves like a dictionary, with multiple attributes
        which can be attached to each point.  At the very least, the `pos`
        attribute is always defined.  Note that all the attributes must have
        the same length!

        Parameters
        ----------
        pos : (Nz, Ny, Nx, Nd) shaped array
            The point coordinates.  Should be a numpy array or castable to one

        Any additional keywords are attached as attributes.  Note that the
        shape of each *must* match that of `pos`, excepting the last dimension.

        Example
        -------
        ~~~
        import numpy as np
        from muvi.geometry import volume

        x = np.linspace(-3, 3, 7)
        y = np.linspace(-4, 4, 9)
        z = np.linspace(-5, 5, 11)
        R = volume.rectilinear_grid(x, y, z)
        r = np.sqrt((R**2).sum(-1))
        V = np.zeros_like(R)
        V[..., 0] = R[..., 1] / r
        V[..., 1] = -R[..., 0] / r

        grid = volume.GridData(
            R = R,
            vel = V,
            r = r
        )

        grid.save('test.vts')
        ~~~
        '''

        self._d = {}
        self['pos'] = pos
        for k, v in attr.items():
            self[k] = v

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, val):
        val = np.asarray(val)
        if key == 'pos':
            if 'pos' in self._d:
                raise ValueError('Position array can not be altered!')
            if val.ndim != 4:
                raise ValueError('position should be a 4D array!')
            self._shape = val.shape[:-1]
        if val.shape[:3] != self._shape:
            raise ValueError(f'Length of attribute ({len(val)}) does not match pos ({self._N})')
        self._d[key] = val

    def keys(self):
        '''Return an iterable of all the attributes'''
        return self._d.keys()

    def __contains__(self, key):
        return key in self._d

    def items(self, force_floats=None):
        '''Return an iterator class with the attribute names and values of the
        points.

        Keywords
        --------
        force_floats : None (default) or numpy data type
            If specified, all floating point attributes are forced to this
            type.

        Returns
        -------
        items : PointItems
            An iterable which behaves like dict.items()
        '''

        return PointItems(self, force_floats=force_floats)

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

        # Do we need to retype?
        if data.dtype != dtype:
            return data.astype(dtype)
        else:
            return data

    def save(self, fn, filetype=None, force_floats='f'):
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

        if filetype == 'vts':
            point_data = []
            scalars = []
            vectors = []

            for k, d in self.items(force_floats=force_floats):
                if k == 'pos':
                    pos = VTKTag('DataArray', d, NumberOfComponents=d.shape[-1])
                elif d.ndim == 3:
                    scalars.append(k)
                    point_data.append(VTKTag('DataArray', d, Name=k))
                elif d.ndim == 4:
                    vectors.append(k)
                    point_data.append(VTKTag('DataArray', d, Name=k, NumberOfComponents=d.shape[-1]))
                else:
                    raise ValueError("can't save attributes which are not vectors or scalars")

            extent = f'0 {self._shape[2]-1} 0 {self._shape[1]-1} 0 {self._shape[0]-1}'

            with VTKWriter(fn, 'StructuredGrid') as f:
                f.write_tag(VTKTag('StructuredGrid', [
                    VTKTag('Piece', [
                        VTKTag('Points', pos),
                        VTKTag('PointData', point_data, Scalars=' '.join(scalars), Vectors=' '.join(vectors))
                    ], Extent=extent),
                ], WholeExtent=extent))
        else:
            raise ValueError('filetype not recognized!')
