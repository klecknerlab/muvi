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

from .points import Points, PointSequence, PointsFromFile, PointSequenceFromFile, PointSequenceFromFunction
from ..filetypes.vtk import VTKReader
import numpy as np
import os
import re

def load_geometry(fn):
    bfn, ext = os.path.splitext(fn)
    ext = ext.lower()

    if ext == '.vtp':  
        vtk = VTKReader(fn)
        if vtk.main.tag == 'PolyData':
            if 'TimeValues' in vtk.main.attrib:
                return PointSequenceFromFile(vtk)
            else:
                return PointsFromFile(vtk)
        else:
            raise ValueError(f"File '{fn}' does not contain geometry of a recognizable type")
    elif ext == '.py':
        with open(fn, 'rt') as f:
            preamble = f.readline()
            m = re.match('\s*#\s*MUVI\s+GEOMETRY:(.+)', preamble)
            if not m:
                raise RuntimeError(f'Failed to load dynamic geometry from "{fn}"; first line should be "# MUVI GEOMETRY: [geometry type]"')
            else:
                geometry_type = m.group(1).strip()

            _globals = {}
            _locals = {'__file__':os.path.abspath(fn)}
            exec(f.read(), _globals, _locals)

            display = _locals.get('_display', None)
            metadata = _locals.get('_metadata', None)

            if geometry_type == 'PointSequence':
                if '_get' not in _locals:
                    raise ValueError(f'Dynamic geometry scripts must define a "_get" function, but none found in "{fn}"')
                if '_valid_frames' not in _locals:
                    raise ValueError(f'Dynamic geometry scripts must define a "_valid_frames" object, but none found in "{fn}"')
                
                return PointSequenceFromFunction(_locals['_get'], valid_frames=_locals['_valid_frames'], display=display, metadata=metadata)
            else:
                raise ValueError(f'Unknown geometry type: "{geometry_type}"')

    else:
        raise ValueError(f'Extension "{ext}" is not supported by the MUVI Geometry module')

def from_pandas(dat, frame='frame', pos=('xc', 'yc', 'zc'), fields={'voxel_index':('x', 'y', 'z')}, display={}, metadata={}):
    '''Convert Pandas data array to a Points or PointSequence

    Parameters
    ----------
    dat : Pandas DataFrame object

    Keywords
    --------
    frame : string (default: 'frame')
        The column to use as frame number.  If `None`, then a Points object is
        returned instead of PointSequence
    pos : list (default: ('x', 'y', 'z'))
        The columns to use as the point position
    fields : dictionary (default: fields={'voxel_index':('x', 'y', 'z')})
        A dictionary of the fields to include in the output.  The key is the
        name of the variable in the points file, and the value is either a
        single string (corresponding to a column) or a list of strings (in
        which case it is encoded as a vector).


    Returns
    -------
    points : Points or PointSequence object
    '''

    f = {col:col for col in dat.columns}

    pos = list(pos)
    for col in pos:
        f.pop(col)

    if frame is not None:
        f.pop(frame)
        frames = np.unique(dat['frame'])

    for field, cols in fields.items():
        if hasattr(col, '__iter__'):
            for col in cols:
                f.pop(col)
            f[field] = list(cols)
        else:
            f[field] = cols

    if frame is not None:
        seq = {}
        for frame in frames:
            dat_f = dat[dat['frame'] == frame]
            d = {f:dat_f[c].to_numpy() for f, c in f.items()}
            seq[int(frame)] = Points(dat_f[pos].to_numpy(), **d)
        return PointSequence(seq, display=display, metadata=metadata)
    else:
        d = {f:dat[c].to_numpy() for f, c in f.items()}
        return Points(dat[pos].to_numpy(), display=display, metadata=metadata, **d)
