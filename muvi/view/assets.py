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

# This module contains classes to contain and display various data types
# In practice, other functions will almost certainly only need to use the
# `load_asset` function, which will return the appropriate type of class

from .. import open_3D_movie, VolumetricMovie
from ..mesh import load_mesh, Mesh
from .ogl import GL, VertexArray, textureFromArray, CUBE_CORNERS, \
    CUBE_TRIANGLES, mag
from .params import ASSET_DEFAULTS, ASSET_PARAMS, VECTOR_OPTIONS, SCALAR_OPTIONS
from .. import geometry
import re
import numpy as np
import os
from copy import copy

#----------------------------------------------------------------------------
# Utility functions
#----------------------------------------------------------------------------


def copyArray(x):
    if isinstance(x, (np.ndarray, list)):
        return x.copy()
    else:
        return x

_SEQUENCE_RE = re.compile('(.*_)frame([0-9]+)')

def load_asset(fn, id=0, parent=None):
    bfn, ext = os.path.splitext(fn)
    ext = ext.lower()

    if ext in ('.vti', '.vts', '.cine'):
        return VolumeAsset(fn, id, parent)
    elif ext in ('.vtp'):
        return GeometryAsset(fn, id, parent)
    elif ext in ('.ply'):
        dir, bfn = os.path.split(fn)
        m = _SEQUENCE_RE.match(bfn)
        if m: # This is a sequence of polygon meshes!
            data = {}
            regex = re.compile(f'{m.group(1)}frame([0-9]+)' + ext)
            for fn in os.listdir(dir):
                m2 = regex.match(fn)
                if m2:
                    data[int(m2.group(1))] = os.path.join(dir, fn)

        return MeshAsset(data, id, parent)
    else:
        raise ValueError('Supported file types are VTI, VTS, VTP, CINE, and PLY')


#----------------------------------------------------------------------------
# DisplayAsset class: an object that can be displayed by the viewer
#----------------------------------------------------------------------------
# All the uniform handling is done here, as this shouldn't vary between data
# types

class DisplayAsset:
    shader = None
    _LABEL = 'Asset'

    def __init__(self, data, id, parent, _loaded_from=None):
        '''Base class for display assets.  Not meant to be used directly, but
        other classes derived from it.

        Parameters
        ----------
        data : str, dict, or a class containing geometric information
            If a string: assumed to be a filename, if a dict, assumed to have
            entries {framenumber:filename}.
        id : int
            The internal id used by the viewer; should be unique.
        parent : muvi.view.View
            The view object which is displaying this asset.
        '''

        if _loaded_from is not None:
            source = _loaded_from
        else:
            source = data

        self.parent = parent
        self.id = id
        self.filename = f'#{id}' # Placeholder!

        if isinstance(source, str):
            # We are loading data from a file, as is usually the case!
            self.dir, self.filename = os.path.split(source)
            self.abspath = os.path.abspath(source)
        elif isinstance(source, dict):
            # A frame sequence...
            fn0 = data[min(source.keys())]
            if isinstance(fn0, str): # A frame sequence with files!
                bfn, ext = os.path.splitext(fn0)
                m = _SEQUENCE_RE.match(bfn)
                if m:
                    fn0 = m.group(1) + "[frame]" + ext
                self.dir, self.filename = os.path.split(fn0)
                self.abspath = os.path.abspath(fn0)

        self.label = f"{self._LABEL}: {self.filename}"

        self.visible = False
        # self.vertexArray = None
        self.validFrame = True
        if not hasattr(self, 'frameRange'):
            self.frameRange = None
        self._frame = None
        self.uniforms = {}
        self.globalUniformNames = set()
        self.globalUniforms = {}

        for key, val in ASSET_DEFAULTS[self.shader].items():
            if key in self.uniforms or key in self.globalUniforms:
                continue
            self[key] = val

        self.globalUniformNames.update(self.parent._shaderDep[self.shader])
        self.globalUniformNames.update(self.parent._rebuildDep.keys())

        if _loaded_from is None:
            self._load(data)

    def _load(self):
        # used by derived classes, here just a placeholder
        pass

    def get_info(self):
        '''Return a list of information items about the asset.'''

        info = [f'Id: {self.id}']

        if hasattr(self, 'abspath'):
            info.append(f'File: {self.abspath}')

        if hasattr(self, 'X0'):
            with np.printoptions(precision=4) as opts:
                info.append(f'Lower Extent: {self.X0}')
                info.append(f'Upper Extent: {self.X1}')

        if getattr(self, 'frameRange', False):
            info.append(f"Frames: {self.frameRange[0]}-{self.frameRange[1]}{' (missing)' if getattr(self, 'missingFrames', False) else ''}")

        return info

    def paramList(self):
        return ASSET_PARAMS[self.shader]

    def __setitem__(self, key, val):
        if key in self.globalUniformNames:
            if self.visible:
                self.parent[key] = val
            self.globalUniforms[key] = val
        elif key == 'frame':
            self.setFrame(val)
        elif key == 'visible':
            if val and (not self.visible) and hasattr(self, 'globalUniforms'):
                self.parent.update(self.globalUniforms)
            self.visible = val
        else:
            self.uniforms[key] = val

    def update(self, d):
        for k, v in d.items():
            self.__setitem__(k, v)

    def allParams(self, prefix=True, hidden=False):
        if prefix is True:
            prefix = f'#{self.id}_'

        d = {prefix+'visible':self.visible}
        d.update({
            prefix+k:copyArray(v)
            for k, v in self.uniforms.items()
            if (hidden or (not k.startswith('_')))
        })

        if hasattr(self, 'globalUniforms'):
            d.update({
                prefix+k:copyArray(v)
                for k, v in self.globalUniforms.items()
                if (hidden or (not k.startswith('_')))
            })

        return d

    def setFrame(self, frame):
        if frame == self._frame or self.frameRange is None:
            pass
        elif frame < self.frameRange[0] or frame > self.frameRange[1]:
            self.validFrame = False
        else:
            self.validFrame = True
            self._frame = frame
            self._set_frame(frame)

    def _set_frame(self, frame):
        # used by derived classes, here just a placeholder
        pass

    def draw(self):
        # used by derived classes, here just a placeholder
        pass

    def delete(self):
        # used by derived classes, here just a placeholder
        pass

#----------------------------------------------------------------------------
# Display class and misc. functions for volumetric data
#----------------------------------------------------------------------------

class VolumeAsset(DisplayAsset):
    shader = 'volume'
    _LABEL = 'Volume'
    _VERT_TYPE = np.dtype([
        ('position', '3float32'),
    ])

    def _load(self, data):
        if isinstance(data, str):
            data = open_3D_movie(data)

        if not isinstance(data, VolumetricMovie):
            raise TypeError('data must be VolumetricMovie')

        self.volume = data
        L = np.array(self.volume.info.get_list('Lx', 'Ly', 'Lz'), dtype='f')
        self.X0 = -0.5 * L
        self.X1 = 0.5 * L
        self.uniforms = dict(
            _vol_L = L,
            _vol_N = np.array(self.volume.info.get_list('Nx', 'Ny', 'Nz'), dtype='f'),
            distortion_correction_factor = self.volume.distortion.var.get('distortion_correction_factor', np.zeros(3, 'f'))
        )

        self.frameRange = (0, len(self.volume) - 1)

        vol = self.volume[0]
        if vol.ndim == 3:
            vol = vol[..., np.newaxis]

        GL.glActiveTexture(GL.GL_TEXTURE1)
        self.volumeTexture = textureFromArray(vol, wrap=GL.GL_CLAMP_TO_EDGE)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        points = np.empty(len(CUBE_CORNERS), self._VERT_TYPE)
        points['position'] = CUBE_CORNERS
        self.vertexArray = VertexArray(points)
        self.vertexArray.attachElements(CUBE_TRIANGLES)

    def draw(self):
        if self.validFrame:
            self.vertexArray.draw()

    def delete(self):
        # Explicitly clean up opengl storage.
        # Trusting the garbage collector to do this isn't a good idea, as it
        #   doesn't work well on app shutdown.
        if hasattr(self, 'volumeTexture'):
            self.volumeTexture.delete()
        if hasattr(self, 'vertexArray'):
            self.vertexArray.delete()

    def _set_frame(self, frame):
        GL.glActiveTexture(GL.GL_TEXTURE1)
        self.volumeTexture.replace(self.volume[frame])


#----------------------------------------------------------------------------
# Display class and misc. functions for mesh data
#----------------------------------------------------------------------------

class MeshAsset(DisplayAsset):
    shader = 'mesh'
    _LABEL = 'Mesh'
    _VERT_TYPE = np.dtype([
        ('position', '3float32'),
        ('normal',   '3float32'),
        ('color',    '4float32')
    ])

    def _load(self, data):

        if isinstance(data, Mesh):
            self.vertexArray, self.X0, self.X1 = self.mesh_to_vertex_array(data)

        elif isinstance(data, dict):
            self.shader = 'mesh'
            self.X0 = None
            self.X1 = None
            self.meshSeq = {}

            X0s = []
            X1s = []

            for key, mesh in data.items():
                if isinstance(mesh, str):
                    mesh = load_mesh(mesh)
                va, X0, X1 = self.mesh_to_vertex_array(mesh)
                self.meshSeq[key] = va
                X0s.append(X0)
                X1s.append(X1)

            self.X0 = np.min(X0s, axis=0)
            self.X1 = np.max(X1s, axis=0)

            keys = self.meshSeq.keys()
            self.frameRange = (min(keys), max(keys))
            self.missingFrames = len(keys) != (self.frameRange[1] - self.frameRange[0] + 1)

    def _set_frame(self, frame):
        self.vertexArray = self.meshSeq.get(frame, None)
        if self.vertexArray is None:
            self.validFrame = False

    def draw(self):
        if self.validFrame:
            self.vertexArray.draw()

    def delete(self):
        # Explicitly clean up opengl storage.
        # Trusting the garbage collector to do this isn't a good idea, as it
        #   doesn't work well on app shutdown.
        if hasattr(self, 'meshSeq'):
            for item in self.meshSeq.values():
                item.delete()
        if hasattr(self, 'vertexArray'):
            self.vertexArray.delete()
            del self.vertexArray

    def mesh_to_vertex_array(self, m):
        N = len(m.points)
        vert = np.empty(N, self._VERT_TYPE)

        if not hasattr(m, 'normals'):
            raise ValueError('Displayed meshes must include point normals!')

        vert['position'] = m.points
        vert['normal'] = m.normals

        if hasattr(m, 'colors'):
            N, channels = m.colors.shape
            m.ensure_linear_colors()
            vert['color'][:, :channels] = m.colors

            if channels == 3:
                vert['color'][:, 3] = 1.0
        else:
            vert['color'] = 1.0

        points = m.points[~(np.isnan(m.points).any(1))]
        X0 = points.min(0)
        X1 = points.max(0)

        va = VertexArray(vert)
        va.attachElements(m.triangles.astype('u4'))

        return va, X0, X1


#----------------------------------------------------------------------------
# Display class and misc. functions for glyph data
#----------------------------------------------------------------------------

_COMPONENT_RE = re.compile('(.*)\.(x|y|z|mag)')
X, Y, Z = np.eye(3)
CONST_VARS = {
    '+x': X,
    '-x': -X,
    '+y': Y,
    '-y': -Y,
    '+z': Z,
    '-z': -Z,
    '1': 1,
}

class GeometryAsset(DisplayAsset):
    def __init__(self, data, id, parent):
        source = None
        is_sequence = False

        if isinstance(data, str):
            source = data
            data = geometry.load_geometry(data)

        if isinstance(data, (geometry.Points, geometry.PointSequence)):
            self.shader = 'points'
            self._LABEL = 'Points'
            self._VERT_TYPE = np.dtype([
                ('position',  '3float32'),
                ('normal',    '3float32'),
                ('size',      'float32'),
                ('color',     'float32'),
                ('glyphType', 'uint32'),
            ])
            self.uniform_vars = {
                'points_normal':  'normal',
                'geometry_size':  'size',
                'geometry_color': 'color',
                'points_glyph':   'glyphType',
            }

            if isinstance(data, geometry.Points):
                self.current_data = data

            else: #PointSequence
                frames = data.keys()
                self.frameRange = (min(frames), max(frames))
                self.frame = min(frames)
                self.current_data = data[self.frame]
                self.is_sequence = True
                self.seq = data

        else:
            raise ValueError('Data type is not displayable by this viewer!')

        super().__init__(data, id, parent, _loaded_from=source)

        self.scalar_options = list(SCALAR_OPTIONS)
        self.vector_options = list(VECTOR_OPTIONS)
        self.scalar_min = None
        self.scalar_max = None

        for key, arr in self.current_data.items():
            minval, maxval = arr.min(), arr.max()
            if self.scalar_min is None or self.scalar_min > minval:
                self.scalar_min = minval
            if self.scalar_max is None or self.scalar_min < maxval:
                self.scalar_max = maxval

            if arr.ndim == 1:
                self.scalar_options.append(key)
            elif arr.ndim == 2 and arr.shape[1] == 3:
                self.vector_options.append(key)
                self.scalar_options += [f'{key}.{c}' for c in ('x', 'y', 'z', 'mag')]


        if 'pos' not in self.current_data:
            raise ValueError('Points object lacking "pos" key... this should never happen!')
        pos = self.current_data['pos']
        self.X0 = pos.min(0)
        self.X1 = pos.max(0)
        self._N = len(pos)

    def get_var(self, key):
        if key in self.current_data:
            return(self.current_data[key])
        elif key in CONST_VARS:
            return CONST_VARS[key]

        m = _COMPONENT_RE.match(key)
        if m:
            key = m.group(1)
            c = m.group(2)
            if key in self.current_data:
                data = self.current_data[key]
                if c == 'x':
                    return data[..., 0]
                elif c == 'y':
                    return data[..., 1]
                elif c == 'z':
                    return data[..., 2]
                elif c == 'mag':
                    return mag(data)

        # If we haven't already returned... this is in invalid key!
        raise ValueError(f'Invalid scalar variable: "{key}"')

    def _build_arrays(self):
        if hasattr(self, 'vertexArray'):
            self.vertexArray.delete()

        pos = self.current_data['pos']
        arr = np.empty(len(pos), self._VERT_TYPE)
        arr['position'] = pos
        for u, var in self.uniform_vars.items():
            val = self.uniforms[u]
            if isinstance(val, str):
                val = self.get_var(val)
            arr[var] = val

        # print(arr[:5])
        self.vertexArray = VertexArray(arr)
        self._N = len(arr)


    def _set_frame(self, frame):
        if frame in self.seq:
            self.current_data = self.seq[int(frame)]
            self._build_arrays()
        else:
            self.validFrame = False


    def draw(self):
        if self.validFrame:
            self.vertexArray.drawArrays(GL.GL_POINTS, 0, self._N)

    def modify_param(self, name, **kwargs):
        if name in self.param_order:
            i = self.param_order[name]
            p = copy(self.params[i])
            for key, val in kwargs.items():
                setattr(p, key, val)
            self.params[i] = p

            if 'default' in kwargs:
                self[name] = kwargs['default']

    def paramList(self):
        # The parameter list needs to be updated based on the available data
        self.params = ASSET_PARAMS[self.shader].copy()
        self.param_order = {p.name:i
            for i, p in enumerate(self.params) if hasattr(p, 'name')}

        color_mod = dict(options=self.scalar_options, default='pos.mag')
        normal_mod = dict(options=self.vector_options)
        if 'vel' in self.vector_options:
            color_mod['default'] = 'vel.mag'
            normal_mod['default'] = 'vel'
        self.modify_param('geometry_color', **color_mod)
        self.modify_param('points_normal', **normal_mod)

        c = self.get_var(color_mod['default'])
        self.modify_param('geometry_c0', min=self.scalar_min,
            max=self.scalar_max, default=c.min())
        self.modify_param('geometry_c1', min=self.scalar_min,
            max=self.scalar_max, default=c.max())

        self.modify_param('geometry_size', options=self.scalar_options)
        L = mag(self.X1 - self.X0)
        self.modify_param('geometry_scale', default = L / np.sqrt(self._N))

        return self.params

    def __setitem__(self, key, val):
        # Handle changes that require us to
        super().__setitem__(key, val)

        if key in self.uniform_vars and self.visible:
            self._build_arrays()

        if key == 'geometry_colormap':
            self.uniforms['colormapOffset'] = self.parent.colormapOffsets[val]
