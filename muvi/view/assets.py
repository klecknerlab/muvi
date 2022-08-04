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
from .ogl import GL, VertexArray, textureFromArray, CUBE_CORNERS, CUBE_TRIANGLES
from .params import ASSET_DEFAULTS, ASSET_PARAMS
import re
import numpy as np
import os

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
        return GlyphAsset(fn, id, parent)
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
    _LABEL = 'Asset:'

    def __init__(self, data, id, parent):
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

        self.parent = parent
        self.id = id
        self.filename = f'#{id}' # Placeholder!
        self._sourcedata = data

        if isinstance(data, str):
            # We are loading data from a file, as is usually the case!
            self.dir, self.filename = os.path.split(data)
            self.abspath = os.path.abspath(data)
        elif isinstance(data, dict):
            # A frame sequence...
            fn0 = data[min(data.keys())]
            if isinstance(fn0, str): # A frame sequence with files!
                bfn, ext = os.path.splitext(fn0)
                m = _SEQUENCE_RE.match(bfn)
                if m:
                    fn0 = m.group(1) + "[frame]" + ext
                self.dir, self.filename = os.path.split(fn0)
                self.abspath = os.path.abspath(fn0)

        self.label = self._LABEL + " " + self.filename

        self.visible = False
        # self.vertexArray = None
        self.validFrame = True
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

        self._load(data)

    def reload(self):
        self._load(self._sourcedata)

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
        if frame < self.frameRange[0] or frame > self.frameRange[1]:
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




class GlyphAsset(DisplayAsset):
    shader = 'glyph'
    _LABEL = 'Glyph:'
    _VERT_TYPE = np.dtype([
        ('position', '3float32'),
        ('X',        '3float32'),
        ('Y',        '3float32'),
        ('scale',    '3float32'),
        ('color',    'float32'),
        ('glyphType','uint32'),
    ])



#----------------------------------------------------------------------------
# Display class and misc. functions for volumetric data
#----------------------------------------------------------------------------



class VolumeAsset(DisplayAsset):
    shader = 'volume'
    _LABEL = 'Volume:'
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
    _LABEL = 'Surface:'
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
