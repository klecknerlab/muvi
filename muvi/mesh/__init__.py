#!/usr/bin/python3
#
# Copyright 2021 Dustin Kleckner
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

from .vectors import norm, mag, mag1, dot, dot1, cross, generate_basis, \
    enforce_shape
from .mesh import Mesh, load_mesh
import numpy as np
import os
from matplotlib import cm

'''
This module contains basic operations for generating meshes of glyphs, intended
to be superimposed on 3D volumes.
'''


CACHED_GLYPHS = {}

GLYPH_DIR = os.path.join(os.path.split(__file__)[0], 'glyphs')

def get_glyph(glyph):
    if glyph not in CACHED_GLYPHS:
        CACHED_GLYPHS[glyph] = load_mesh(os.path.join(GLYPH_DIR, glyph + '.ply'))
    return CACHED_GLYPHS[glyph]


def generate_glyphs(X, glyph="sphere", a=1, color=None, N=np.array([1, 0, 0], dtype='f'), B=None, cmap=None, clim=None, keep_alpha=False):
    '''Generate a polygon mesh composed of glyphs at different points in space,
    possibly with orientation and scaling.

    Parameters
    ----------
    X : (N, 3) shaped array
        The locations of the points, where N is the number of points


    Keywords
    --------
    glyph : str or Mesh (default: "sphere")
        The glyph to use.  The x-axis of the glyph will be aligned with N.
        Valid string options are:
            - "sphere": A icosahedral approximation to a sphere
            - "sphere[2/3/4]": Subdivided "sphere" with 80/320/1280 triangles
            - "arrow": An arrow
            - "arrow2": An arrow with more points
            - "cylinder" A cylinder; diameter = 1/5 of radius.
            - "cylinder2": A cylinder with more points.
            - "tick": An arrowhead
            - "tick2": An arrowhead with more points
    a : float or (N) shaped array (default: 1)
        The size of the glyphs (e.g. the diameter of a sphere or the length
        of an arrow)
    color : None, str, or (N), (N, 3), or (N, 4) shaped array (default: None)
        The color of the glyph.  If not specified, derived from the glyph mesh,
        otherwise can be specified from matplotlib color spec [string],
        directly [(N, 3-4) shaped array], or from a colormap [(N) shaped array].
    N : (N, 3) or (3) shaped array (default: [1, 0, 0])
        The orientation of the glyph (corresponds to the x-axis of the glyph)
    B : None, (N, 3) or (3) shaped array (default: None)
        The orientation of the glyph (correspons to the y-axis of the glyph;
        will be used to generate an orthonormal basis).  If not specified,
        created automatically to be perpendicular to N.  Note: all the default
        glyphs are cylindrically symmetric, in which case this has no meaningful
        effect!
    cmap : str (default: matplotlib default colormap, 'viridis')
    clim : tuple of two floats (default: determined from limits of `color`)
        Used to generate a colormap for the glyphs
    keep_alpha : bool (default: False)
        If False, drop the alpha channel from the color (if present).
    '''

    # Axes: (glyph, vert, xyz, [xyz])

    if isinstance(glyph, str):
        glyph = get_glyph(glyph)

    T = generate_basis(N, B).reshape(-1, 1, 3, 3)

    X = enforce_shape(X, (-1, 3))

    n_glyphs = len(X)
    n_points = len(glyph.points)

    a = np.asarray(a)
    if not a.shape:
        a = np.full(n_glyphs, a)

    points = X.reshape(-1, 1, 3) + ((a.reshape(n_glyphs, 1, -1) * glyph.points).reshape(n_glyphs, n_points, 3, 1) * T).sum(-2)

    normals = getattr(glyph, 'normals', None)

    if normals is not None:
        normals = (glyph.normals.reshape(1, n_points, 3, 1) * T).sum(-2)
    else:
        normals = None

    tris = glyph.triangles.reshape(-1, 1, 3) + n_points * np.arange(n_glyphs).reshape(-1, 1)

    output = Mesh(points.reshape(-1, 3), tris.reshape(-1, 3), normals.reshape(-1, 3))

    if color is not None:
        # color = enforce_shape(color, (len(points),))
        if clim is None:
            clim = (color.min(), color.max())
        rgba = cm.get_cmap(cmap)((color - clim[0]) / (clim[1] - clim[0]))
        output.colors = np.tile(rgba[:, np.newaxis, :3], (1, n_points, 1)).reshape(-1, 3)
    else:
        colors = getattr(glyph, 'colors', None)
        if colors is not None:
            output.colors = np.tile(glyph.colors, (n_glyphs, 1))

    return output
