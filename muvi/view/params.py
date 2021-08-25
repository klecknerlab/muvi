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

import os
from collections import OrderedDict
import base64
import glob
import xml.etree.ElementTree as ET
import re
import numpy as np
import warnings

'''
This file contains information about all the parameters used for the view
settings, broken out into a separate module for clarity.

This module also handles finding the shader and colormaps files, and
determining their names, etc.  (which are critical to setting up the
parameters, as they are options for them!)
'''

# ----------------------------------------------------------
# Paths for finding critical files
# ----------------------------------------------------------

_module_path = os.path.split(os.path.abspath(__file__))[0]
SHADER_PATH = os.path.join(_module_path, 'shaders')


# ----------------------------------------------------------
# Load the colormaps from an XML file
# ----------------------------------------------------------
# Note: these are generated by "generate_colormaps" in the shaders directory;
#  if you want to add new colormaps, do it there!

COLORMAPS = OrderedDict()
_colormap_names = OrderedDict()
_color_remaps = OrderedDict()

class ColorMap:
    def __init__(self, xml):
        self.short_name = xml.get('ShortName', None)
        self.name = xml.get('Name', self.short_name)
        self.length = int(xml.get('Length'))
        if self.length != 256:
            raise ValueError('Colormaps should have length 256! (found %d in "%s")' % (self.length, self.shortname))
        self.data = base64.b64decode(xml.text)
        if len(self.data) != self.length * 3:
            raise ValueError('Colormap "%s" has invalid length' % self.short_name)


for cm in ET.parse(os.path.join(SHADER_PATH, 'colormaps.xml')).getroot():
    cm = ColorMap(cm)
    COLORMAPS[cm.short_name] = cm
    _colormap_names[cm.short_name] = cm.name

for i1, c1 in enumerate("rgb"):
    for i2, c2 in enumerate("rgb"):
        for i3, c3 in enumerate("rgb"):
            c = c1 + c2 + c3 + "a"
            _color_remaps[c] = str((i1+1)*100 + (i2+1)*10 + (i3+1))


# -------------------------------------------------------
# Get a list of Shaders
# -------------------------------------------------------

SUBSHADER_TYPES = ('cloud_color', )
SUBSHADER_NAMES = {}
SUBSHADER_SOURCE = {}

def refresh_shaders():
    '''Load shaders from the shader directory.

    Normally not called by the user, but can be used to udpate sources,
    if they are being edited while a program is being run.
    '''

    for subshader in SUBSHADER_TYPES:
        ds = {}
        dn = {}

        for fn in sorted(glob.glob(os.path.join(SHADER_PATH, subshader + "_*.glsl"))):
            short_name = re.match('.*' + subshader + '_(.*).glsl', fn).group(1)

            with open(fn) as f:
                code = f.read()

            m = re.match(r'^\s*//\s*NAME:\s*(.*)\s*$', code, flags=re.MULTILINE)
            long_name = m.group(1) if m else short_name
            ds[short_name] = code
            dn[short_name] = long_name

        SUBSHADER_NAMES[subshader] = dn
        SUBSHADER_SOURCE[subshader] = ds

        # setattr(self, subshader + "_source", ds)
        # setattr(self, subshader + "_names", dn)

refresh_shaders()

#--------------------------------------------------------
# List of Parameters for Display
#--------------------------------------------------------
# Previously this data was stored in the View class, but this is a cleaner
#   solution because we shouldn't need to create an instance to get the
#   parameter names, limits, etc.  (which are used to build GUIs)

PARAMS = OrderedDict()
PARAMS_WITH_VOLUME_RANGES = {}
PARAM_CATAGORIES = OrderedDict()

class ViewParam:
    def __init__(self, name, display_name, cat, vcat, default, min=None,
            max=None, step=None, logstep=None, options=None, param_type=None,
            tooltip=None, max_from_vol=None):
        self.name = name
        self.display_name = display_name
        self.cat = cat
        self.vcat = vcat
        self.default = default

        if min is not None:
            self.min = min
        if max is not None:
            self.max = max
        if step is not None:
            self.step = step
        if logstep is not None:
            self.logstep = logstep
        if options is not None:
            self.options = options
            param_type = 'options'
        if tooltip is not None:
            self.tooltip = tooltip
        if max_from_vol is not None:
            self.max_from_vol = max_from_vol
            PARAMS_WITH_VOLUME_RANGES[name] = self

        PARAMS[name] = self

        if param_type is None:
            self.type = type(self.default)
        else:
            self.type = param_type

    # type = property(lambda self: type(self.default))

PARAM_CATAGORIES['Playback'] = [
    ViewParam('frame', 'Frame', 'playback', 'view', 0, 0, 99, step=1,
        param_type='playback', max_from_vol='Nt'),
]

MAX_CHANNELS = 3
_default_colormaps = ['inferno', 'viridis', 'cividis']

PARAM_CATAGORIES['Render'] = [
    ViewParam('density', 'Density', 'render', 'uniform', 0.5, min=2**-10,
        max=2, logstep=2**(1/2),
        tooltip='The density of the cloud rendering.  If glow=1, this is the opacity of a single voxel with maximum vaue.'),
    ViewParam('glow', 'Glow', 'render', 'uniform', 1.0, min=1, max=128,
        logstep=2,
        tooltip='Increasing glow makes the cloud more transparent while proportionally increasing the brightness, giving the appearance of a glowing cloud.'),
    OrderedDict()
]

for n in range(1, MAX_CHANNELS + 1):
    color = np.zeros(3, dtype='f')
    color[n-1] = 1

    PARAM_CATAGORIES['Render'][-1][f'Ch. {n}'] = [
         ViewParam(f'exposure{n}', 'Exposure', 'render', 'uniform', 0.0, min=0,
            max=8, step=0.5,
            tooltip='Adjust the brightness of the raw data.  Specified in stops (powers of 2).'),

        'Cloud',
        ViewParam(f'cloud{n}_active', f'Enabled', 'render', 'shader', (n == 1),
            tooltip='Enable cloud rendering for this color channel.'),
        ViewParam(f'colormap{n}', 'Colormap', 'render', 'view',
            _default_colormaps[n-1], options=_colormap_names,
            tooltip='Select the color mapping used to render this channel.'),

        'Isourface',
        ViewParam(f'iso{n}_active', f'Enabled', 'isosurface', 'shader',
            False,
            tooltip='Enable isosurface rendering for this color channel.'),
        ViewParam(f'iso{n}_level', 'Level', 'isosurface', 'uniform', 0.5,
            min=0.0, max=1.0, step=0.1,
            tooltip='The level to display the isosurface at: 0 = minimum value, 1 = maximum.  Note that this is affected by exposure!'),
        ViewParam(f'iso{n}_color', 'Color', 'isosurface', 'uniform', color,
            param_type='color',
            tooltip='The color of the isosurface.'),
        ViewParam(f'iso{n}_opacity', 'Opacity', 'isosurface', 'uniform', 0.3,
            min=0.0, max=1.0, step=0.1,
            tooltip='The opacity of the isosurface.')
    ]

PARAM_CATAGORIES['View'] = [
    ViewParam('R', 'Rotation', 'view', 'view', np.eye(3, dtype='f'),
        param_type='rot'),
    ViewParam('scale', 'Scale', 'view', 'view', 1.0, logstep=1.25, param_type='hidden'),
    ViewParam('fov', 'FOV (degrees)', 'view', 'view', 30.0, min=0.0, max=120.0, step=10.0,
        tooltip='The field of view of the display, measured in degrees.  Setting this to 0 gives an orthographic display.'),
    ViewParam('framerate', 'Playback Rate', 'playback', 'view', 30, 1, 120, 10,
        tooltip='The playback rate in volumes/second.'),
    ViewParam('background_color', 'Background Color', 'view', 'view', np.array([0, 0, 0, 1], dtype='f'), param_type='color',
        tooltip='The background color of the display.'),
    ViewParam('X0', 'Displayed Volume Lower Limit', 'view', 'uniform', np.zeros(3, dtype='f'),
        max_from_vol=('Lx', 'Ly', 'Lz'),
        tooltip='Lower limit of displayed volume (in physical units)'),
    ViewParam('X1', 'Displayed Volume Upper Limit', 'view', 'uniform', np.ones(3, dtype='f') * 100,
        max_from_vol=('Lx', 'Ly', 'Lz'),
        tooltip='Upper limit of displayed volume (in physical units)'),
    ViewParam('center', 'Display Center', 'view', 'view', np.ones(3, dtype='f') * 128,
        max_from_vol=('Lx', 'Ly', 'Lz'), param_type='hidden'),
]

PARAM_CATAGORIES['Adv.'] = [
    ViewParam('step_size', 'Render Step', 'advanced', 'uniform', 1.0,
        min=0.125, max=2, logstep=2**(1/2),
        tooltip='The step size used in the internal rendering algorithm.  Increasing this will improve the quality of the display, but slows down the rendering engine proportionally.'),
    ViewParam('perspective_xfact', 'Persp. X Coeff.', 'advanced', 'uniform',
        0.0, min=-1, max=1, step=0.05,
        tooltip='The coefficient for perspective correction in the x direction (=Lx/dx).  Should only be non-zero if the scanner is displaced in the x direction.'),
    ViewParam('perspective_yfact', 'Persp. Y Coeff.', 'advanced', 'uniform', 0.0, min=-1, max=1, step=0.05,
        tooltip='The coefficient for perspective correction in the y direction (=Ly/dy).  Should only be non-zero if the scanner is displaced in the y direction.'),
    ViewParam('perspective_zfact', 'Persp. Z Coeff.', 'advanced', 'uniform', 0.0, min=-1, max=1, step=0.05,
        tooltip='The coefficient for perspective correction in the z direction (=Lz/dz).'),
    ViewParam('color_remap', 'Color Remap', 'advanced', 'shader', 'rgba', options=_color_remaps,
        tooltip='Reshuffles the color channels in the corresponding order.  Can be used, for example, to have multiple isosurfaces.'),
    ViewParam('cloud_color', 'Cloud Shader', 'advanced', 'shader', 'colormap', options=SUBSHADER_NAMES['cloud_color'],
        tooltip='The cloud color sub-shader to use for the display.  Normally not changed.'),
    ViewParam('gamma2', 'Gamma 2', 'advanced', 'shader', False,
        tooltip='If checked, treat the raw data as if it has gamma 2.  Normally this is automatically determined from the volume itself.')
]

def range_from_volume(vol):
    ranges = {}
    for name, param in PARAMS_WITH_VOLUME_RANGES.items():
        try:
            if param.type == np.ndarray:
                lower = np.zeros_like(param.default)
                upper = np.array([vol.info[p] for p in param.max_from_vol], dtype=param.default.dtype)
            else:
                t = type(param.default)
                lower = t(0)
                upper = t(vol.info[param.max_from_vol])
        except:
            warnings.warn(f"Could not get range of parameter '{name}' from volume info.")
        else:
            ranges[name] = (lower, upper)

    return ranges
