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
            c = c1 + c2 + c3
            _color_remaps[c] = str((i1+1)*100 + (i2+1)*10 + (i3+1))


# -------------------------------------------------------
# Get a list of Shaders
# -------------------------------------------------------

SUBSHADER_TYPES = ('cloud_shade', 'surface_shade', 'perspective_model')
SUBSHADER_NAMES = {}
SUBSHADER_SOURCE = {}

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

# def refresh_shaders():
#     '''Load shaders from the shader directory.
#
#     Normally not called by the user, but can be used to udpate sources,
#     if they are being edited while a program is being run.
#     '''
#
#     for subshader in SUBSHADER_TYPES:
#         ds = {}
#         dn = {}
#
#         for fn in sorted(glob.glob(os.path.join(SHADER_PATH, subshader + "_*.glsl"))):
#             short_name = re.match('.*' + subshader + '_(.*).glsl', fn).group(1)
#
#             with open(fn) as f:
#                 code = f.read()
#
#             m = re.match(r'^\s*//\s*NAME:\s*(.*)\s*$', code, flags=re.MULTILINE)
#             long_name = m.group(1) if m else short_name
#             ds[short_name] = code
#             dn[short_name] = long_name
#
#         SUBSHADER_NAMES[subshader] = dn
#         SUBSHADER_SOURCE[subshader] = ds
#
#         # setattr(self, subshader + "_source", ds)
#         # setattr(self, subshader + "_names", dn)
#
# refresh_shaders()

#--------------------------------------------------------
# List of Parameters for Display
#--------------------------------------------------------
# Previously this data was stored in the View class, but this is a cleaner
#   solution because we shouldn't need to create an instance to get the
#   parameter names, limits, etc.  (which are used to build GUIs)

PARAMS = OrderedDict()
PARAMS_WITH_VOLUME_RANGES = {}
PARAM_CATEGORIES = OrderedDict()
ASSET_PARAMS = {}
ALL_ASSET_PARAMS = {}
ASSET_DEFAULTS = {}

_ASSET = None

class ViewParam:
    def __init__(self, name, display_name, default, min=None,
            max=None, step=None, logstep=None, options=None, param_type=None,
            tooltip=None, max_from_vol=None, range_update=None, extend=None):
        self.name = name
        self.display_name = display_name
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
        if range_update is not None:
            self.range_update = range_update
        if extend is not None:
            self.extend = extend

        if param_type is None:
            self.type = type(self.default)
        else:
            self.type = param_type

        PARAMS[name] = self

        if _ASSET is not None:
            if _ASSET not in ASSET_DEFAULTS:
                ASSET_DEFAULTS[_ASSET] = {name:default}
            else:
                ASSET_DEFAULTS[_ASSET][name] = default

            ALL_ASSET_PARAMS[name] = self

    # type = property(lambda self: type(self.default))

class ViewAction:
    def __init__(self, action, display_name, *args, **kw):
        self.action = action
        self.display_name = display_name
        self.args = args
        self.kw = kw

PARAM_CATEGORIES['Playback'] = [
    ViewParam('frame', 'Frame', 0, 0, 0, step=1, param_type='playback',
        max_from_vol='Nt'),
]

zero = np.zeros(3, dtype='f')
one = np.ones(3, dtype='f')

zero4 = np.zeros(4, dtype='f')
one4 = np.ones(4, dtype='f')

PARAM_CATEGORIES['Keyframe'] = [
    ViewParam("_frames", "Output Frames", -1, -1, 100, step=10,
        tooltip='''The number of frames assigned to this keyframe.
    -1 indicates automatic determination:
        - 1 if this is the first frame
        - number of elapsed frames if the frame # changes
        - 15 for any other situation.
    0 can be used to change parameters without outputing a frame.'''),
    ViewParam("_interp", "Interpolation", "smooth", options = {
        "smooth": "Smooth Step",
        "smoother": "Smoother Step",
        "linear": "Linear",
    }, tooltip = "The method used to interpolate all vectors for this keyframe."),
    ViewParam("_camera", "Camera Motion", "object", options = {
        "object": "Maintain distance",
        "direct": "Direct"
    }, tooltip = """The type of camera motion interpolation used.
        - Maintain distance: smoothly interpolate the rotation and distance to object.
            Gives the appearance of a rotating object.
        - Direct: smoothly interpolate the camera position and look at location.
            Gives the appearance of a camera flying around the object."""),
    ViewParam("_spin", "Spin", "none", options = {
        "none": "No spin",
        "+x": "+x",
        "-x": "-x",
        "+y": "+y",
        "-y": "-y",
        "+z": "+z",
        "-z": "-z"
    }, tooltip="If specified, the axis around which to spin the volume following this keyframe (right hand rule direction)"),
    ViewParam("visible", "Active", False, param_type="hidden"),
    ViewParam("_label", "Label", "label", param_type="hidden"),
]

PARAM_CATEGORIES['Asset List'] = [
    ViewParam('_autoupdate_limits', 'Auto-update limits/view', True,
    tooltip="If enabled, the display limits are automatically adjusted when the visible items are toggled."),
]

PARAM_CATEGORIES['Limits'] = [
    # ViewAction('resetView', 'Recenter View', None, None),
    ViewParam('disp_X0', 'Displayed Volume Lower Limit', -50*one, min=-50*one,
        max=50*one, range_update='data_limits+', extend=1,
        tooltip='Lower limit of displayed volume (in physical units)'),
    ViewParam('disp_X1', 'Displayed Volume Upper Limit', 50*one, min=-50*one,
        max=50*one, range_update='data_limits+', extend=1,
        tooltip='Upper limit of displayed volume (in physical units)'),
    ViewParam('mesh_clip', 'Clip geometry to Limits', True,
        tooltip='If True, clip non-volumetric data to volumetric data limits.'),
]

PARAM_CATEGORIES['View'] = [
    # ViewAction('resetView', 'Recenter View', None, None),
    ViewParam('fov', 'FOV (degrees)', 30.0, min=0.0, max=120.0, step=10.0,
        tooltip='The field of view of the display, measured in degrees.  Setting this to 0 gives an orthographic display.'),
    ViewParam('framerate', 'Playback Rate', 30, 1, 120, 10,
        tooltip='The playback rate in volumes/second.'),
    ViewParam('camera_pos', 'Camera Position', np.array([50, 50, 500], dtype='f'),
        min=-500*one, max=500*one, range_update='camera_limits',
        tooltip='The position of the camera in volume physical units.'),
    ViewParam('look_at', 'Camera Target', 50*one, min=-100*one, max=100*one,
        range_update='data_limits',
        tooltip='The position which the camera is looking at.'),
    ViewParam('up', 'Camera Up Direction', np.array([0, 1, 0], dtype='f'), min=-one, max=one,
        tooltip='The normal which defines the up direction of the camera.  Length is ignored.'),
]

_BG = 0.1**2.2

PARAM_CATEGORIES['Display'] = [
    ViewParam('background_color', 'Background', one*_BG, param_type='color',
        tooltip='The color of the display background'),
    ViewParam('axis_background_color', 'Axes Background', zero, param_type='color',
        tooltip='The color of the background behind the axes'),
    ViewParam('surface_shade', 'Surface Shade', 'camera', options=SUBSHADER_NAMES['surface_shade'],
        tooltip='The lighting model used to shade surfaces.'),
    ViewParam('surface_brightness', 'Surface Brightness', 1.0, min=0.5, max=4.0, logstep=2**0.25,
        tooltip='Overall brigthness adjustment of surface shading.'),

    'Axis Labels',
    ViewParam('show_axis_labels', 'Show', True,
        tooltip='Show the axes labels.'),
    ViewParam('axis_label_x', 'X Label', 'X',
        tooltip='The label for the x-axis.'),
    ViewParam('axis_label_y', 'Y Label', 'Y',
        tooltip='The label for the x-axis.'),
    ViewParam('axis_label_z', 'Z Label', 'Z',
        tooltip='The label for the x-axis.'),
    ViewParam('axis_label_padding', 'Label Spacing', 3.5, 0.0, 10.0, step=0.5,
        tooltip='Padding between the axis and the label'),
    ViewParam('axis_orient_labels', 'Orient w/ Axis', True,
        tooltip='If true, rotate the axis labels with the axis'),
    ViewParam('axis_label_color', 'Color', one, param_type='color',
        tooltip='The color of the axis lines and labels.'),
    ViewParam('axis_label_size', 'Font Size', 12., 6., 60., step=1.0),
    ViewParam('axis_single_label', 'One Label per Axis', True,
        tooltip='If true, one label per axis is drawn at most.'),
    ViewParam('axis_label_angle', 'Preferred Orientation', 215., 0., 360., step=45,
        tooltip='Prioritize labels drawn at this angle.\n0 = top, 90 = right, 180 = bottom, 270 = left\n(Only used if there is one label per axis.)'),
    ViewParam('axis_angle_exclude', 'Angle Exclude', 10., 0., 90., step=5.0,
        tooltip='Any axis whose angle is less than this with respect to the camera will not have their labels drawn.'),

    'Axis Lines and Ticks',
    ViewParam('show_axis', 'Show', True,
        tooltip='Show the axes.'),
    ViewParam('axis_color', 'Color', one, param_type='color',
        tooltip='The color of the axis lines and labels.'),
    ViewParam('axis_ticks_out', 'Ticks Face Out', False),
    ViewParam('axis_line_width', 'Line Width', 1., 0.5, 10.0, step=1.0),
    ViewParam('axis_major_tick_spacing', 'Major Tick Spacing', 20.0, min=1E-3, max=1E3, logstep=2,
        tooltip='The spacing of major ticks on the axis.'),
    ViewParam('axis_tick_padding', 'Tick Label Spacing', 0.5, 0.0, 10.0, step=0.5,
        tooltip='The spacing between the axis and the tick labels'),
    ViewParam('axis_minor_ticks', 'Minor Ticks', 4, min=1, max=20, step=1,
        tooltip='The number of minor divisions per major tick.  1 = no minor ticks.'),
    ViewParam('axis_major_tick_length_ratio', 'Major Tick Length', 0.15, 0.0, 1.0, step=0.05,
        tooltip='The length of the major ticks relative to the distance between them.'),
    ViewParam('axis_minor_tick_length_ratio', 'Minor Tick Length', 0.6, 0.0, 1.0, step=0.05,
        tooltip='The length of the minor ticks relative to the major ticks.'),
]

MAX_CHANNELS = 3
_default_colormaps = ('inferno', 'viridis', 'cividis')

VECTOR_OPTIONS = ('+x', '+y', '+z', '-x', '-y', '-z')
SCALAR_OPTIONS = ('1', )
GLYPH_TYPES = {0:'Sphere', 1:'Arrow', 2:'Tick', 3:'Cylinder'}

_ASSET = "points"
ASSET_PARAMS['points'] = [
    ViewParam('geometry_color', 'Color', '1', options=SCALAR_OPTIONS,
        tooltip='Parameter used to color glyphs'),
    ViewParam('geometry_colormap', 'Colormap', 'RdBu', options=_colormap_names,
        tooltip='Select the color map used to display glyphs'),
    ViewParam('geometry_c0', 'Min. value', -1.0, -10.0, 10.0, extend=2,
        tooltip='Minimum value used in color scaling'),
    ViewParam('geometry_c1', 'Max. value', +1.0, +10.0, 10.0, extend=2,
        tooltip='Maximum value used in color scaling'),
    ViewParam('geometry_shade_color', 'Shade Color', np.array([1.0, 0.1, 0.1]),
        param_type='color', tooltip='Solid color shading for geometry (use "shade" to enable).'),
    ViewParam('geometry_shade', 'Shade', 0.0, 0.0, 1.0,
        tooltip='If 0, use colormap to color the geometry; if 1, use the shade color.'),
    ViewParam('geometry_normal', 'Direction', '+y', options=VECTOR_OPTIONS,
        tooltip='Orientation vector of the glyphs'),
    ViewParam('geometry_size', 'Size', '1', options=SCALAR_OPTIONS,
        tooltip='Parameter used to size glyphs'),
    ViewParam('geometry_scale', 'Scale factor', 1.0, 1E-3, 1000, logstep=2,
        tooltip='Scale factor for glyphs'),
    ViewParam('points_glyph', 'Glyph', 1, options=GLYPH_TYPES,
        tooltip='Type of glyph to display at each point'),
    ViewParam('points_skip', 'Skip', 1, 1, 20, step=1,
        tooltip='Used to thin points: 1 shows every point, 2 shows every other, and so on'),
]

LINE_TYPES = {0:'Round', 1:'Ribbon', 2:'Thick Ribbon', 3:'Ellipse', 4:'Triangle', 5:'Square'}

_ASSET = "loop"
ASSET_PARAMS['loop'] = [
    ViewParam('geometry_color', 'Color Variable', '1', options=SCALAR_OPTIONS,
        tooltip='Parameter used to color loop'),
    ViewParam('geometry_colormap', 'Colormap', 'RdBu', options=_colormap_names,
        tooltip='Select the color map used to color loop'),
    ViewParam('geometry_c0', 'Min. value', -1.0, -10.0, 10.0, extend=-1,
        tooltip='Minimum value used in color scaling'),
    ViewParam('geometry_c1', 'Max. value', +1.0, +10.0, 10.0, extend=-1,
        tooltip='Maximum value used in color scaling'),
    ViewParam('geometry_shade_color', 'Shade Color', np.array([1.0, 0.1, 0.1]),
        param_type='color', tooltip='Solid color shading for geometry (use "shade" to enable).'),
    ViewParam('geometry_shade', 'Shade', 0.0, 0.0, 1.0,
        tooltip='If 0, use colormap to color the geometry; if 1, use the shade color.'),
    ViewParam('geometry_normal', 'Direction', '+y', options=VECTOR_OPTIONS,
        tooltip='Orientation vector of the loop'),
    ViewParam('geometry_size', 'Thickness', '1', options=SCALAR_OPTIONS,
        tooltip='Parameter used to determine thickness of loop'),
    ViewParam('geometry_scale', 'Scale factor', 1.0, 1E-3, 1000, logstep=2,
        tooltip='Scale factor for loop thickness'),
    ViewParam('loop_glyph', 'Glyph', 0, options=LINE_TYPES,
        tooltip='Type of glyph to display at each point'),
    ViewParam('loop_angle', 'Rotation', 0.0, 0.0, 360.0, 45.0,
        tooltip='Rotation of the cross-section of the loop.  By default, ribbons are oriented with the normal, but this can modify that.')
]

_ASSET = "mesh"
ASSET_PARAMS['mesh'] = [
    ViewParam('mesh_scale', 'Scale of Mesh', 1.0, 1E-3, 1000, logstep=2**0.5,
        tooltip='Scale of the mesh, relative to the volumetric data units.'),
    ViewParam('mesh_offset', 'Mesh Offset', zero, -100*one, 100*one,
        tooltip='The offset of the mesh, relative to the volumetric data units'),
]

_ASSET = "volume"
ASSET_PARAMS['volume'] = [
    ViewParam('vol_density', 'Density', 0.5, min=2**-10, max=2, logstep=2**(1/2),
        tooltip='The density of the cloud rendering.  If glow=1, this is the opacity of a single voxel with maximum vaue.'),
    ViewParam('vol_glow', 'Glow', 1.0, min=1, max=128, logstep=2,
        tooltip='Increasing glow makes the cloud more transparent while proportionally increasing the brightness, giving the appearance of a glowing cloud.'),
    ViewParam('vol_step_size', 'Render Step', 1.0, min=0.125, max=2, logstep=2**(1/2),
        tooltip='The step size used in the internal rendering algorithm.  Decreasing this will improve the quality of the display, but slows down the rendering engine proportionally.'),
    ViewParam('vol_background_color', 'Background', np.zeros(4), param_type='color',
        tooltip='The color of the display background'),
    OrderedDict()
]

for n in range(1, MAX_CHANNELS + 1):
    color = np.zeros(3, dtype='f')
    color[n-1] = 1

    ASSET_PARAMS['volume'][-1][f'Ch. {n}'] = [
         ViewParam(f'vol_exposure{n}', 'Exposure', 0.0, min=0, max=8, step=0.5,
            tooltip='Adjust the brightness of the raw data.  Specified in stops (powers of 2).'),

        'Cloud',
        ViewParam(f'vol_cloud{n}', f'Enabled', (n == 1),
            tooltip='Enable cloud rendering for this color channel.'),
        ViewParam(f'vol_colormap{n}', 'Colormap', _default_colormaps[n-1],
            options=_colormap_names,
            tooltip='Select the color mapping used to render this channel.'),

        'Isosurface',
        ViewParam(f'vol_iso{n}', 'Enabled', False,
            tooltip='Enable isosurface rendering for this color channel.'),
        ViewParam(f'vol_iso{n}_level', 'Level', 0.5, min=0.0, max=1.0, step=0.1,
            tooltip='The level to display the isosurface at: 0 = minimum value, 1 = maximum.  Note that this is affected by exposure!'),
        ViewParam(f'vol_iso{n}_color', 'Color', color, param_type='color',
            tooltip='The color of the isosurface.'),
        ViewParam(f'vol_iso{n}_opacity', 'Opacity', 0.7, min=0.0, max=1.0, step=0.1,
            tooltip='The opacity of the isosurface.')
    ]

ASSET_PARAMS['volume'] += [
    ViewParam('distortion_correction_factor', 'Distortion Coefficients', zero, min=-one, max=+one, step=0.05,
        tooltip='The coefficient for perspective correction along each axis (=Li/di).  Normally only one of x/y is nonzero.'),
    ViewParam('cloud_shade', 'Cloud Shade', 'colormap', options=SUBSHADER_NAMES['cloud_shade'],
        tooltip='The cloud color sub-shader to use for the display.  Normally not changed.'),
    ViewParam('gamma2', 'Gamma 2', False,
        tooltip='If checked, treat the raw data as if it has gamma 2.  Normally this is automatically determined from the volume itself.'),
    ViewParam('color_remap', 'Color Remap', 'rgb', options=_color_remaps,
        tooltip='Reshuffles the color channels in the corresponding order.  Can be used, for example, to have multiple isosurfaces.'),
]

# _ASSET = 'text'
#
# ASSET_PARAMS['text'] += [
#     ViewParam('text_type', 'Type', 'overlay', options={'overlay':'Overlay', 'marker':'Marker'},
#         tooltip='Type of text label, either an overlay (on top of data), or marker (inside data)'),
#     ViewParam('maker_pos', 'Marked Position', zero, min=-100*one, max=100*one, range_update="data_limits",
#         tooltip='Marked location for marker text (ignored for inline data'),
#     # ViewParam('alignment', 'Al)
# ]


_ASSET = None

THEMES = {
    'Light': dict(
        background_color = one*0.9,
        axis_background_color = one,
        axis_color = zero,
        axis_label_color = zero,
        # axis_line_width = 1.5,
    ),
    'Dark': dict(
        background_color = _BG*one,
        axis_background_color = zero,
        axis_color = one,
        axis_label_color = one,
        # axis_line_width = 1.0,
    ),
    'Dark on Light': dict(
        background_color = one,
        axis_background_color = zero,
        axis_color = one,
        axis_label_color = zero,
        # axis_line_width = 1.5,
    ),
}
