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

# This script generates the colormaps used by the viewer; it is not actually
#   part of the library, but rather used to create a piece of it.  Although
#   the colormaps could be generated directly in the library, this would add
#   a dependence on Matplotlib.

import numpy as np
from matplotlib import cm
import base64

colormaps = {
    'viridis': 'Viridis',
    'gray': 'Greyscale',
    'plasma': 'Plasma',
    'inferno': 'Inferno',
    'magma': 'Magma',
    'cividis': 'Cividis',
    'RdBu': 'Red to Blue',
    'PiYG': 'Pink to Lime',
    'BrBG': 'Brown to Teal',
    'PuOr': 'Purple to Orange',
    'RdGy': 'Red to Gray',
    'Spectral': 'Spectral',
    'coolwarm':'Cool to Warm',
    'twilight': 'Twilight',
    'twilight_shifted': 'Shift Twi.',
}

x = np.linspace(0, 1, 256)
with open('../muvi/view/shaders/colormaps.xml', 'wt') as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<ColorMaps>')

    for sn, ln in colormaps.items():
        cmap = (np.clip(cm.get_cmap(sn)(x) * 255, 0, 255))[:, :3].astype('u1')
        print(ln, cmap.shape, cmap.dtype)

        f.write('  <ColorMap ShortName="%s" Name="%s" Length="%s" Format="RGB" Encoding="Base64">\n' % (sn, ln, len(cmap)))
        f.write('    ' + base64.b64encode(cmap.tostring()).decode('UTF-8') + '\n')
        f.write('  </ColorMap>\n')

    f.write('</ColorMaps>')
