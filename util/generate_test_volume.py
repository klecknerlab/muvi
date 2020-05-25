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

import sys
from muvi import VolumetricMovie
import numpy as np
import time
from scipy import ndimage
import os
from matplotlib import cm
#
# W = 128
# N = 128
#
# if len(sys.argv) == 1:
#     fn = 'colored_gyroid.vti'
# else:
#     fn = sys.argv[1]
fn = 'test.vti'

# start = time.time()
# scale = 2*np.pi/W

Nx = 128
Ny = 192
Nz = 256


x = np.arange(Nx).reshape(1, 1, -1) / Nx
y = np.arange(Ny).reshape(1, -1, 1) / Ny
z = np.arange(Nz).reshape(-1, 1, 1) / Nz

gyroid_base = np.sin(2*np.pi*x)*np.cos(2*np.pi*y) + np.sin(2*np.pi*y)*np.cos(2*np.pi*z) + np.sin(2*np.pi*z)*np.cos(2*np.pi*x)
gyroid = np.clip(gyroid_base - 0.5, 0, 1)[..., np.newaxis]
frame = (cm.get_cmap('hsv')(z)[..., :3] * gyroid * 255).astype('u1')

print('Volume shape:', frame.shape)

vol = VolumetricMovie([frame])

for key, val in vol.info.items():
    print('%15s: %s' % (key, repr(val)))

vol.save(fn)
