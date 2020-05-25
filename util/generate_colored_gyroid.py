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

W = 128
N = 128

if len(sys.argv) == 1:
    fn = 'colored_gyroid.vti'
else:
    fn = sys.argv[1]


start = time.time()
scale = 2*np.pi/W

x = np.arange(W).reshape(1, 1, -1) * scale
y = np.arange(W).reshape(1, -1, 1) * scale
z = np.arange(W).reshape(-1, 1, 1) * scale

phi = 2 * z

gyroid_base = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x)
noise = 5*ndimage.gaussian_filter((np.random.rand(W, W, W)-0.5) * 1.0, 3, mode='wrap')

frames = []


for n in range(N):
    base = 255 * np.clip(np.roll(gyroid_base, n, axis=0) + noise - 0.5, 0, 1)
    f = np.empty((W, W, W, 2), dtype='u1')
    f[..., 0] = base * abs(np.sin(phi))
    f[..., 1] = base * abs(np.cos(phi))
    frames.append(f)

el = time.time() - start

print('Generate volume: %.1f ms' % (el * 1000))
print('Fraction empty: %.1f%%' % ((frames[0]==0).mean()*100))


info = {
    'Lx': 100,
    'Ly': 100,
    'Lz': 100,
    'units': 'mm',
}

start = time.time()
vol = VolumetricMovie(frames, **info)
vol.save(fn)
el = time.time() - start

print(vol.info)

fs = os.path.getsize(fn)
rs = N * frames[0].nbytes
print('Save compressed file: %.1f ms (%.2f raw GB/s)' % (el*1000, rs/1E9/el))
print('Final size: %d MB (compression ratio=%.2f)' % (fs/1E6, rs/fs))
