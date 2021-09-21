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

'''
An example which generates a volume with a distorted cube of points
'''

import sys
from muvi import VolumetricMovie, VolumeProperties
from muvi.distortion import get_distortion_model
import numpy as np
import time
from scipy import ndimage
import os
from scipy.spatial.transform import Rotation

W = 128
N = 128

output_dir = "tracking_example"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

info = VolumeProperties(
    Lx = 70,
    Ly = 90,
    Lz = 100,
    Nx = 64,
    Ny = 64,
    Nz = 128,
    dx = 75,
    dz = 100
)

indices = np.mgrid[:info['Nx'], :info['Ny'], :info['Nz']].T # Tranposing changes shape to [Nz, Ny, Nx, 3]
distortion = get_distortion_model(info)
X = distortion.convert(indices, 'index-xyz', 'physical')

def gaussians(X0, X, w=1):
    data = np.zeros(X.shape[:-1], 'd')
    for x0 in X0:
        data += np.exp(-((X - x0)**2).sum(-1) / w**2)
    return data

cube = 30 * ((np.arange(8).reshape(-1, 1) // 2**np.arange(3)) % 2 - 0.5).astype('f')

frame = np.clip(gaussians(cube, X, 2) * 200, 0, 255).astype('u1')

VolumetricMovie([frame], info).save(os.path.join(output_dir, "tracking.vti"))

sys.exit()

if len(sys.argv) == 1:
    fn = 'gyroid.vti'
else:
    fn = sys.argv[1]


start = time.time()
scale = 2*np.pi/W

x = np.arange(W).reshape(1, 1, -1) * scale
y = np.arange(W).reshape(1, -1, 1) * scale
z = np.arange(W).reshape(-1, 1, 1) * scale

gyroid_base = np.sin(x)*np.cos(y) + np.sin(y)*np.cos(z) + np.sin(z)*np.cos(x)
noise = 5*ndimage.gaussian_filter((np.random.rand(W, W, W)-0.5) * 1.0, 3, mode='wrap')

np_frames = [
    (255 * np.clip(np.roll(gyroid_base, n, axis=0) + noise - 0.5, 0, 1)).astype('u1').reshape(W, W, W, 1) for n in range(N)
]

el = time.time() - start

print('Generate volume: %.1f ms' % (el * 1000))
print('Fraction empty: %.1f%%' % ((np_frames[0]==0).mean()*100))


info = {
    'Lx': 100,
    'Ly': 100,
    'Lz': 100,
    'units': 'mm',
}

start = time.time()
VolumetricMovie(np_frames, **info).save(fn)
el = time.time() - start

fs = os.path.getsize(fn)
rs = N * np_frames[0].nbytes
print('Save compressed file: %.1f ms (%.2f raw GB/s)' % (el*1000, rs/1E9/el))
print('Final size: %d MB (compression ratio=%.2f)' % (fs/1E6, rs/fs))
