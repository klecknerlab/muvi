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
A simple example which generates some arrows moving in a 2-vortex flow field.

The resulting frames can be viewed as a video of moving glyphs.
'''

from muvi import mesh
import numpy as np
import os

def mag(X):
    return np.sqrt((X**2).sum(-1))

def vortex(X, X0=np.zeros(3, 'd')):
    X = np.asarray(X - X0).reshape(-1, 3)
    s2 = (X[:, :2]**2).sum(-1)
    V = np.zeros_like(X)
    V[:, 0] = X[:, 1] / s2
    V[:, 1] = -X[:, 0] / s2
    return V

output_dir = 'vortex_frames'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def vel(X):
    return vortex(X, (0.5, 0, 0)) - vortex(X, (-0.5, 0, 0))


X = np.random.randn(150, 3)
for n in range(128):
    V = vel(X)
    v = np.clip(mag(V), 0, 5)
    mesh.generate_glyphs(50+X*25, 'arrow', a=8, color=np.sqrt(v), N=V).save(os.path.join(output_dir, f'vortex_frame{n:08d}.ply'))

    # Poor man's integrator!
    for n in range(10):
        X += vel(X) * 0.002
