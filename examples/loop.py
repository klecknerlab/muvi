#!/usr/bin/python3
#
# Copyright 2023 Dustin Kleckner
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

from muvi import geometry
import os
import numpy as np
import time

odir = 'geometry_examples'
if not os.path.exists(odir):
    os.makedirs(odir)
os.chdir(odir)



N = 500
Nt = 100
ϕ = np.linspace(0, 2*np.pi, N, False)

seq = []

for i in range(Nt):
    θ = 2*np.pi*i/Nt
    aspect = 0.3 * (1.5 + np.cos(θ))

    r = 1 + aspect * np.sin(3*ϕ)

    X = np.zeros((N, 3))
    X[:, 0] = r * np.cos(2*ϕ)
    X[:, 1] = r * np.sin(2*ϕ)
    X[:, 2] = aspect * np.cos(3*ϕ)
    thickness = 0.05 * (1.2 + np.sin(19*ϕ + 2*θ))
    color = np.sin(9*ϕ -θ)

    seq.append(geometry.Points(X, thickness=thickness, color=color))


loop = geometry.PointSequence(
    seq,
    display = dict(
        render_as = 'loop',
        size = 'thickness',
        scale = 1,
        color = 'color',
        X0 = [-2, -2, -1],
        X1 = [2, 2, 1],
        colormap = 'twilight',
    )
).save('trefoil.vtp')
