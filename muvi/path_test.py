#!/usr/bin/env python
# -*- coding: utf-8 -*-
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

import numpy as np
import pathgeometry
import matplotlib.pyplot as plt

a = 0.5
trefoil = pathgeometry.torus_knot(a=a, N=20)
x, y, z = trefoil.X.T

ax1 = plt.subplot(311)
plt.plot(x, y, 'ro')
plt.gca().set_aspect('equal')
ax2 = plt.subplot(312)
plt.plot(x, z, 'ro', label='Coarse')
plt.gca().set_aspect('equal')
ax3 = plt.subplot(313)
plt.plot(trefoil.t[:trefoil.n], trefoil.dXdt[:, 1], 'ro')

T = 0.5 * trefoil.T * trefoil.ds.reshape(-1, 1)

for XX, TT in zip(trefoil.X, T):
    P = np.array([XX - TT, XX + TT])
    plt.sca(ax1)
    plt.plot(P[:, 0], P[:, 1], 'k-')
    plt.sca(ax2)
    plt.plot(P[:, 0], P[:, 2], 'k-')

trefoil_smooth = trefoil.smooth_resample()
x, y, z = trefoil_smooth.X.T
plt.sca(ax1)
plt.plot(x, y, 'b-', label='Resampled')
plt.sca(ax2)
plt.plot(x, z, 'b-')

plt.sca(ax3)
plt.plot(trefoil_smooth.t[:trefoil_smooth.n], trefoil_smooth.dXdt[:, 1], 'b-')


trefoil_smooth2 = pathgeometry.torus_knot(a=a, N=1000)
x, y, z = trefoil_smooth2.X.T
params = dict(alpha=0.3, zorder=-1, linewidth=5)
plt.sca(ax1)
plt.plot(x, y, 'k', **params)
plt.sca(ax2)
plt.plot(x, z, 'k', label='Exact', **params)
plt.sca(ax3)
plt.plot(trefoil_smooth2.t[:trefoil_smooth2.n], trefoil_smooth2.dXdt[:, 1], 'k', **params)

plt.sca(ax1)
yl = plt.ylim()
plt.sca(ax2)
plt.ylim(yl)
plt.legend()

plt.show()
