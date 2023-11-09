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

    # The geometry.Points object holds the data for a single frame.
    #   Note that we can append *arbitrarily named* attributes to the points,
    #   which can be used in the display software later!
    #   In this case we are adding the fields "t" and "c", but the names are
    #   completely user defined.  It is also possible to attach vectors, in
    #   which case you should pass an array which shape (N, 3), where N is the
    #   number of points.
    seq.append(geometry.Points(X, t=thickness, c=color))


loop = geometry.PointSequence(
    seq,
    display = dict(
        render_as = 'loop', # Display as a closed loop rather than discrete points
        size = 't', # The field used to determine diameter of the rendered 3D tube; can vary with position!
        scale = 1, # Scale factor for the size
        color = 'c', # The field used to determine the (varying) color of the line
        X0 = [-2, -2, -1], # The display limits; if not specified these are automatically determined.
        X1 = [2, 2, 1], # The display limits; if not specified these are automatically determined.
        colormap = 'twilight', # The colormap used to convert the color value ("c" field) to an actual tube color.
    )
).save('trefoil.vtp')
