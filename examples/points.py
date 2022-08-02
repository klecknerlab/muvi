#!/usr/bin/python3
#
# Copyright 2022 Dustin Kleckner
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

odir = 'geometry_examples'
if not os.path.exists(odir):
    os.makedirs(odir)
os.chdir(odir)

# First examples: Point class
# This is for a single timestep containing a bunch of points and attached data
# For example: tracked particles
# This class pretty much behaves like a dictionary containing numpy arrays.
# At the very least it will have a 'pos' entry, which is the point positions
# Arbitrary other arrays can be attached assuming the length is the same!
# By convention, we will use 'vel' for the velocity of the points, which will
# be used by other things later.
# This class supports saving and loading from vtp files, and is compatible with
# Paraview!

# Create a new points array with mass and velocity attached attributes
# Note that velocity is a vector, and mass is a scalar, but both have
#   the same length as the position vectors!
N = 50
points = geometry.Points(
    np.random.rand(N, 3),
    vel = np.random.rand(N, 3),
    mass = np.random.rand(N)
)

# Get mass and promote order of array
# (If you don't do this you can't compute m * vel, for example!)
m = points['mass'].reshape(-1, 1)

# Set some new attributes
points['p'] = points['vel'] * m
points['moment'] = points['pos'] * m

# Save to a CSV file
points.save('points.csv')

# Save to a VTP file
points.save('points.vtp')

# Check that the VTP file contains the same points
points_loaded = geometry.load_geometry('points.vtp')
print(f'Type of object returned from loading "points.vtp": {type(points_loaded)}')
print(f"RMS difference between original and saved data: {(points['pos'] - points_loaded['pos']).std():.2e}")
print('Note: nonzero because of double -> single precision conversion!')

# Save to a VTP file with double precision, and check difference
points.save('points_dp.vtp', force_floats=False)
points_loaded = geometry.load_geometry('points_dp.vtp')
print(f"RMS difference between original and double precision saved data: {(points['pos'] - points_loaded['pos']).std():.2e}")


# Second example: PointSequence class
# This contains a sequence of 'Points' obtains, and also behaves like a list
# or dictionary with integer entries
# This can also be saved/loaded from VTP files, and viewed with Paraview

# Start with some positions and velocities
X = np.random.rand(N, 3)
V = np.random.rand(N, 3) - 0.5
seq = {}

# Here our timesteps start at 10, instead of 0
# This is useful, for example, if we have only tracked certain frames in
#   a dataset.
# Note that ParaView will ignore this, and all steps start at 0!
for i in range(10, 50):
    t = i * 0.05
    # Change the number of points in each, just to demonstrate capability
    X = X[:-1]
    V = V[:-1]

    V -= (X-0.5) * 0.05 # Change the velocity a bit
    X += V * 0.05

    seq[i] = geometry.Points(X.copy(), vel=V.copy())
    # Note that the Points class doesn't copy arrays by default, so if we
    # modify them post-facto it will change the data!

# Turn this into a sequence, and save!
geometry.PointSequence(seq).save('sequence.vtp')

# Check that the saved data is the same
seq2 = geometry.load_geometry('sequence.vtp')
print(f'Type of object returned from loading "sequence.vtp": {type(seq2)}')

frame = 30
p1 = seq[frame]
p2 = seq2[frame]
#
# print(p1.keys())
# print(p2.keys())

print(f"Sequence: RMS difference between original and saved data: {(p1['pos'] - p2['pos']).std():.2e}")
