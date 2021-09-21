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
from muvi import VolumetricMovie, VolumeProperties, open_3D_movie
from muvi.distortion import get_distortion_model
import numpy as np
import time
from scipy import ndimage
import os
from scipy.spatial.transform import Rotation
import pickle
from scipy.interpolate import UnivariateSpline

GAUSSIAN_SIZE = 2
FRAMES = 60
DT = 1 / FRAMES

def mag(X):
    return np.sqrt((X**2).sum(-1))

output_dir = "tracking_example"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Our sample points are a cube, which is then rotated.
cube = 5 + 20 * ((np.arange(8).reshape(-1, 1) // 2**np.arange(3)) % 2 - 0.5).astype('f')

vec = np.array([1, 2, 4], 'f')
vec *= 2 * np.pi / mag(vec)

print(mag(vec))

rot_cube = [
    Rotation.from_rotvec(n / FRAMES * vec).apply(cube)
        for n in range(FRAMES)]

# MUVI file name
mfn = os.path.join(output_dir, "tracking.vti")
if not os.path.exists(mfn):
    print(f'Generating volumetric movie: {mfn}')

    # Generate a properties object so we can get a distortion object
    # Note that by specifying dx/dz, which are introducing distortion, which
    #   is quite severe in this case.  Try setting the distortion corection
    #   factors to 0 in the viewer to see how warped the raw data is!
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

    # Create a grid of indices, which correspond to the voxel indices...
    indices = np.mgrid[:info['Nx'], :info['Ny'], :info['Nz']].T # Tranposing changes shape to [Nz, Ny, Nx, 3]
    # ... and convert this to physical positions in the distorted volume.
    distortion = get_distortion_model(info)
    X = distortion.convert(indices, 'index-xyz', 'physical')

    # Each point will be a Gaussian blob
    def gaussians(X0, X, w=1):
        data = np.zeros(X.shape[:-1], 'd')
        for x0 in X0:
            data += np.exp(-((X - x0)**2).sum(-1) / w**2)
        return data

    # Generate the frames
    frames = [np.clip(gaussians(r_cube, X, GAUSSIAN_SIZE) * 200, 0, 255).astype('u1') for r_cube in rot_cube]

    # ... and save them!
    VolumetricMovie(frames, info).save(mfn)
else:
    print(f'Found volumetric movie: {mfn}')
    print('Skipping creation; delete file to regenerate...')


dfn = os.path.join(output_dir, 'tracking.pickle')
if not os.path.exists(dfn):
    print("Tracking points...")
    import trackpy as tp
    # Open the 3D movie
    vol = open_3D_movie(mfn)

    # Identify the points in the volume
    data = tp.batch(vol, 5)
    # Note: the second parameter is the size in the frame.  In this case,
    #   5 is approximately right, but you would need to tweak for a different
    #   data set!

    # Link the particles into tracks
    data = tp.link(data, 10, memory=3)
    print(f'Found {data.particle.max()+1} particle tracks')

    # Find the physical coordinates.  These will appear as the columns "xc", "yc",
    # and "zc" in the data frame *after* you run this command, and will be in the
    # physical units of the data (i.e. L and not N)
    vol.distortion.update_data_frame(data)

    # Save to a pickle file
    with open(dfn, "wb") as f:
        pickle.dump(data, f)
else:
    print(f"Found existing track data, delete {dfn} to regenerate...")


with open(dfn, "rb") as f:
    data = pickle.load(f)

# Let's make a movie where each point is a sphere!
pfn = os.path.join(output_dir, 'tracking_points_frame%d.ply')
from muvi.mesh import generate_glyphs

for i in range(FRAMES):
    points = np.array(data[data.frame == i][["xc", "yc", "zc"]])
    generate_glyphs(points, "sphere", a=GAUSSIAN_SIZE*1.5).save(pfn % i)

# Let's make a movie with vector arrows for each point
vfn = os.path.join(output_dir, 'tracking_vel_frame%d.ply')

# First we need to construct a trajectory for each particle track, which
#   we'll do with UnivariateSplines.
# This is a lot to keep track of, so we'll define a class for this!
# Note: err is the expected error for each particle location.  Setting this
#   higher will result in a smoother trajectory -- note that the output
#   locations from the trajectories will *not* exactly match the input positions
#   (unless you set s=0, which you probably shouldn't!)
class Trajectories:
    def __init__(self, data, dt=1.0, err=0.25, min_length=3, columns=["xc", "yc", "zc"]):
        self.tracks = []

        for i in range(data.particle.max() + 1):
            track = data[data.particle == i]
            if len(track) < min_length:
                continue

            t = np.array(track['frame'] * dt)

            items = [(t.min(), t.max())]
            X = []
            V = []

            for column in columns:
                spline = UnivariateSpline(t, np.array(track[column]), s=len(t) * err**2)
                X.append(spline)
                V.append(spline.derivative())

            self.tracks.append(((t.min(), t.max()), tuple(X), tuple(V)))

    def __call__(self, t):
        X = []
        V = []
        for limits, XS, VS in self.tracks:
            if t >= limits[0] and t <= limits[1]:
                X.append([S(t) for S in XS])
                V.append([S(t) for S in VS])

        return np.array(X), np.array(V)

# Build the trajactory object
traj = Trajectories(data, dt=DT)

# And make the frames!
for i in range(FRAMES):
    X, V = traj(i * DT)
    generate_glyphs(X, "tick", a=GAUSSIAN_SIZE*5, N=V, color=mag(V), clim=(0, 150)).save(vfn % i)
