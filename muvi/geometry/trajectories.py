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

import numpy as np
from .points import Points, PointSequence
import sys, os, time
from .resample import MSGResampler


class Trajectories(PointSequence):
    def __init__(self, traj, dt=1, N=15, P=2, min_track_len=5, min_valid=None,
            window=None, pos=('xc', 'yc', 'zc'), vectors={}, scalars={},
            dtype='f', verbose=None, max_tracks=None):
        '''Create a PointSequence corresponding to the trajectories obtained
        from TrackPy or similar.  The actual positions and velocities are
        computed by running the data through a modified Savitzky Golay resampler
        which improves noise rejection.

        (Effectively the position and velocity are determined by fitting the
        neighborhood around each point.)

        Parameters
        ----------
        traj : Pandas data array
            Data array containing the tracked and linked particles.  Must have
            the 'frame' and 'particle' fields defined.

        Keywords
        --------
        dt : float (default: 1)
            The time interval between frames (used to normalize velocities)
        N : int (default: 15)
            The size of the fit window
        P : int (default: 2)
            The order of the polynomial used in the fit
        min_track_len : int (default: 5)
            The minimum track length to consider
        min_valid : int (default: P+1)
            The minimum number of points in a window required to return a
            value (has no effect unless there are missing points in a track)
        window : array of length N (default: None)
            If specified, the window to use for the fit weights.  Default is a
            Hann window
        dtype : string or numpy data type (default: 'f')
            The type used to store data.  By default single precision is used
            to save memory.
        pos : tuple (default: ('xc', 'yc', 'zc'))
            Fields used to compute position vector
        vectors : dict (default: {})
            Dictionary containing keys which are strings (vector names) and
            entries which are tuples of fields which correspond to that vector.
            (Example: `vectors={'index':('x', 'y', 'z')}`)
        scalars : dict (default: {'mass':'mass', 'size':'size'})
            Dictionary containing scalar output names and corresponding fields
        verbose : bool (default: auto-determined)
            If True, automatically print status updates as the data is loaded
            or saved.  By default, this happens if the source data contains
            more than 10^4 particles identified
        max_tracks : int (default: auto-determined)
            Maximum number of particle tracks to evaluate -- can be used to
            test data without completing full tracks.  By default, all tracks
            are included
        '''

        if verbose is None:
            verbose = len(traj) > 10000
        self.verbose = bool(verbose)

        if self.verbose:
            print('Sorting particles...', end='')
            sys.stdout.flush()
            start = time.time()

        traj.sort_values(['particle', 'frame'], inplace=True)

        if self.verbose:
            el = time.time() - start
            print(f'\rParticles sorted in {el:.2f} s.')

        if self.verbose:
            print('Splitting trajectories...', end='')
            sys.stdout.flush()
            start = time.time()

        tracks = []
        p = traj['particle'].to_numpy()


        frames = traj['frame']
        self.frame0 = frames.min()
        self.frame1 = frames.max()
        self._Nt = self.frame1 - self.frame0 + 1

        self.count = np.zeros(self._Nt, 'u8')
        self.last_i = np.zeros_like(self.count)

        split = np.where(p[1:] != p[:-1])[0] + 1

        Np = 0
        i0 = 0
        for i in range(len(split)+1):
            i1 = split[i] if (i < len(split)) else (len(traj) + 1)
            if (i1 - i0) >= min_track_len:
                track = traj[i0:i1]
                tracks.append(track)
                frames = track['frame']
                f0 = frames.min() - self.frame0
                f1 = frames.max() - self.frame0 + 1
                self.count[f0:f1] += 1
                Np += 1
                if max_tracks and Np >= max_tracks:
                    break
            i0 = i1

        if self.verbose:
            el = time.time() - start
            print(f'\rTrajectories split in {el:.2f} s.')

        self._N = self.count.max()
        self._dat = {
            'pos': np.full((self._Nt, self._N, 3), np.nan, dtype=dtype),
            'vel': np.full((self._Nt, self._N, 3), np.nan, dtype=dtype),
        }

        if self.verbose:
            print('Resampling trajectories...', end='')
            sys.stdout.flush()
            start = time.time()

        resampler = MSGResampler(N=N, P=P, min_valid=min_valid, window=window)

        for track in tracks:
            f = track['frame'].to_numpy()
            i0, i1 = f.min() - self.frame0, f.max() - self.frame0 + 1
            for i, field in enumerate(pos):
                # print(f)
                y, x, dx = resampler(f, track[field].to_numpy(), dx=dt, remove_invalid=False)
                j = y - self.frame0
                k = self.last_i[j]
                self._dat['pos'][j, k, i] = x
                self._dat['vel'][j, k, i] = dx

            self.last_i[i0:i1] += 1

        if self.verbose:
            el = time.time() - start
            print(f'\rTrajectories resampled in {el:.2f} s.')

        self._d = {i:None for i in range(self.frame0, self.frame1)}

    def save(self, *args, **kwargs):
        # if self.verbose:
        #     print('Saving...', end='')
        #     sys.stdout.flush()
        #     start = time.time()
        super().save(*args, **kwargs, print_status=self.verbose)
        #
        # if self.verbose:
        #     el = time.time() - start
        #     print(f'\rSaved in {el:.4f} s.')

    def _get(self, i):
        if i not in self._d:
            raise KeyError(f"Invalid timestep: {i}")

        good = np.where(np.isfinite(self._dat['pos'][i-self.frame0, :, 0]))[0]
        kwargs = {key:dat[i-self.frame0, good] for key, dat in self._dat.items()}
        return Points(kwargs.pop('pos'), **kwargs)
