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

# Modifiied Savitzky Golay resampler
# Data is cached in this directory
if sys.platform.startswith("win"):
    CACHE_BASE = os.getenv("LOCALAPPDATA")
elif sys.platform.startswith("darwin"):
    CACHE_BASE = "~/Library/Application Support"
else: # linux
    CACHE_BASE = os.getenv("XDG_DATA_HOME", "~/.local/share")

CACHE_DIR = os.path.join(os.path.expanduser(CACHE_BASE), 'muvi')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
CACHE_PATH = os.path.join(CACHE_DIR, 'msg_kernels_%d.npz')

# Actual resampler class
class MSGResampler:
    def __init__(self, N, P, min_valid=None, window=None, override_size=False):
        '''A class for resampling data using a modified Savitzky Golay method.

        As compared to a traditional S.G. resampler, this method can deal with
        missing points, and also uses a window function to weight the effective
        fits.

        Creating the resmapler object may take up to several seconds, as the
        weight matrices are computed at this time.  It should be reused if
        needed for multiple resamplings.  It is also cached locally, so it
        should be fast the next time you use it.

        Parameters
        ----------
        N : int
            Number of points in the window, must be odd
        P : int
            Order of resampler

        Keywords
        --------
        min_valid : int (default: N//2 + 1)
            The number of points required for a valid point; nan's are returned for invalid
            points
        window : numpy array of size N
            The window used to compute the weight matrix.  Default is a Hanning window.
        '''

        if N % 2 != 1:
            raise ValueError('N must be odd!')
        M = N // 2

        if M > 7 and not override_size:
            raise ValueError('M > 7 -- this will take too long to compute\n(Use override_size keyword to do it anyway)')

        if min_valid is None:
            min_valid = P + 1

        if window is None:
            self.w = np.hanning(N+2)[1:-1] # Endpoints are zero... so eliminate them
        elif window.shape != (N,):
            raise ValueError('Size of window must be N')
        else:
            self.w = window

        self.min_valid = min_valid
        self.N = N
        self.M = M
        self.P = P
        self.bits = 1<<(np.arange(N))

        fn = CACHE_PATH % hash((self.N, self.P, self.min_valid) + tuple(self.w))
        if os.path.exists(fn):
            try:
                dat = np.load(fn)
                self.k0 = dat['k0']
                self.k1 = dat['k1']

            except:
                print('MSGResampler failed to load cached kernels -- rebuilding...')

        if not hasattr(self, 'k0'):
            self._build_kernels()
            np.savez(fn, k0=self.k0, k1=self.k1)

    def _build_kernels(self):
        N, M, P = self.N, self.M, self.P
        i = np.arange(-M, M+1)
        self.k0 = np.zeros((2**N, N))
        self.k1 = np.zeros((2**N, N))

        ip = i.reshape(1, -1) ** np.arange(P+1).reshape(-1, 1)

        # Compute the Kernels for each "bitmask", which corresponds to which
        #   points are valid in the fitting region
        for bitmask in range(2**N):
            valid = (bitmask & self.bits) > 0
            if valid.sum() < self.min_valid:
                self.k0[bitmask] = np.nan
                self.k1[bitmask] = np.nan
                continue

            # Set the weights of invalid points to zero
            w_m = self.w * valid

            # Gram-Schmidt normalization of polynomials
            A = np.eye(P+1)
            p = np.empty((P+1, N))
            for j in range(P+1):
                p[j] = (A[j].reshape(-1, 1) * ip).sum(0)

                for k in range(j):
                    A[j] -= (w_m * p[j] * p[k]).sum() * A[k]

                p[j] = (A[j].reshape(-1, 1) * ip).sum(0)
                norm = (w_m * p[j]**2).sum()**(-0.5)
                A[j] *= norm
                p[j] *= norm

            A_d1 = A[:, 1:] * np.arange(1, P+1)
            p_d1 = (A_d1[..., np.newaxis] * ip[:-1]).sum(1)

            # Fill in kernels for function and it's derivative
            self.k0[bitmask] = (w_m * p * p[:, M].reshape(-1, 1)).sum(0)
            self.k1[bitmask] = (w_m * p * p_d1[:, M].reshape(-1, 1)).sum(0)

    def __call__(self, x, y, dx=1, return_derivitive=True, remove_invalid=True):
        if not isinstance(x, np.ndarray) or not np.issubdtype(x.dtype, np.integer):
            raise TypeError('x must be an array of integers')

        x0 = x[0] - self.M
        i = np.arange(x0, x[-1] + self.M + 1)

        valid = np.zeros(i.shape, 'u1')
        valid[x - x0] = 1

        yf = np.zeros(i.shape, y.dtype)
        yf[x - x0] = y

        Nr = x[-1] - x[0] + 1
        yr = np.empty(Nr, y.dtype)
        if return_derivitive:
            dyr = np.empty(Nr, y.dtype)

        for i0 in range(Nr):
            i1 = i0 + self.N

            bitmask = (valid[i0:i1] * self.bits).sum()
            yr[i0] = (yf[i0:i1] * self.k0[bitmask]).sum()
            if return_derivitive:
                dyr[i0] = (yf[i0:i1] * self.k1[bitmask]).sum() / dx

        ir = i[self.M:-self.M]

        if remove_invalid:
            good = np.where(np.isfinite(yr))
            if return_derivitive:
                return ir[good], yr[good], dyr[good]
            else:
                return ir[good], yr[good]
        else:
            if return_derivitive:
                return ir, yr, dyr
            else:
                return ir, yr


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
