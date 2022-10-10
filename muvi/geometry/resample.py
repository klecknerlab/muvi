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
import sys, os, time
from scipy.spatial import KDTree
import numba

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


_WEIGHT_FUNCS = {
    "flat": 0,
    "gaussian": 1,
    "hann": 2,
}

@numba.njit(cache=True)
def _numba_poly_resample(X, Xi, Yi, indices, ends, cutoff, wf, P, min_points):
    Np, Nd = P.shape
    Ni, Ns = Yi.shape
    Nx, Nd = X.shape

    Y = np.empty((Nx, Ns), Yi.dtype)
    dY = np.empty((Nx, Nd, Ns), Yi.dtype)

    for p in range(Nx):
        # Get input data
        start = ends[p-1] if p else 0
        end = ends[p]
        Nn = end - start
        Xn = Xi[indices[start:end]] - X[p]
        Yn = Yi[indices[start:end]]

        # Make sure there are enough neighbors here...
        if (Nn < min_points):
            Y[p] = np.nan
            dY[p] = np.nan
            continue

        # Compute weights
        rs = (Xn**2).sum(-1)
        rp = np.sqrt(rs) / cutoff

        weight = np.ones(Nn, Yi.dtype)
        if wf == 1:
            weight[:] = np.exp(-rs / (2 * (cutoff/3)**2))
        elif wf == 2:
            weight[:] = 1 + np.cos(rp * np.pi)

        # Build polynomial coefficients and vectors
        A = np.eye(Np, dtype=X.dtype)
#         V = (Xn ** P.reshape(Np, 1, Nd)).prod(-1)
        V = np.ones((Np, Nn), dtype=Y.dtype)
        for i in range(Np):
            for j in range(Nd):
                V[i] *= Xn[:, j] ** P[i, j]

        # Gram-Schmidt normalization (with weights)
        for i in range(Np):
            for j in range(i):
                overlap = (weight * V[j] * V[i]).sum()
                A[i] -= overlap * A[j]
                V[i] -= overlap * V[j]

            norm = (weight * V[i]**2).sum()
            if np.abs(norm) > 0: #Make sure this isn't a zero weight vector!
                norm = norm ** (-0.5)
            A[i] *= norm
            V[i] *= norm

        # Resample data using reconstructed kernels
        for i in range(Nd+1):
            # Kernel construction works because the poly terms are
            # always returned in a specific order
            # 0th index poly -> constant term -> only nonzero poly at origin
            # 1st index poly -> "x" term -> only nonzero x-deriv at origin
            # 2nd index poly -> "y" term...
            kernel = weight * (A[:, i:i+1] * V).sum(0)
            # Thus... this kernel, when multiplied by the data values
            # gives either the value at the origin (i=0) or the derivatives
            # along an axis!

            for j in range(Ns):
                result = (kernel * Yn[:, j]).sum()

                if i:
                    dY[p, i-1, j] = result
                else:
                    Y[p, j] = result

    return Y, dY


@numba.njit(cache=True, parallel=True)
def _numba_poly_resample_parallel(X, Xi, Yi, indices, ends, cutoff, wf, P, min_points):
    Np, Nd = P.shape
    Ni, Ns = Yi.shape
    Nx, Nd = X.shape

    Y = np.empty((Nx, Ns), Yi.dtype)
    dY = np.empty((Nx, Nd, Ns), Yi.dtype)

    for p in numba.prange(Nx):
        # Get input data
        start = ends[p-1] if p else 0
        end = ends[p]
        Nn = end - start
        Xn = Xi[indices[start:end]] - X[p]
        Yn = Yi[indices[start:end]]

        # Make sure there are enough neighbors here...
        if (Nn < min_points):
            Y[p] = np.nan
            dY[p] = np.nan
            continue

        # Compute weights
        rs = (Xn**2).sum(-1)
        rp = np.sqrt(rs) / cutoff

        weight = np.ones(Nn, Yi.dtype)
        if wf == 1:
            weight[:] = np.exp(-rs / (2 * (cutoff/3)**2))
        elif wf == 2:
            weight[:] = 1 + np.cos(rp * np.pi)

        # Build polynomial coefficients and vectors
        A = np.eye(Np, dtype=X.dtype)
#         V = (Xn ** P.reshape(Np, 1, Nd)).prod(-1)
        V = np.ones((Np, Nn), dtype=Y.dtype)
        for i in range(Np):
            for j in range(Nd):
                V[i] *= Xn[:, j] ** P[i, j]

        # Gram-Schmidt normalization (with weights)
        for i in range(Np):
            for j in range(i):
                overlap = (weight * V[j] * V[i]).sum()
                A[i] -= overlap * A[j]
                V[i] -= overlap * V[j]

            norm = (weight * V[i]**2).sum()
            if np.abs(norm) > 0: #Make sure this isn't a zero weight vector!
                norm = norm ** (-0.5)
            A[i] *= norm
            V[i] *= norm

        # Resample data using reconstructed kernels
        for i in range(Nd+1):
            # Kernel construction works because the poly terms are
            # always returned in a specific order
            # 0th index poly -> constant term -> only nonzero poly at origin
            # 1st index poly -> "x" term -> only nonzero x-deriv at origin
            # 2nd index poly -> "y" term...
            kernel = weight * (A[:, i:i+1] * V).sum(0)
            # Thus... this kernel, when multiplied by the data values
            # gives either the value at the origin (i=0) or the derivatives
            # along an axis!

            for j in range(Ns):
                result = (kernel * Yn[:, j]).sum()

                if i:
                    dY[p, i-1, j] = result
                else:
                    Y[p, j] = result

    return Y, dY


def poly_powers(d, order):
    layers = [np.zeros((1, d), 'u4')]

    for i in range(order):
        layer = []
        for j in range(d):
            l = layers[-1].copy()
            l[:, j] += 1
            layer.append(l)
        layer = np.vstack(layer)
        layers.append(np.unique(layer, axis=0)[::-1])

    return np.vstack(layers)


def windowed_polynomial_resample(X, Xi, Yi, cutoff=1, order=2, min_points=None, window='hann', return_counts=False, parallel=False):
    '''Resample a randomly sampled function and its spatial
    derivatives using a windowed polynomial fit resampler.

    Parameters
    ----------
    X : (..., Nd) shaped array
        The points to output the data at
    Xi : (Ni, Nd) shaped array
        The input points (must a 2D array!)
    Yi : (Ni, Ns) shaped array
        The input values (must be 2D array!)

    Keywords
    --------
    cutoff : float (default: 1)
        The radius used for the fit window
    order : int (default: 2)
        The order of the polynomial fit to use
    min_points : int (default: None)
        Minimum number of points needed for fit.  If not specified (or
        the default, None), this is equal to the number of polynomial
        coefficients.  Points which can't be fit will be filled with
        nans
    window: string (default: "hann")
        The weighting function used for the fit.  Options are:
            * "flat": all points in the radius are equally weights
            * "gaussian": a gaussian weight function with σ = cutoff / 3
            * "hann": a single cosine lobe (exactly 0 at the cutoff)
    return_counts : bool (default: False)
        If True, also return neighbor counts for each point
    parallel : bool (default: False)
        If True, use the parallel kernel for resampling.  May give errors on
        some platforms

    Returns
    -------
    Y : (..., Ns) shaped array
        The resampled values
    dY : (..., Nd, Ns) shaped array
        The spatial derivatives of the resampled
        values
    * counts : (...) shaped array
        The number of neighbors used to compute each polynomial
        * only returned if return_counts = True

    '''

    X_shape = X.shape
    Nd = X.shape[-1]
    X = X.reshape(-1, Nd)
    Ns = Yi.shape[-1]

    # Find the neighbors of each point
    # This algorithm is pretty efficient, so no need to reinvent!
    indices = KDTree(X).query_ball_tree(KDTree(Xi), r=cutoff)

    # indices is a list of lists, so we need to alter it to send
    # to numba.  We do this by packing it into a single array,
    # and then keeping track of the indices for each input point
    count = np.array(list(map(len, indices)), dtype='u4')
    end = np.cumsum(count, dtype='u4')
    packed = np.concatenate([np.array(i, 'u4') for i in indices])

    # Select the window
    if window not in _WEIGHT_FUNCS:
        raise ValueError(f"'weight' should be one of: {tuple(_WEIGHT_FUNCS.keys())}")
    weight_func = _WEIGHT_FUNCS[window]

    P = poly_powers(Nd, order)
    if min_points is None:
        min_points = len(P)

    if parallel:
        Y, dY = _numba_poly_resample_parallel(X.reshape(-1, Nd), Xi, Yi, packed, end, float(cutoff), weight_func, P, min_points)
    else:
        Y, dY = _numba_poly_resample(X.reshape(-1, Nd), Xi, Yi, packed, end, float(cutoff), weight_func, P, min_points)

    Y = Y.reshape(X_shape[:-1] + (Ns,))
    dY = dY.reshape(X_shape[:-1] + (Nd, Ns))

    if return_counts:
        return Y, dY, count.reshape(X_shape[:-1])
    else:
        return Y, dY
