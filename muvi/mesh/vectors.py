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


import numpy as np

'''
This module contains basic operations for working with vectors.  In general,
functions expect vectors to be (..., 3) shaped arrays, where the last axis is
the dimension (generally 3, but most of the functions work with 2D vectors as
well.)
'''

class ShapeError(Exception):
    pass

def shape_str(shape):
    return '(' + ', '.join(map(lambda d: '-any-' if d < 0 else str(d), shape)) + ')'


#------------------------------------------------------------------------------
# Basic Vector Operations
#------------------------------------------------------------------------------
def mag(X):
    '''Calculate the length of an array of vectors.'''
    return np.sqrt((np.asarray(X)**2).sum(-1))

def mag1(X):
    '''Calculate the length of an array of vectors, keeping the last dimension
    index.'''
    return np.sqrt((np.asarray(X)**2).sum(-1))[..., np.newaxis]


def dot(X, Y):
    '''Calculate the dot product of two arrays of vectors.'''
    return (np.asarray(X)*Y).sum(-1)


def dot1(X, Y):
    '''Calculate the dot product of two arrays of vectors, keeping the last
    dimension index'''
    return (np.asarray(X)*Y).sum(-1)[..., np.newaxis]


def norm(X):
    '''Computes a normalized version of an array of vectors.'''
    return X / mag1(X)


def plus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, -1, 0)


def minus(X):
    '''Return a shifted version of an array of vectors.'''
    return np.roll(X, +1, 0)


def cross(X, Y):
    '''Return the cross-product of two vectors.'''
    return np.cross(X, Y)


def proj(X, Y):
    r'''Return the projection of one vector onto another.

    Parameters
    ----------
    X, Y : vector array

    Returns
    -------
    Z : vector array
        :math:`\vec{Z} = \frac{\vec{X} \cdot \vec{Y}}{|Y|^2} \vec{Y}`
    '''
    Yp = norm(Y)
    return dot1(Yp, X) * Yp


def midpoint_delta(X):
    '''Returns center point and vector of each edge of the polygon defined by the points.'''
    Xp = plus(X)
    return (Xp + X) / 2., (Xp - X)


def arb_perp(V):
    '''For each vector, return an arbitrary unit vector that is perpendicular.

    **Note: arbitrary does not mean random!**'''
    p = np.eye(3, dtype=V.dtype)[np.argmin(abs(V), -1)]
    return norm(p - proj(p, V))


def angle_between(V1, V2, normalized=False):
    '''Compute the angle between two vectors.

    Parameters
    ----------
    V1, V2 : (..., 3) shaped arrays of floats
        The vectors to compute the angles between.

    Keywords
    --------
    normalized : bool (default: False)
        If False, do not assume the vectors are normal length.  Otherwise, it
        will (and will given incorrect results if they are not)

    Returns
    -------
    phi : (...) shaped array of floats
        The angles between the vectors, in radians
    '''
    if not normalized:
        V1 = norm(V1)
        V2 = norm(V2)

    return np.arccos(np.clip(dot(V1, V2), -1, 1))

#------------------------------------------------------------------------------
# Building vectors intelligently
#------------------------------------------------------------------------------
def vec(x=[0], y=[0], z=[0]):
    '''Generate a [..., 3] vector from seperate x, y, z.

    Parameters
    ----------
    x, y, z: array
        coordinates; default to 0, may have any shape

    Returns
    -------
    X : [..., 3] array'''

    x, y, z = map(np.asarray, [x, y, z])

    s = [1]

    for a in (x, y, z):
        while a.ndim > len(s): s.prepend(1)
        s = [max(ss, n) for ss, n in zip(s, a.shape)]

    v = np.empty(s + [3], 'd')
    v[..., 0] = x
    v[..., 1] = y
    v[..., 2] = z

    return v

#------------------------------------------------------------------------------
# Rotations and basis operations
#------------------------------------------------------------------------------

def rot(a, X=None, cutoff=1E-10):
    '''Rotate points around an arbitrary axis.

    Parameters
    ----------
    a : [..., 3] array
        Rotation vector, will rotate counter-clockwise around axis by an amount
        given be the length of the vector (in radians).  May be a single vector
        or an array of vectors if each point is rotated separately.

    X : [..., 3] array
        Vectors to rotate; if not specified generates a rotation basis instead.

    cutoff : float
        If length of vector less than this value (1E-10 by default), no rotation
        is performed.  (Used to avoid basis errors)

    Returns
    -------
    Y : [..., 3] array
        Rotated vectors or rotation basis.
    '''

    #B = np.eye(3, dtype='d' if X is None else X.dtype)

    a = np.asarray(a)
    if X is None: X = np.eye(3).astype(a.dtype)

    phi = mag(a)
    if phi.max() < 1E-10: return X

    #http://en.wikipedia.org/w/index.php?title=Rotation_matrix#Axis_and_angle
    n = norm(a)
    n[np.where(np.isnan(n).any(-1))] = (1, 0, 0)

    B = np.zeros(a.shape[:-1] + (3, 3), dtype=a.dtype)
    c = np.cos(phi)
    s = np.sin(phi)
    C = 1 - c

    for i in range(3):
        for j in range(3):
            if i == j:
                extra = c
            else:
                if (j - i)%3 == 2:
                    extra = +s * n[..., (j-1)%3]
                else:
                    extra = -s * n[..., (j+1)%3]

            B[..., i, j] = n[..., i]*n[..., j]*C + extra

    if X is not None: return apply_basis(X, B)
    else: return B


def generate_basis(N, B=None):
    '''Generate an orthonormal basis (or bases) from one or two input vectors.

    Parameters
    ----------
    N : (..., 3) shaped array
        Normal vector

    Keywords
    --------
    B : (..., 3) shaped array
        The binormal vector; will be normalized using Gram-Schmidt.  If not
        specified `arb_perp` is used to find a perpendicular vector.

    Returns
    -------
    T : (3, 3) or (N, 3, 3) shaped array
        An orthonormal basis.
    '''

    T = np.empty(N.shape + (3,), dtype=N.dtype)
    T[..., 0, :] = norm(N)

    if B is None:
        T[..., 1, :] = arb_perp(N)
    else:
        T[..., 1, :] = norm(B - dot1(T[..., 0, :], B) * T[..., 0, :])

    T[..., 2, :] = np.cross(T[..., 0, :], T[..., 1, :])

    return T


def normalize_basis(B):
    '''Create right-handed orthonormal basis/bases from input basis.

    Parameters
    ----------
    B : [..., 1-3, 3] array
        input basis, should be at least 2d.  If the second to last axis has
        1 vectors, it will automatically create an arbitrary orthonormal basis
        with the specified first vector.
        (note: even if three bases are specified, the last is always ignored,
        and is generated by a cross product of the first two.)

    Returns
    -------
    NB : [..., 3, 3] array
        orthonormal basis
    '''

    B = np.asarray(B)
    NB = np.empty(B.shape[:-2] + (3, 3), dtype='d')


    v1 = norm(B[..., 0, :])
    v1[np.where(np.isnan(v1).any(-1))] = (1, 0, 0)


    v2 = B[..., 1, :] if B.shape[-2] >= 2 else np.eye(3)[np.argmin(abs(v1), axis=-1)]
    v2 = norm(v2 - v1 * dot1(v1, v2))
    v3 = cross(v1, v2)

    for i, v in enumerate([v1, v2, v3]): NB[..., i, :] = v

    return NB


def path_basis(X, T=None, N=None, closed=False):
    '''Generate an orthonormal basis for a 3D path.

    Parameters
    ----------
    X : (N, 3) shaped array
        The points on the path

    Keywords
    --------
    T, N : None or (N, 3) shaped array
        The tangent and normal vector to the path.  If not specified, created
        automatically.
    closed : bool (default: False)
        If True, assume the path is closed.  The endpoint/startpoint should
        *not* be repeated, and may mess up tangent vector calculation if it is.
        Closed paths where N is not specified will incorporate the minumum
        amount of twist to close the path smoothly.

    Returns
    -------
    B : (N, 3, 3) shaped array
        Orthonormal basis for the path.
    '''

    nX = len(X)
    r = None

    if T is None:
        if closed:
            Δ = plus(X) - X
            r = mag(Δ)
            # Weight neighboring normals based on their distance
            Δmod = norm(Δ) / r
            T = Δmod + minus(Δmod)
        else:
            Δ = X[1:] - X[:-1]
            r = mag(Δ)
            Δmod = norm(Δ) / r
            T = np.emptylike(X)
            T[:-1] = Δmod
            T[-1] = 0
            T[1:] += Δmod

    if N is not None:
        # Normal vector is specified, so let's use it!
        B = np.empty((nX, 2, 3), X.dtype)
        B[:, 0] = T
        B[:, 0] = N
        return normalize_basis(B)

    else:
        # Normal vector is not specified -- construct a minimally twisted path,
        #   defined by parallel transport.
        B = np.empty((nX, 3, 3))
        B[:, 0] = norm(T)
        B[0, 1] = arb_perp(T[0])

        for n in range(1, nX):
            B[n, 1] = norm(B[n-1, 1] - B[n, 0] * dot1(B[n, 0], B[n-1, 1]))

        B[n, 2] = cross(B[n, 0], B[n, 1])

        if closed:
            # If we're closed, we need to reconnect the path in an untwisted
            #  manner -- otherwise there will be a jump in twist at the first
            #  point, which is ugly!
            if r is None:
                r = mag(plus(X) - X)

            Nf = norm(B[-1, 1] - B[0, 0] * dot1(B[0, 0], B[-1, 1]))

            #The extra angle we need to deal with is given by:
            #    arcsin(Ni × Nf ⋅ Ti)
            ϕf = np.arcsin(np.clip(dot(cross(B[0, 1], Nf), B[0, 0]), -1, 1))
            s = np.cumsum(r)
            ϕ = s[:-1] / s[-1] * ϕf
            cϕ = np.cos(ϕ)
            sϕ = np.sin(ϕ)

            Nr =  B[1:, 1] * cϕ + B[1:, 2] * sϕ
            Br = -B[1:, 1] * sϕ + B[1:, 2] * cϕ

            B[1:, 1] = Nr
            B[1:, 2] = Br

        return B

#------------------------------------------------------------------------------
# Utility functions, used by other modules in package
#------------------------------------------------------------------------------

def enforce_shape(arr, shape):
    arr = np.asarray(arr)

    if arr.shape == shape: #Shortcut
        return arr

    if arr.ndim > len(shape):
        raise ShapeError('input has two many dimensions (%s) to cast into shape %s' % (arr.ndim, shape_str(shape)))
    elif arr.ndim < len(shape):
        arr.shape = (1, ) * (len(shape) - arr.ndim) + arr.shape

    new_shape = tuple(so if st<=0 else st for so, st in zip(arr.shape, shape))

    if arr.shape == new_shape: #Shortcut
        return arr

    else:
        tile = []
        for so, st in zip(arr.shape, new_shape):
            if so == st: tile.append(1)
            elif so == 1: tile.append(st)
            else: raise ShapeError('input not castable into array of shape %s' % shape_str(shape))

        return np.tile(arr, tile)


def path_tangents(X, closed=True):
    '''Estimate the tangent vectors for a path

    Parameters
    ----------
    X : (N, 2-3) shaped array
        The points on the path

    Keywords
    --------
    closed : bool (default: True)
        Is this a closed or open path?

    Returns
    -------
    T : array like X
        The normalized tangent vectors.  The tangents are approximately the
        tangents to a circle going through each 3 point section
    '''
    if closed:
        Δ = plus(X) - X
        Δ /= mag1(Δ)**2
        return norm(Δ + minus(Δ))
    else:
        Δ = X[1:] - X[:-1]
        Δ /= mag1(Δ)**2
        T = np.empty_like(X)
        T[:-1] = Δ
        T[-1] = 0
        T[1:] += Δ
        return norm(T)


def path_delta(X, closed=True):
    if closed:
        return plus(X) - X
    else:
        return X[1:] - X[:-1]


def path_sum_delta(ds, closed=True):
    n = len(ds) + (0 if closed else 1)
    s = np.empty(n, dtype=ds.dtype)

    s[0] = 0
    cs = np.cumsum(ds)

    if closed:
        s[1:] = cs[:-1]
    else:
        s[1:] = cs

    return s
