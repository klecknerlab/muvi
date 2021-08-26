from muvi.mesh import *
import numpy as np
π = np.pi
import os

os.chdir('../muvi/mesh/glyphs')

# Construct all the corners of a unit icosahedron
# https://en.wikipedia.org/wiki/Regular_icosahedron#Cartesian_coordinates
ϕ = (1 + 5**.5) / 2
d1 = (1 + ϕ**2)**-0.5
d2 = ϕ * d1
N = np.array([(0, d1, d2), (0, -d1, d2), (0, d1, -d2), (0, -d1, -d2)], dtype='f')
N = np.vstack([N, np.roll(N, -1, axis=-1), np.roll(N, -2, axis=-1)])

# The corners of the faces of an icosahedron
# These were determined by hand -- a pain!
tris = np.array([
    (0, 1, 8), (0,10, 1), (0, 8, 4), (0, 4, 5), (0, 5,10),
    (1,10, 7), (1, 7, 6), (1, 6, 8), (2, 3,11), (2,11, 5),
    (2, 5, 4), (2, 4, 9), (2, 9, 3), (3, 9, 6), (3, 6, 7),
    (3, 7,11), (4, 8, 9), (5,11,10), (6, 9, 8), (7,10,11),
], dtype='i4')

sphere = Mesh(0.5*N, tris, N)
sphere.save('sphere.ply')

def subdivide_sphere(m):
    edges = {}
    points = []

    for i in range(len(m.triangles)):
        for j in range(3):
            e = (m.triangles[i, j], m.triangles[i, (j+1)%3])

            if e not in edges:
                n = len(points) + len(m.points)
                edges[e] = n
                edges[e[::-1]] = n
                points.append(m.points[e[0]] + m.points[e[1]])

    points = np.vstack([m.points, np.array(points)])

    tris = []

    for v1, v2, v3 in m.triangles:
        e1 = edges[(v1, v2)]
        e2 = edges[(v2, v3)]
        e3 = edges[(v3, v1)]
        tris.append((v1, e1, e3))
        tris.append((e1, v2, e2))
        tris.append((e2, v3, e3))
        tris.append((e1, e2, e3))

    tris = np.vstack(tris)

    N = norm(points)

    return Mesh(0.5*N, tris, N)

sphere2 = subdivide_sphere(sphere)
sphere2.save('sphere2.ply')

sphere3 = subdivide_sphere(sphere2)
sphere3.save('sphere3.ply')

sphere4 = subdivide_sphere(sphere3)
sphere4.save('sphere4.ply')

def cap_indices(N, flip=False):
    i = np.arange(1, N-1, dtype='i4')
    indices = np.array([np.zeros(N-2, dtype='i4'), i+1, i]).T

    if flip:
        indices = indices[:, ::-1]

    return indices

def tube_indices(N, flip=False, cap=False):
    i = np.arange(N, dtype='i4')
    ip = (i+1) % N
    if cap:
        indices = np.array([i, i+N, ip]).T.reshape(-1, 3)
    else:
        indices = np.array([i, i+N, ip+N, i, ip+N, ip]).T.reshape(-1, 3)

    if flip:
        indices = indices[:, ::-1]

    return indices

for N_circum, label in [(10, ""), (40, "2")]:
    radius = 0.1

    ϕ = np.linspace(0, 2*π, N_circum, False)
    N = np.array([0*ϕ, np.sin(ϕ), np.cos(ϕ)]).T
    xh = np.eye(3)[0]

    points = np.vstack([
        N * radius - 0.5*xh,
        N * radius - 0.5*xh,
        N * radius + 0.5*xh,
        N * radius + 0.5*xh
    ])

    normals = np.vstack([
        -np.tile(xh, (N_circum, 1)),
        N,
        N,
        np.tile(xh, (N_circum, 1)),
    ])

    i = np.arange(N_circum, dtype='i')
    tris = np.vstack([
        cap_indices(N_circum, flip=True),
        tube_indices(N_circum) + N_circum,
        cap_indices(N_circum) + 3 * N_circum
    ])

    Mesh(points, tris, normals).save(f'cylinder{label}.ply')

    radius = 0.075
    cap_radius = radius * 2
    cap_h = 0.5

    cap_N = norm(N * cap_h + (cap_radius, 0, 0))

    δ = π / N_circum
    Nr = np.array([0*ϕ, np.sin(ϕ + δ), np.cos(ϕ + δ)]).T
    cap_Nr = norm(Nr * cap_h + (cap_radius, 0, 0))

    points = np.vstack([
        N * radius - 0.5*xh,
        N * radius - 0.5*xh,
        N * radius + (0.5-cap_h)*xh,
        N * radius + (0.5-cap_h)*xh,
        N * cap_radius + (0.5-cap_h)*xh,
        N * cap_radius + (0.5-cap_h)*xh,
        Nr * 0 + 0.5*xh,
    ])

    normals = np.vstack([
        -np.tile(xh, (N_circum, 1)),
        N,
        N,
        -np.tile(xh, (N_circum, 1)),
        -np.tile(xh, (N_circum, 1)),
        cap_N,
        cap_Nr
    ])

    tris = np.vstack([
        cap_indices(N_circum, flip=True),
        tube_indices(N_circum) + N_circum,
        tube_indices(N_circum) + 3*N_circum,
        tube_indices(N_circum, cap=True) + 5*N_circum,
    ])

    Mesh(points, tris, normals).save(f'arrow{label}.ply')


    radius = 0.15

    cap_N = norm(N + (radius, 0, 0))

    δ = π / N_circum
    Nr = np.array([0*ϕ, np.sin(ϕ + δ), np.cos(ϕ + δ)]).T
    cap_Nr = norm(Nr + (radius, 0, 0))

    points = np.vstack([
        N * radius - 0.5*xh,
        N * radius - 0.5*xh,
        Nr * 0 + 0.5*xh,
    ])

    normals = np.vstack([
        -np.tile(xh, (N_circum, 1)),
        cap_N,
        cap_Nr
    ])

    tris = np.vstack([
        cap_indices(N_circum, flip=True),
        tube_indices(N_circum, cap=True) + N_circum,
    ])

    # keep = np.array([0, N_circum, 1]) + N_circum
    # points = points[keep]
    # normals = normals[keep]
    # tris = np.array([0, 1, 2])

    Mesh(points, tris, normals).save(f'tick{label}.ply')
