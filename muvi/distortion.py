#!/usr/bin/python3
#
# Copyright 2020 Dustin Kleckner
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


# Known coordinate spaces:
#  raw: texture coordinates (0-1), in raw volume
#  corrected: texture coordinates (0-1), in corrected space
#  physical:
import numpy as np

class DistortionModel:
    # C code to obtain raw coordinates for corrected
    c_defs = []
    c_u = 'u'
    c_v = 'v'
    c_w = 'w'

    # C code to obtain derivative of raw coordinates with respect to corrected
    c_du = '1.0, 0.0, 0.0'
    c_dv = '0.0, 1.0, 0.0'
    c_dw = '0.0, 0.0, 1.0'

    def __init__(self, info):
        self.update_params(info)
        self.update_size(info)


    def update_params(self, info):
        self.params = ()


    def update_size(self, info):
        self.Lx = info.get('Lx', info.get('Nx', 1))
        self.Ly = info.get('Ly', info.get('Ny', 1))
        self.Lz = info.get('Lz', info.get('Nz', 1))


    def raw_to_corrected(self, up, vp=None, wp=None):
        '''Converted raw coordinates to idealized coordinates ('corrected')
        in normalized volume space.

        Can be called by passing each coordinate separately or a numpy array
        with all coordinates.
        '''
        if vp is None:
            u, v, w = self._raw_to_corrected(up[..., 0], up[..., 1], up[..., 2])
            result = np.empty_like(up)
            result[..., 0] = u
            result[..., 1] = v
            result[..., 2] = w
            return result

        else:
            return self._raw_to_corrected(up, vp, wp)


    def _raw_to_corrected(self, up, vp, wp):
        return (up, vp, wp)


    def corrected_to_raw(self, u, v=None, w=None):
        '''Converted idealized coordinates ('corrected') to raw coordinates
        in normalized volume space.

        Can be called by passing each coordinate separately or a numpy array
        with all coordinates.
        '''
        if v is None:
            up, vp, wp = self._corrected_to_raw(u[..., 0], u[..., 1], u[..., 2])
            result = np.empty_like(u)
            result[..., 0] = up
            result[..., 1] = vp
            result[..., 2] = wp
            return result
        else:
            return self._corrected_to_raw(u, v, w)


    def _corrected_to_raw(self, u, v, w):
        return (u, v, w)


    def raw_to_physical(self, u, v=None, w=None):
        pass

    def glsl_funcs(self):
        fmt = {
            'defs':'\n    '.join('float %s;' % d for d in self.c_defs),
            'u': self.c_u,
            'v': self.c_v,
            'w': self.c_w,
            'du': self.c_du,
            'dv': self.c_dv,
            'dw': self.c_dw,
        }

        return '''
vec3 distortion_map(in vec3 U) {{
    float u = U.x;
    float v = U.y;
    float w = U.z;
    {defs}

    return vec3({u}, {v}, {w});
}}

mat4x3 distortion_map_gradient(in vec3 U){{
    float u = U.x;
    float v = U.y;
    float w = U.z;
    {defs}

    mat4x3 map_grad;
    map_grad[0] = vec3({u}, {v}, {w});
    map_grad[1] = vec3({du});
    map_grad[2] = vec3({dv});
    map_grad[3] = vec3({dw});

    return map_grad
}}
        '''.format(**fmt)


class Undistorted(DistortionModel):
    # The default distortion model is undistorted -- this is here for clarity
    pass


class XScan(DistortionModel):
    # C code to obtain raw coordinates for corrected
    c_defs = [
        'eps_x = 0.25 * param0 * (1.0 - 2.0 * u)',
        'eps_z = 0.25 * param1 * (1.0 - 2.0 * w)',
        'div_x = 1.0 / (1.0 + 2.0 * eps_x)',
        'div_z = 1.0 / (1.0 + 2.0 * eps_z)'
    ]
    c_u = '(u + eps_z) * div_z'
    c_v = '(v + eps_z) * div_z'
    c_w = '(w + eps_x) * div_x'

    # C code to obtain derivative of raw coordinates with respect to corrected
    c_du = 'div_z, 0, param1 * (u - 0.5) * div_z*div_z'
    c_dv = '0, div_z, param1 * (v - 0.5) * div_z*div_z'
    c_dw = 'param0 * (w - 0.5) * div_x*div_x, 0, div_x'


    def update_params(self, info):
        self.params = (
            info['Lx'] / info['dx'],
            info['Lz'] / info['dz']
        )

    def _corrected_to_raw(self, u, v=None, w=None):

        eps_x = 0.25 * self.params[0] * (1.0 - 2.0 * u)
        eps_z = 0.25 * self.params[1] * (1.0 - 2.0 * w)
        div_x = 1.0 / (1.0 + 2.0 * eps_x)
        div_z = 1.0 / (1.0 + 2.0 * eps_z)

        return (
            (u + eps_z) * div_z,
            (v + eps_z) * div_z,
            (w + eps_x) * div_x
        )


    def _raw_to_corrected(self, up, vp=None, wp=None):
        eps_x = 0.25 * self.params[0] * (1.0 - 2.0 * up)
        eps_z = 0.25 * self.params[1] * (1.0 - 2.0 * wp)
        div = 1.0 / (1.0 - 4.0 * eps_x * eps_z)

        return (
            (up + eps_z * (2 * up - 1 - 2 * eps_x)) * div,
            (vp + eps_z * (2 * vp - 1 - 2 * eps_x)) * div,
            (wp + eps_x * (2 * wp - 1 - 2 * eps_z)) * div,
        )

def get_distortion_map(info):
    if 'dx' in info and 'dz' in info:
        if 'dy' in info:
            raise ValueError('Volume info specified both dx and dy -- should only have one or the other!')
        return XScan(info)

    else:
        return Undistorted(info)


if __name__ == "__main__":
    # import numpy as np
    info = {'Lx': 10, 'Ly': 8, 'Lz': 5, 'dx':100, 'dz':200}
    # info = {}

    distortion = get_distortion_map(info)

    print(distortion.glsl_funcs())

    X = np.random.rand(10, 3)
    # print(X)

    Xp = distortion.corrected_to_raw(X)
    # print(Xp)

    Xpp = distortion.raw_to_corrected(Xp)
    print('RMS error in round trip calculation: %e' % (Xpp-X).std())
