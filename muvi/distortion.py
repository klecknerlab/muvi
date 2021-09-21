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
    '''Used to convert betwenn coordinate systems for distorted volumes.
    There are three basic coordinate systems you should be aware of:

    * `"physical"`: The real coordinates in physical space
        - x = (-Lx/2 -- +Lx/2) = (u - 1/2) * Lx
        - y = (-Ly/2 -- +Ly/2) = (v - 1/2) * Ly
        - z = (-Lz/2 -- +Lz/2) = (w - 1/2) * Lz

    * `"raw"`: A normalized coordinate in the raw imaged volume (corresponds directly
        to a pixel in a volume).
        - u' = up = (0 -- 1)
        - v' = vp = (0 -- 1)
        - w' = wp = (0 -- 1)

    * `"corrected"`: A normalized coordinate in a distortion corrected space.
        - u = (0 -- 1)
        - v = (0 -- 1)
        - w = (0 -- 1)

    * `"index"`: The indices of the volume.  Note that the order of the indices
        is reversed, as in accessing the volume-array.  (e.g. vol[k, j, i])
        - k = (0 -- Nz-1)
        - j = (0 -- Ny-1)
        - i = (0 -- Nx-1)

    * `"index-xyz"`: The indices, in "natural" order.  This is what is
        returned by TrackPy, among other things.
        - i = (0 -- Nx-1)
        - j = (0 -- Ny-1)
        - k = (0 -- Nz-1)

    The main job of this class is to connect raw to corrected coordinates,
    which in general depends on the physical camera setup.

    The base model (refered to as "simple" in the glsl shader), assumes camera
    perspective and scanning angle distortion.

    Note: the GLSL shader fragment code for this model is
    "perspective_model_simple.glsl" in [MUVI SOURCE]/view/shaders
    '''

    NAME = 'simple'
    VARIABLES = {
        "distortion_correction_factor" : np.zeros(3, 'd'),
        "vol_N": np.ones(3, 'i'),
        "vol_L": np.ones(3, 'd'),
    }

    CORRECTED_SPACES = {"physical", "corrected"}
    RAW_SPACES = {"raw", "index", "index-xyz"}
    SPACES = CORRECTED_SPACES | RAW_SPACES

    def __init__(self, info):
        self.var = self.VARIABLES.copy()

        self.var['vol_N'] = np.array(info.get_list('Nx', 'Ny', 'Nz'), dtype='d')
        self.var['vol_L'] = np.array(info.get_list('Lx', 'Ly', 'Lz'), dtype='d')

        if 'Lx' in info and 'dx' in info:
            self.var["distortion_correction_factor"][0] = info['Lx'] / info['dx']
        if 'Ly' in info and 'dy' in info:
            self.var["distortion_correction_factor"][1] = info['Ly'] / info['dy']
        if 'Lz' in info and 'dz' in info:
            self.var["distortion_correction_factor"][2] = info['Lz'] / info['dz']

    def convert(self, X, input="index", output="physical"):
        '''Convert coordinates between spaces.

        Paramaters
        ----------
        X : (..., 3) shaped array like
            The input coordinates.

        Keywords
        --------
        input : str (default: "index")
            The input coordinate space
        output : str (default: "physical")
            The output coordinate space

        Returns
        -------
        X' : (..., 3) shaped array
            The output coordinates, coverted to the new space.
        '''

        if input not in self.SPACES:
            raise ValueError(f'input keyword should be one of {self.SPACES}')

        if output not in self.SPACES:
            raise ValueError(f'output keyword should be one of {self.SPACES}')

        X = np.asarray(X, 'd')
        if X.shape[-1] != 3:
            raise ValueError('The last axis of the input coordinates must have size 3')


        N = self.var['vol_N']
        L = self.var["vol_L"]

        if input in self.CORRECTED_SPACES:
            if input == "physical":
                U = X / L + 0.5
            else: # "corrected"
                U = X
            if output in self.RAW_SPACES:
                Up = self.corrected_to_raw(U)

        else:
            if input.startswith("index"):
                if input != "index-xyz":
                    X = X[::-1]
                Up = (X + 0.5) / N
            else: # "raw"
                Up = X
            if output in self.CORRECTED_SPACES:
                U = self.raw_to_corrected(Up)

        if output == "index":
            return (Up * N)[::-1] - 0.5
        elif output == "index-xyz":
            return (Up * N) - 0.5
        elif output == "raw":
            return Up
        elif output == "physical":
            return (U - 0.5) * L
        else: # "corrected"
            return U

    def raw_to_corrected(self, Up):
        '''Convert from raw to corrected coordinates.

        Parameters
        ----------
        Up: (..., 3) shaped array, or array like
            The raw coordinates.

        Returns
        -------
        U: (..., 3) shaped array
            The corrected coordinates.
        '''

        Up = np.asarray(Up, 'd')
        if Up.shape[-1] != 3:
            raise ValueError('The last axis of the input must have size 3')

        U = np.empty(Up.shape, 'd')

        dcf = self.var['distortion_correction_factor']
        eps = 0.25 * (dcf * (1 - 2*Up))
        eps_xy = eps[..., 0] + eps[..., 1]
        eps_z  = eps[..., 2]

        U[..., 0] = Up[..., 0] * (1 + 2*eps_z ) - eps_z  * (1 + 2*eps_xy)
        U[..., 1] = Up[..., 1] * (1 + 2*eps_z ) - eps_z  * (1 + 2*eps_xy)
        U[..., 2] = Up[..., 2] * (1 + 2*eps_xy) - eps_xy * (1 + 2*eps_z )
        U /= (1 - 4*eps_xy*eps_z)[..., np.newaxis]

        return U

    def corrected_to_raw(self, U):
        '''Convert from corrected to raw coordinates.

        Parameters
        ----------
        U: (..., 3) shaped array
            The corrected coordinates.

        Returns
        -------
        Up: (..., 3) shaped array, or array like
            The raw coordinates.
        '''

        U = np.asarray(U)
        if U.shape[-1] != 3:
            raise ValueError('The last axis of the input must have size 3')

        Up = np.empty(U.shape, 'd')

        dcf = self.var['distortion_correction_factor']
        eps = 0.25 * (dcf * (1 - 2*U))
        eps_xy = eps[..., 0] + eps[..., 1]
        eps_z  = eps[..., 2]

        Up[..., 0] = (U[..., 0] + eps_z ) / (1 + 2*eps_z )
        Up[..., 1] = (U[..., 1] + eps_z ) / (1 + 2*eps_z )
        Up[..., 2] = (U[..., 2] + eps_xy) / (1 + 2*eps_xy)

        return Up


distortion_models = {
    "simple": DistortionModel,
}

def get_distortion_model(info):
    '''Given VolumeProperties object, return an appropriate initialized
    DistortionModel object.

    Parameters
    ----------
    info : VolumeProperties object

    Returns
    -------
    model : DistortionModel or derived class

    **Note:** presently there is only one distortion model, so it only returns
    this.  In future iterations multiple models may be supported.
    '''
    return DistortionModel(info)

if __name__ == "__main__":
    import random
    from muvi import VolumeProperties

    info = VolumeProperties(
        Nx = 50,
        Ny = 75,
        Nz = 100,
        Lx = 20,
        Ly = 30,
        Lz = 40,
        dx = -15,
        dy = 10,
        dz = 10
    )

    dm = DistortionModel(info)
    spaces = list(DistortionModel.SPACES)

    # Stress test 1: convert back and forth between 2
    for n in range(10):
        seq = random.sample(spaces, k=2)
        X0 = np.random.rand(50, 3)

        if seq[0].startswith("index"):
            X0 *= dm.var['vol_N']
        elif seq[0] == "physical":
            X0 *= dm.var['vol_L']

        X = dm.convert(X0, seq[0], seq[1])
        X = dm.convert(X, seq[1], seq[0])

        print(f'{seq[0]:9s} <-> {seq[1]:9s}: {(X - X0).std()}')

    # Stress test 2: convert back and forth along a sequence of 3
    for n in range(20):
        seq = random.sample(spaces, k=3)
        X0 = np.random.rand(50, 3)

        if seq[0].startswith("index"):
            X0 *= dm.var['vol_N']
        elif seq[0] == "physical":
            X0 *= dm.var['vol_L']

        X = dm.convert(X0, seq[0], seq[1])
        X = dm.convert(X, seq[1], seq[2])
        X = dm.convert(X, seq[2], seq[0])

        print(f'{seq[0]:9s} -> {seq[1]:9s} -> {seq[2]:9s} -> {seq[0]:9s}: {(X - X0).std()}')
