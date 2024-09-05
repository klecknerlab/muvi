#!/usr/bin/python3
#
# Copyright 2024 Diego Tapia Silva 
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
# limitations under the License

#Imported modules
import numpy as np
import time 
#Developed modules
from muvi.lcs import FTLEs
from muvi.lcs.flow_models import double_gyre
from muvi.lcs.visualization import plot_2D

#Initialize parameters:
T = 10
dt = 1/10
num_particles = 500
scaling_factors = np.array([2, 1])
X0 = np.random.rand(num_particles, 2) * scaling_factors
Nx, Ny = 201, 101
xx, yy = np.meshgrid(np.linspace(0, scaling_factors[0], Nx), np.linspace(0, scaling_factors[1], Ny))
R = np.stack((xx, yy), axis = -1)
interpolator = 'rbf'
interp_params = dict(kernel='cubic', smoothing=2, neighbors=20)
method = 'FMC'

def main():
    start_time = time.time()
    LCS = compute_ftle(T, dt, X0, R, interpolator, method, **interp_params)
    plot_2D(R, LCS, out_dir = 'double_gyre_' + method)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    print(f"Computing time: {elapsed_time_seconds:.2f} s, "
          f"{elapsed_time_seconds / 60:.2f} min, "
          f"{elapsed_time_seconds / 3600:.2f} hr.")



def compute_ftle(T, dt, X0, R, interpolator, method = 'FMC', **interp_params):
    """
    Computes the forward and backward FTLE field.

    Parameters:
    - T: Final time.
    - dt: Time step.
    - X0: Coordinates of randomly spaced particles.
    - R: Regular grid to interpolate on.
    - interpolator: Interpolating function.
    - method: velocity field integration 'VFI' or flow map compiliation 'FMC'
    - interp_params: Interpolating parameters.
    """
    t = np.arange(0, T + dt, dt)

    if method == 'VFI':
        #Appending to t guarantees that flow map will be computed from 0 to T (VFI method uses RK4)
        t = np.append(t, np.array([T+dt, T+2*dt]))
        traj = FTLEs().RK4(X0, double_gyre, t, dt)

        vel = np.zeros_like(traj)
        for i, time_step in enumerate(t):
            vel[i, ...] = double_gyre(traj[i, ...], time_step)

        F = FTLEs().VFI(R, dt, traj, vel, interpolator, **interp_params)
    
    elif method == 'FMC':
        traj = FTLEs().RK4(X0, double_gyre, t, dt)
        F = FTLEs().FMC(R=R, traj=traj, interpolator=interpolator, **interp_params)
    
    else:
        raise ValueError(f"Select method 'VFI' or 'FMC'.")
    
    G = FTLEs().grad_flowmap(F)

    return FTLEs().ftle_field(G, T)

if __name__ == "__main__":
    main()