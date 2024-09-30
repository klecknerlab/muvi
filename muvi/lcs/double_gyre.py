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
# limitations under the License.
#
# DESCRIPTION:
# This script computes and visualizes Finite-Time Lyapunov Exponent (FTLE) fields for a double gyre flow model
# using a specified interpolation method ('VFI'/'FMC' with windowed polynomial/rbf interpolation). It initializes a random 
# set of particles, computes their trajectories under the double gyre flow, and calculates the forward and backward 
# FTLE fields. The results are plotted as 2D slices, and root mean square (RMS) errors are computed to evaluate 
# the accuracy of the interpolated FTLE fields.
#
# Workflow:
# 1. Initialize parameters, including time steps, number of particles, and interpolation settings.
# 2. Compute the exact FTLE fields for a defined interpolation grid.
# 3. Randomly generate particle positions.
# 4. Compute the interpolated FTLE fields using Forward and Backward Mapping with interpolator.
# 5. Calculate the RMS error between the exact and interpolated FTLE fields.
# 6. Plot the computed FTLE fields and RMS error values.
# 7. Measure and print the elapsed computation time.
#
# Main Functions:
# - `exact_solution(R)`: Computes the exact FTLE fields based on particle trajectories using the RK4 integrator.
# - `interp_solution(X0, R)`: Computes the interpolated FTLE fields using randomly spaced particles and interpolation.
# - `rms_step(R)`: Calculates the RMS error between the exact and interpolated FTLE fields.
# - `compute_rms(particles, N)`: Plots the RMS error for varying particle counts and grid resolutions.
# - `main()`: Coordinates the overall workflow, handles plotting, and tracks computation time.

#Imported modules
import numpy as np 
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import time
import os
#Developed modules
from muvi.lcs import FTLEs
from muvi.lcs.flow_models import double_gyre
from muvi.lcs.visualization import plot_2D

#Initialize time, resolution and particle count parameters:
T = 10
delta_t = [0.01]
N = [(128, 56), (256, 128), (512, 256), (1024, 512)]
particles = [500, 1000, 1500, 2000, 2500, 5000, 75000, 10000, 15000, 20000, 25000, 50000]
#Specify method for flow map computation:
method = 'FMC'
#Initialize interpolators:
interpolator = ['rbf', 'windowed_polynomial']
σ_m = 1 #place holder for avg spacing between particles in field, gets updated downstream
interp_params = [dict(kernel = 'thin_plate_spline', smoothing = 2, neighbors = 50), dict(cutoff = σ_m * 3.0, order = 2)]
#Spefcify path where figures will be saved to:
out_dir = 'error_estimate_particles'

class DoubleGyre:
    """
    A class to compute Finite-Time Lyapunov Exponent (FTLE) fields for a double gyre flow.
    
    Attributes:
    - T: Final time for integration.
    - out_dir: Directory to save output plots.
    - interpolator: Interpolation method used ('windowed polynomial' or others).
    - interp_params: Additional parameters for the interpolator.
    - bounds: Domain bounds for the double gyre (default x = 2, y = 1).
    """
    def __init__(self, method, T, out_dir, interpolator, **interp_params):
        self.out_dir = out_dir
        self.interpolator = interpolator
        self.interp_params = interp_params
        self.bounds = (2, 1) #bounds are set to x = 2, y = 1, because double gyre is confined to this region
        self.T = T
        self.method = method
    
    def exact_solution(self, R, rk4_dt=10**-1):
        """
        Computes the exact FTLE field using particle trajectories and RK4 integration.
        
        Parameters:
        - R: Grid coordinates over which FTLE will be computed.
        
        Returns:
        - LCS: A list of forward and backward FTLE fields.
        """
        filename = os.path.join(self.plots_path, f"{self.interpolator}_exact_lcs.npy")
        if os.path.exists(filename):
            print(f"File '{filename}' exists. Loading data.")
            LCS = np.load(filename, allow_pickle=True)

        else:     
            print(f"File '{filename}' not found. Computing FTLEs.")
            ftle = FTLEs()            
            t = np.arange(0, self.T + rk4_dt, rk4_dt)
            traj_first = R
            X = R.reshape(-1, 2)
            for t_ in (t):
                traj_step = ftle.rk4_step(X, double_gyre, t_, rk4_dt)
                X = traj_step
            
            traj = np.array([traj_first, traj_step.reshape(R.shape)])
            
            ofn = 'exact_flowmap'
            plot_2D(R, traj[-1][...,0], ofn, self.plots_path)
            
            G = ftle.grad_step(traj[0], traj[-1])
            LCS = ftle.ftle_field(G, self.T)

            ofn = ['exact_fwd_ftle', 'exact_bwd_ftle']
            [plot_2D(R, LCS[i], ofn[i], self.plots_path) for i in range(len(ofn))]

            np.save(filename, LCS)
            print(f"Saved LCS to '{filename}'.")

        return LCS
    
    def interp_solution(self, X0, R):
        """
        Computes the interpolated FTLE field using the chosen interpolation method.
        
        Parameters:
        - X0: Randomly spaced particle coordinates.
        - R: Grid coordinates over which FTLE will be computed.
        
        Returns:
        - LCS: A list of forward and backward FTLE fields using interpolated trajectories.
        """
        filename = os.path.join(self.plots_path, f"{self.interpolator}_interp_lcs_{self.ext}.npy")
        if os.path.exists(filename):
            print(f"File '{filename}' exists. Loading data.")
            LCS = np.load(filename, allow_pickle=True)
        
        else:
            print(f"File '{filename}' not found. Computing LCS.")
            t = np.arange(0, self.T + self.dt, self.dt)
            ftle = FTLEs()

            if self.interpolator == 'windowed_polynomial':
                σ_m = ((self.bounds[0]*self.bounds[1])/(self.num_particles))**(1/2)
                self.interp_params['cutoff'] = σ_m * 3
                print(f"σ_m: {σ_m}")

            R_init = R.copy()
            for i, t_ in enumerate(t[:-1]):
                print(f"Computing flow map: {i+1}/{len(t)-1}")
                p0 = ftle.rk4_step(X0, double_gyre, t_, self.dt)
                f_step = ftle.fmc_step(np.array([X0, p0]), R, self.interpolator, **self.interp_params)
                
                if self.interpolator != 'rbf':
                    nan_mask = np.isnan(f_step)
                    print(f"interrrr")
                    print(f"Number of NaN values in f_step: {np.sum(nan_mask)}")
                    f_step[nan_mask] = 0
                
                R = f_step
                X0 = p0
            
            F = np.array([R_init, f_step])
            print(f"F.shape: {F.shape}")
            G = ftle.grad_step(F[0], F[-1])
            LCS = ftle.ftle_field(G, self.T)

            #traj = ftle.RK4(X0, double_gyre, t, self.dt)
            #F = ftle.FMC(traj, R, self.interpolator, **self.interp_params)

            #G = ftle.grad_step(F[0], F[-1])
            #LCS = ftle.ftle_field(G, self.T)


            ofn = [self.interpolator + '_fwd_ftle_' + self.ext,
                    self.interpolator + '_bwd_ftle_' + self.ext]
            
            [plot_2D(R_init, LCS[i], ofn[i], self.plots_path) for i in range(len(ofn))]

            np.save(filename, LCS)
            print(f"Saved LCS to '{filename}'.")

        return LCS
    
    def rms_step(self, R):
        """
        Computes the RMS error between the exact and interpolated FTLE fields.
        
        Parameters:
        - R: Grid coordinates over which FTLE is computed.
        
        Returns:
        - RMS error value.
        """
        X0 = np.random.rand(self.num_particles, 2) *self.bounds
        LCS_exact = self.exact_solution(R)
        LCS_interp = self.interp_solution(X0, R)
        L = LCS_exact[0] - LCS_interp[0]
        L_finite = L[np.isfinite(L)]
        
        return np.sqrt(np.mean(L_finite**2))
    
    def compute_rms(self, delta_t, particles, N):
        """
        Computes and plots the RMS error for different particle counts and grid resolutions.
        
        Parameters:
        - particles: List of particle counts to test.
        - N: List of grid resolutions (Nx, Ny) to compute the FTLE.
        """
        self.len_delta_t = len(delta_t)
        self.len_particles = len(particles)
        
        if not ((len(delta_t) == 1 and  self.len_particles > 1) or (self.len_delta_t > 1 and  self.len_particles == 1)):
            raise ValueError("Invalid input: Exactly one of delta_t or particles must have length 1 for plotting purposes (RMS vs delta_t or RMS vs particles).")

        for Nx, Ny in N:
            resolution = self.method + f'Nx{Nx}_Ny{Ny}'
            self.plots_path = os.path.join(self.out_dir, resolution)
            os.makedirs(self.plots_path, exist_ok=True)
            #Generate grid
            xx, yy = np.meshgrid(np.linspace(0, self.bounds[0], Nx, endpoint=True), np.linspace(0, self.bounds[1], Ny, endpoint=True))
            R = np.stack((xx, yy), axis=-1) 
            
            rms_ls = []  #Reset this list for each (Nx, Ny) combination
            if self.len_delta_t > 1:
                self.num_particles = particles[0]
                x_label = 'dt'
                x_data = delta_t
                
                for self.dt in delta_t:
                    self.ext = 'dt' + str(self.dt).replace('.', '_')
                    rms_ls.append(self.rms_step(R))
   
            else:
                x_label = 'particle count'
                x_data = particles

                for self.num_particles in particles:
                    self.ext = 'particles' + str(self.num_particles)
                    rms_ls.append(self.rms_step(R))
                    
            #Plot the results for the current grid size (Nx, Ny)
            plt.loglog(x_data, rms_ls, label=f'{Nx}x{Ny}')  # log-log plot

        plt.xlabel(x_label)
        plt.ylabel('RMS Error') 
        plt.legend()  
        
        plt.savefig(os.path.join(self.out_dir, self.interpolator + "_rms_error" + ".png"), dpi=1200, bbox_inches='tight', pad_inches=0)
        plt.close()  

def main():
    """
    Main function to compute and plot the RMS errors for multiple grid resolutions and particle counts.
    """
    start_time = time.time()
    for i in range(len(interpolator)):
        obj = DoubleGyre(method, T, out_dir, interpolator[i], **interp_params[i])
        obj.compute_rms(delta_t, particles, N)

    end_time = time.time()
    elapsed_time_seconds = end_time - start_time

    print(f"Computing time: {elapsed_time_seconds:.2f} s, "
          f"{elapsed_time_seconds / 60:.2f} min, "
          f"{elapsed_time_seconds / 3600:.2f} hr.")

if __name__ == "__main__":
    main()