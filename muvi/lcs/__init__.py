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
from scipy.interpolate import RBFInterpolator
import pandas as pd
#Developed modules
from muvi.geometry.resample import windowed_polynomial_resample
from muvi.lcs.particle_tracking_filters import adj_frames_filter

class FTLEs:
    '''
    A class used to compute FTLE fields.
    -------
    '''

    def __init__(self):
        pass
    

    def rk4_step(self, p0, func, t, dt):
        K1 = dt * func(p0, t)
        K2 = dt * func(p0 + 0.5*K1, t + 0.5*dt)
        K3 = dt * func(p0 + 0.5*K2, t + 0.5*dt)
        K4 = dt * func(p0 + K3, t + dt)

        return p0 + (1/6) * (K1 + 2*K2 + 2*K3 + K4)
    
    def RK4(self, p0, func, t, dt):
        sol = []
        for t_ in t:
            sol.append(p0)
            p0 = self.rk4_step(p0, func, t_, dt)
        
        return sol
    
    def fmc_step(self, traj, R, interpolator='rbf', **interp_params):
        interpolators = ['rbf', 'windowed_polynomial']
        if interpolator not in interpolators:
            raise ValueError(f"Select interpolator 'rbf' or 'windowed_polynomial'.")
        
        if interpolator == interpolators[0]:
            f_step = RBFInterpolator(traj[0], traj[1], **interp_params)(R.reshape(-1, R.shape[-1])).reshape(R.shape)
            return f_step
        
        else:
            #Using interpolator 'windowed polynomial' allows us to obtain the gradient for free
            f_step = windowed_polynomial_resample(R, traj[0], traj[1], **interp_params)[0]
            
            return f_step
    
    def FMC(self, data, R, interpolator='rbf', **interp_params):
        """
        -Flow map compiliation method to compute flow map. 
        -Supports particle tracking data, particles will dissapear and appear from frame to frame, through the 
        adj_frame_filter function.
        Args:
            R: Interpolation grid.
            data: DataFrame (typically resulting from trackpy) or data store as a numpy array.
            interpolator: Type of interpolator to use. Default is 'rbf'.
            **interp_params: Additional parameters for interpolation.
        Returns:
            Flow map or Gradient of flow map.
            Note: If interpolator is set to 'windowed polynomial' we obtain the GRADIENT of the FLOW MAP for FREE!
        """
        print(f"Computing flow maps using flow map compiliation (FMC)...")
        if  isinstance(data, pd.DataFrame):
            print(f"Processing with pandas DataFrame.")
            end_frame = data['frame'].max()
            start_frame = data['frame'].min()
            frames = np.arange(start_frame, end_frame)
            particle_tracking = True
            
        else:
            print("Processing with provided numpy array.")
            frames = np.arange(0, data.shape[0])
            particle_tracking = False


        f0 = np.expand_dims(R, axis=0)
        #f = np.zeros(((len(frames) - 1,) + R.shape))
        for i, frame in enumerate(frames[:-1]): #frames[:-1] because fmc_step uses the i and the i+1 entry
            print(f"Flow map: {i+1}/{len(frames)-1}...")

            if particle_tracking == True:
                traj = adj_frames_filter(data, frame)
                result = self.fmc_step(traj, R, interpolator, **interp_params)
                
            else:
                traj = data
                result = self.fmc_step(traj[i:i+2], R, interpolator, **interp_params)

            if interpolator == 'windowed_polynomial':
                f_step = result #G is the derivative of the windowed polynomial fit
                nan_mask = np.isnan(f_step)
                print(f"Number of NaN values in f_step: {np.sum(nan_mask)}")
                f_step[nan_mask] = 0

            else:
                f_step = result
            
            R = f_step   
            #f[i, ...] = f_step
        
        return np.array([f0, f_step])
        
    
    
    def vfi_step(self, pos_traj, vel_traj, R, dt, interpolator ='rbf', **interp_params):
        interpolators = ['rbf', 'windowed_polynomial']
        if interpolator not in interpolators:
            raise ValueError(f"Select interpolator 'rbf' or 'windowed_polynomial'.")
    
        if interpolator == interpolators[0]:
            K1 = dt * RBFInterpolator(pos_traj[0, ...], vel_traj[0, ...], **interp_params)(R.reshape(-1, 2)).reshape(R.shape)
            K2 = dt * RBFInterpolator(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K1).reshape(-1, 2)).reshape(R.shape)
            K3 = dt * RBFInterpolator(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K2).reshape(-1, 2)).reshape(R.shape)
            K4 = dt * RBFInterpolator(pos_traj[2,...], vel_traj[2,...], **interp_params)((R + 2*dt*K3).reshape(-1, 2)).reshape(R.shape)
        
        else:
            K1 = dt * windowed_polynomial_resample(R, pos_traj[0], vel_traj[0], **interp_params)[0]
            K2 = dt * windowed_polynomial_resample(R + dt*K1, pos_traj[1,...], vel_traj[1,...], **interp_params)[0]
            K3 = dt * windowed_polynomial_resample(R + dt*K2, pos_traj[1,...], vel_traj[1,...], **interp_params)[0]
            K4 = dt * windowed_polynomial_resample(R + 2*dt*K3, pos_traj[2,...], vel_traj[2,...], **interp_params)[0]
        
        f_step =  (1/6) * (K1 + 2*K2 + 2*K3 + K4) + R
        
        return f_step 
        
    def VFI(self, pos_traj, vel_traj, R, dt, interpolator='rbf', **interp_params):
        """
        -Velocity field integration method to compute flow map. 
        -Does NOT support particle tracking data (more speed efficient to use FMC) because particle tracking velocimetry (PTV)
        is needed to resolve the velocities of particles (see muvi.geometry.trajectories for more details) which increases computational cost.
        -
        Args:
            R: Interpolation grid.
            dt: Frame rate.
            pos_traj: Position trajectories of particles.
            vel_traj: Velocity trajectories of particles.
            interpolator: Type of interpolator to use. Default is 'rbf'.
            **interp_params: Additional parameters for interpolation.

        Returns:
            Flow map
        """
        print(f"Computing flow maps using velocity field integration (VFI)...")

        frames = np.arange(0, pos_traj.shape[0])

        f0 = np.expand_dims(R, axis=0)
        f = np.zeros(((len(frames)-3,) + R.shape)) #len(frames)-3 because vfi_step uses the i, the i+1 and i+2

        for i in frames[:-3]:
            print(f"Flow map: {i+1}/{len(frames)-3}...")
            f_step = self.vfi_step(pos_traj[i:i+3], vel_traj[i:i+3], R, dt, interpolator, **interp_params)
            R = f_step   
            f[i, ...] = f_step
                
        return np.array([f0, f])
    
    
    def grad_step(self, Ft0, FT):
        """
        -Computes gradient of flow map for arbitary number of dimensions
        Args:
            Ft0: Flow map at time step t0.
            FT: Flow map at time step T.

        Returns:
            Gradient of flow map
        """
        print(f"Computing gradient of flow map...")

        G = np.zeros((Ft0.shape + (Ft0.shape[-1],)))
        #Compute number of dimension
        num_dims = Ft0.shape[-1]
        #Compute gradient along each dimension
        grad_Ft0 = np.gradient(Ft0, axis=tuple(np.arange(num_dims)))
        grad_FT = np.gradient(FT, axis=tuple(np.arange(num_dims)))

        g = []
        
        for k in range(num_dims):
            for i, j in zip(np.arange(num_dims), np.flip(np.arange(num_dims))):
                #The following returns [delta_x1/delta_X1, delta_x2/delta_X1, delta_x1/delta_X2, delta_x2/delta_X2]
                g.append(grad_FT[j][..., i]/grad_Ft0[(num_dims-1) - k][..., k])

        #Store gradient of the flow field, and stores it as follows [[delta_x1/delta_X1, delta_x2/delta_X1],[delta_x1/delta_X2, delta_x2/delta_X2]]
        for i in range(num_dims):
            for j in range(num_dims):
                G[..., i, j] = g[i+j]
        
        return G
    

    def ftle_field(self, G, T):
        """
        -Computes FTLE field
        Args:
            G: Gradient of the flow map.
            T: Final time.
        Returns:
            Finite time Lyapnuov exponent (FTLE) field
        """
        print(f"Computing FTLE field...")
        from numpy import linalg as LA
        spatial_shape = G.shape[:-2]  #Get the spatial shape of the field (excluding the Jacobian matrix dimensions)
        ftles_fwd = np.zeros(spatial_shape)
        ftles_bwd = np.zeros(spatial_shape)
        for index in np.ndindex(spatial_shape):
            G_jk = G[index]  #Extract the Jacobian matrix at this spatial location
            C = np.einsum('ij...,jk...->ik...', G_jk.T, G_jk)  #Compute Cauchy-Green tensor
            eigenvalues = LA.eigvals(C)
            max_eigenvalue = np.max(np.abs(eigenvalues))
            min_eigenvalue = np.min(np.abs(eigenvalues))

            ftles_fwd[index] = (1 / np.abs(T)) * np.log(np.sqrt(max_eigenvalue))
            ftles_bwd[index] = -(1 / np.abs(T)) * np.log(np.sqrt(min_eigenvalue))

        return [ftles_fwd, ftles_bwd]
    
#start from the grid and computer the flow map from the grid to 0 to h and then start off again and computer it from h to 2h