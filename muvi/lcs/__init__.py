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

class FTLEs:
    '''
    A class used to compute FTLE fields.
    -------
    '''

    def __init__(self):
        pass
    
    def adj_frames_filter(self, data, start_frame):
        """
        -Filters data (either pandas data frame) keeping only particles that appear in two adjacent frames.
        This is used particularly for particle tracking data because particles can dissapear and reappear between frames.

        Args:
            data: pandas data frame (e.g. from trackpy). 
            start_frame: starting frame for filtering.
        Returns:
            Filtered trajectories
        """
        if isinstance(data, pd.DataFrame):
            #Determine end_frame 
            end_frame = start_frame + 1

            #Define frames to be considered
            frames = np.arange(start_frame, end_frame + 1)
            filtered_data = data[data['frame'].isin(frames)]
            df_adj_frames = filtered_data[filtered_data['frame'].isin(frames)]

            #Identify particles and their frame counts
            particle_frame_counts = df_adj_frames.groupby('particle')['frame'].nunique()
            valid_particles = particle_frame_counts[particle_frame_counts == len(frames)].index

            #Filter duplicates to keep only those particles appearing in all frames
            duplicates = df_adj_frames[df_adj_frames['particle'].isin(valid_particles)]

            corrected_coords = ['xc', 'yc', 'zc']
            uncorrected_coords = ['x', 'y', 'z']

            #Find existing corrected coordinates
            existing_corrected = [coord for coord in corrected_coords if coord in data.columns]

            #If at least one corrected coordinate is found, use it
            if existing_corrected:
                coords = existing_corrected
            else:
                # Find existing uncorrected coordinates if no corrected coordinates are found
                existing_uncorrected = [coord for coord in uncorrected_coords if coord in data.columns]
                if existing_uncorrected:
                    coords = existing_uncorrected
                else:
                    raise ValueError("Neither corrected ('xc', 'yc', 'zc') nor uncorrected ('x', 'y', 'z') coordinates are present in the DataFrame.")
            
            num_dims = len(coords)

            if num_dims <2:
                raise ValueError(f"num_dims must be either 2 or 3. Current num_dims: {num_dims}. With coordinates: {coords}")

            #Check if preferred coordinates are present
            coords = corrected_coords if all(coord in data.columns for coord in corrected_coords) else uncorrected_coords

            #Extract and sort trajectories
            particles0 = duplicates[duplicates['frame'] == start_frame]['particle']
            particles1 = duplicates[duplicates['frame'] == start_frame + 1]['particle']
            traj0 = duplicates[duplicates['frame'] == start_frame][coords].to_numpy()
            traj1 = duplicates[duplicates['frame'] == start_frame + 1][coords].to_numpy()

            sorted_indices0 = particles0.argsort()
            sorted_indices1 = particles1.argsort()

            traj0 = traj0[sorted_indices0]
            traj1 = traj1[sorted_indices1]

        else:
            raise ValueError(f"data must be a pandas data frame which results from particle tracking (e.g. trackpy).")

        return np.stack((traj0, traj1), axis=0)
    
    def rk4_step(self, p0, func, t, dt):
        K1 = dt * func(p0, t)
        K2 = dt * func(p0 + 0.5*K1, t + 0.5*dt)
        K3 = dt * func(p0 + 0.5*K2, t + 0.5*dt)
        K4 = dt * func(p0 + K3, t + dt)

        return p0 + (1/6) * (K1 + 2*K2 + 2*K3 + K4)
    
    def RK4(self, p0, func, t, dt):
        sol = np.zeros((len(t),) + p0.shape)

        for i, t in enumerate(t):
            sol[i] = p0
            p0 = self.rk4_step(p0, func, t, dt)

        return sol
    
    def fmc_step(self, traj, R, interpolator = 'rbf', **interp_params):
        #Interpolate between the small time flow maps, f, onto the discrete grid denoted by R.
        if interpolator == 'rbf':
            f_step = RBFInterpolator(traj[0], traj[1], **interp_params)(R.reshape(-1, R.shape[-1])).reshape(R.shape)
        
        elif interpolator == 'windowed polynomial':
            f_step = windowed_polynomial_resample(traj[0], traj[1], **interp_params)(R.reshape(-1, R.shape[-1])).reshape(R.shape)
        
        else:
            print("Please choose interpolator 'rbf' or 'windowed polynomial'.")
            
        return f_step
    
    def FMC(self, R, pickle_fn=None, traj=None, interpolator='rbf', **interp_params):
        """
        -Flow map compiliation method to compute flow map. 
        -Supports particle tracking data, particles will dissapear and appear from frame to frame, through the 
        adj_frame_filter function.
        Args:
            R: Interpolation grid.
            pickle_fn (optional): Path to the pickle file which has data stored in a dataframe. Default is None.
            traj (optional): Trajectory data. Default is None.
            interpolator: Type of interpolator to use. Default is 'rbf'.
            **interp_params: Additional parameters for interpolation.
        Returns:
            Flow map
        """
        print(f"Computing flow maps using flow map compiliation (FMC)...")
        
        if pickle_fn is not None:
            data = pd.read_pickle(pickle_fn)
            end_frame = data['frame'].nunique()
            
            print(f"Processing with pickle file: {pickle_fn}")
            frames = np.arange(0, end_frame)

            particle_tracking = True

        else:
            print("Processing with provided trajectory data.")
            data = traj
            frames = np.arange(0, data.shape[0])
            particle_tracking = False


        f0 = np.expand_dims(R, axis=0)
        f = np.zeros(((len(frames) - 1,) + R.shape))
        
        for i, frame in enumerate(frames[:-1]): #frames[:-1] because fmc_step uses the i and the i+1 entry
            print(f"Flowmap: {i+1}/{len(frames)-1}...")

            if particle_tracking == True:
                traj = self.adj_frames_filter(data, frame)
                f_step = self.fmc_step(traj, R, interpolator, **interp_params)

            else:
                f_step = self.fmc_step(traj[i:i+2], R, interpolator, **interp_params)
            
            R = f_step   
            f[i, ...] = f_step
        
        print(f"F shape: {np.vstack((f0, f)).shape}")

        return np.vstack((f0, f))
    
    def vfi_step(self, pos_traj, vel_traj, R, dt, interpolator ='rbf', **interp_params):
        if interpolator == 'rbf':
            K1 = dt * RBFInterpolator(pos_traj[0, ...], vel_traj[0, ...], **interp_params)(R.reshape(-1, 2)).reshape(R.shape)
            K2 = dt * RBFInterpolator(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K1).reshape(-1, 2)).reshape(R.shape)
            K3 = dt * RBFInterpolator(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K2).reshape(-1, 2)).reshape(R.shape)
            K4 = dt * RBFInterpolator(pos_traj[2,...], vel_traj[2,...], **interp_params)((R + 2*dt*K3).reshape(-1, 2)).reshape(R.shape)
        
        elif interpolator == 'windowed polynomial':
            K1 = dt * windowed_polynomial_resample(pos_traj[0], vel_traj[0], **interp_params)(R.reshape(-1, R.shape[-1])).reshape(R.shape)
            K2 = dt * windowed_polynomial_resample(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K1).reshape(-1, 2)).reshape(R.shape)
            K3 = dt * windowed_polynomial_resample(pos_traj[1,...], vel_traj[1,...], **interp_params)((R + dt*K2).reshape(-1, 2)).reshape(R.shape)
            K4 = dt * windowed_polynomial_resample(pos_traj[2,...], vel_traj[2,...], **interp_params)((R + 2*dt*K3).reshape(-1, 2)).reshape(R.shape)

        else:
            print("Please choose interpolator 'rbf' or 'windowed polynomial'.")
        
        f_step =  (1/6) * (K1 + 2*K2 + 2*K3 + K4) + R
        
        return f_step 
        
    def VFI(self, R, dt, pos_traj, vel_traj, interpolator='rbf', **interp_params):
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
            print(f"Flowmap: {i+1}/{len(frames)-3}...")
            f_step = self.vfi_step(pos_traj[i:i+3], vel_traj[i:i+3], R, dt, interpolator, **interp_params)
            R = f_step   
            f[i, ...] = f_step
        
        print(f"F shape: {np.vstack((f0, f)).shape}")
        
        return np.vstack((f0, f))
    
    
    def grad_flowmap(self, F):
        """
        -Computes gradient of flow map for arbitary number of dimensions
        Args:
            F: Flow map.

        Returns:
            Gradient of flow map
        """
        print(f"Computing gradient of flow map")
        #Initialize flow map, F from 0 to T.
        Ft0 = F[0]
        FT = F[-1]
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

        #Store Jacobian of the flow field, and stores it as follows [[delta_x1/delta_X1, delta_x2/delta_X1],[delta_x1/delta_X2, delta_x2/delta_X2]]
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
    
