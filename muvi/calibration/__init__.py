
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
#Developed modules
from muvi.distortion import get_distortion_model
from muvi import VolumeProperties
from muvi import open_3D_movie
from muvi.readers.cine import Cine
#Imported modules
from scipy import optimize
from scipy.spatial.transform import Rotation as rot
from scipy.interpolate import CubicSpline
import os
import matplotlib.pyplot as plt
import pandas as pd
import trackpy as tp
import pickle
import numpy as np
import json


class CalibrationProperties:
    _param_types = {
        'p1_n': float,
        'p2_n': float,
        'A1': float,
        'A2': float,
        'C1': float,
        'C2': float,
        'de': float,
        'channel': int,
        'start': int,
        'vols': int,
        'diameter': int,
        'separation': int,
        'mass': int,
        'search': int,
        'dx_f': float,
        'dz_f': float
    }
    
    def __init__(self, json_data=None, json_file=None):
        if json_data is not None:
            if isinstance(json_data, dict):
                self.json_data = json.dumps(json_data)
            else:
                self.json_data = json_data
        elif json_file is not None:
            with open(json_file, 'r') as file:
                self.json_data = file.read()
        else:
            self.json_data = json.dumps({})  # Empty JSON string

        self.parse_json()
        self.initialize_defaults()

    def parse_json(self):
        data = json.loads(self.json_data)
        properties = data.get("CalibrationProperties", {})
        
        # Check if each parameter is valid according to _param_types
        for param, value in properties.items():
            if param not in self._param_types:
                raise ValueError(f"{param} is not a valid parameter.")
            
            # Enforce data types
            setattr(self, param, self._param_types[param](value))

    def update_from_file(self, json_file):
        with open(json_file, 'r') as file:
            self.json_data = file.read()
        self.parse_json()

    def initialize_defaults(self):
        # Initialize default values for parameters if they are not already set
        for param, param_type in self._param_types.items():
            if not hasattr(self, param):
                setattr(self, param, param_type())

    def __getitem__(self, key):
        return getattr(self, key)
    
    def __contains__(self, key):
        return hasattr(self, key)

class TrackingModel:
    '''
    A class used for tracking of VTI file.

    Methods
    -------
    '''
    _DEFAULT_CAL = {
    'start': 0,
    'diameter': 5,
    'separation': 5,
    'mass': 1.3,
    'search': 1,
    }
    def __init__(self, vti, setup_xml, setup_json):
        #Setting VolumeProperties
        self.muvi_info = VolumeProperties()
        self.muvi_info.update_from_file(setup_xml)
        #Setting CalibrationProperties
        self.cal_info = CalibrationProperties(json_data=self._DEFAULT_CAL)
        print(f"start, {self.cal_info['start']}")
        self.cal_info.update_from_file(setup_json)
        #Creating new directory to store tracks
        parent_dir = os.path.dirname(vti)
        vti_dir = os.path.join(parent_dir, os.path.splitext(vti)[0])
        self.new_path = os.path.join(vti_dir, f"channel{self.cal_info['channel']}")
        if not os.path.exists(self.new_path):
            os.makedirs(self.new_path)
        
        self.vti = vti

    def vti_tracks(self,):
        print(f"Analyzing tracks for Channel {self.cal_info['channel']}...")
        vm = open_3D_movie(self.vti)
        
        if 'vols' in self.cal_info:
            #Ensuring that the end frame cannot exceed the end frame in the VTI movie
            if self.cal_info['vols'] + self.cal_info['start'] < len(vm):
                end = self.cal_info['vols'] + self.cal_info['start'] - 1 
            else: 
                end = len(vm) - 1
        else:
            end = len(vm) - 1
        
        print(f"Start Frame: {self.cal_info['start']}, End Frame: {end}")
        
        #Grab vti frames and dividing the VTI movie into 'chunks' for memory purposes
        chunk_size = 10
        vol_divisons = int(np.ceil(((end + 1) - self.cal_info['start'])/(chunk_size-1)))
        print(f"Splitting VTI movie into {vol_divisons} divisons...")
        particle_location_df = pd.DataFrame()
        for chunk in range (vol_divisons):
            print(f"Computing tracks for {chunk+1}/{vol_divisons} divisons...")
            vol_ls = []        
            #Setting the end frame of the final chunk
            if chunk == vol_divisons-1:
                #Ensuring that the end frame cannot exceed the end frame in the VTI movie
                if 'vols' in self.cal_info:
                    if self.cal_info['start'] + self.cal_info['vols'] < len(vm):
                        end_frame = self.cal_info['start'] + self.cal_info['vols']
                    else:
                        end_frame = len(vm)
                else:
                    end_frame = len(vm) 
            else:
                end_frame = self.cal_info['start'] + ((chunk + 1)*(chunk_size-1)) + 1
           
            for frame in range(self.cal_info['start'] + (chunk*(chunk_size-1)), end_frame):
                print(f"Frame {frame}")
                vol = (vm[frame].astype(float)/255)**self.muvi_info['gamma']
                if self.muvi_info['channels'] == 1:
                    vol_ls.append(vol[:, :, :])
                else:
                    vol_ls.append(vol[:, :, :, self.cal_info['channel']])
                    
            particle_location = tp.batch(vol_ls, diameter = self.cal_info['diameter'], 
                                         separation = self.cal_info['separation'], minmass = self.cal_info['mass'])
            
            #Correct for frame number 
            particle_location['frame'] += self.cal_info['start'] + (chunk*(chunk_size-1))
            
            #After each chunk exclude the first row, this is because the end frame of chunk i is the start frame of chunk i+1
            if chunk != 0:
                first_frame = particle_location['frame'].iloc[0]
                particle_location = particle_location[particle_location['frame'] != first_frame]
                #print(f"Rows deleted with corresponding frame: {first_frame} to avoid repetition.")
            else:
                test_frame = vol_ls[0][int(self.muvi_info['Lx']) - 100:int(self.muvi_info['Lx']) + 100, 
                       int(self.muvi_info['Ly']) - 100:int(self.muvi_info['Ly']) + 100, 
                       int(self.muvi_info['Lz']) - 100:int(self.muvi_info['Lz']) + 100].sum(-1)
                
                plt.figure(figsize=(10,10))
                particles = tp.locate(test_frame, diameter = self.cal_info['diameter'], separation = self.cal_info['separation'])
                plt.plot(particles['x'],particles['y'], 'r.')
                plt.savefig(os.path.join(self.new_path, 'muvi_track.png'))
                plt.close()
                print(f"Output tracking plot {os.path.join(self.new_path, 'muvi_track.png')}")
          
            
            #Concantenating pandas dataframes into a single dataframe
            particle_location_df = pd.concat([particle_location_df, particle_location])

        #Linking trajectories
        print(f"Linking trajectories for {chunk+1}/{vol_divisons} divisons...")
        traj = tp.link(particle_location_df, search_range = self.cal_info['search'])
        print(f"Succesfully linked trajectories, start frame: {traj['frame'].iloc[0]} end frame: {traj['frame'].iloc[-1]}.")

        #Plotting linked particles by size vs mass
        fig = plt.figure()
        ax = fig.add_subplot(111)  
        tp.mass_size(traj.groupby('particle').mean(), ax=ax)
        plt.savefig(os.path.join(self.new_path, 'trackpy_size_mass.png'), bbox_inches='tight')
        plt.close(fig)  # Close the figure to suppress it from popping up
        print(f"Output tracking plot {os.path.join(self.new_path, 'trackpy_size_mass.png')}")
       
        #Filtering dataframe based on particle mass and size
        mass_bound = (0, 1)
        size_bound = (1, 2.25)
        traj_filtered = traj[
            (traj['mass'] > mass_bound[0]) &
            (traj['mass'] < mass_bound[1]) &
            (traj['size'] > size_bound[0]) &
            (traj['size'] < size_bound[1])
        ]
        print('Particle count before filtering:', traj['particle'].nunique())
        print('Particle count after filtering:', traj_filtered['particle'].nunique())
        
        #Updating coordinates from pixel coordinates to physical coordinates
        vm.distortion.update_data_frame(traj_filtered, columns =('x','y','z'), output_columns=('xc','yc','zc')) 
        
       # with open(os.path.join(self.new_path, 'muvi_track.pickle'), "wb") as f:
       #     pickle.dump(traj_filtered, f)

        #print(f"Output file {os.path.join(self.new_path, 'muvi_track.pickle')}")




class IntensityModel:
    '''
    A class used for intensity correction.

    Methods
    -------
    '''
    _DEFAULT_VOL = {
    'flip_y': True,
    }
    _DEFAULT_CALIBRATION = {
    'A1': 0.03, #(mm^-1)
    'A2': 0.01, #(mm^-1)
    'p1_n': 0.04,
    'p2_n': 0.03,
    'de': 4.5,
    }

    def __init__(self, channel, cine, setup_xml):
        self.cine = cine
        parent_dir = os.path.dirname(self.cine)
        cine_dir = os.path.join(parent_dir, os.path.splitext(self.cine)[0])
        self.new_path = os.path.join(cine_dir, f'channel{channel}')

        if not os.path.exists(self.new_path):
            os.makedirs(self.new_path)

        self.channel = channel
        self.video = Cine(filename = cine, output_bits = 16, remap = False)
        self.vol_info = VolumeProperties(self._DEFAULT_VOL)
        self.vol_info.update_from_file(setup_xml)
        self.cal_info = CalibrationProperties()
        self.cal_info.update_from_file(setup_xml)

        if channel == 0:
            self.A = self.cal_info['A1'] 
            self.C = self.cal_info['C1'] 
        else:
            self.A = self.cal_info['A2'] 
            self.C = self.cal_info['C2']

        #Laser pulse noise depends on the specific laser
        if self.channel == 0:
            #UV laser
            self.p_n = self.cal_info['p1_n']
        else:
            #Green laser
            self.p_n = self.cal_info['p2_n'] 


    def plot_cine_data(self,):
        plt.figure()
        plt.imshow(self.It_avg.mean(axis = 0))
        plt.xlabel('Nx')
        plt.ylabel('Ny')
        plt.colorbar()
        plt.savefig(os.path.join(self.new_path, f"It_avg.png"))

        plt.close()

        plt.figure()
        plt.plot(self.lx.mean(0).mean(0), self.It_avg.mean(0).mean(0), 'b.')
        plt.xlabel(f"lx {self.vol_info['units']}")
        plt.ylabel('Intensity (Cts)')
        plt.savefig(os.path.join(self.new_path, f"x_attenuation.png"))

        plt.close()


        plt.figure()
        plt.plot(self.sy.mean(axis = 0).mean(axis = 1), self.It_avg.mean(0).mean(1), 'b.')
        plt.xlabel('sy (radians)')
        plt.ylabel('Intensity (Cts)')
        plt.savefig(os.path.join(self.new_path, f"y_spline.png"))


        plt.figure()
        plt.plot(self.lz.mean(-1).mean(-1), self.It_avg.mean(-1).mean(-1), 'b.')
        plt.xlabel(f"lz {self.vol_info['units']}")
        plt.ylabel('Intensity (Cts)')
        plt.savefig(os.path.join(self.new_path, f"lz_attenuation.png"))

        plt.close()

        plt.figure()
        plt.imshow(self.gx.mean(axis = 0))
        plt.xlabel('Nx')
        plt.ylabel('Ny')
        plt.colorbar()
        plt.savefig(os.path.join(self.new_path, f"gx.png"))

        plt.close()


        return

    def plot_fit(self,):
        plt.figure()
        plt.plot(self.lx.mean(0).mean(0), self.It_avg.mean(0).mean(0), 'b.')
        plt.plot(self.lx.mean(0).mean(0), self.I_rel.mean(0).mean(0), color = 'red')
        plt.xlabel(f"lx {self.muvi_info['units']}")
        plt.ylabel('Intensity (Cts)')
        plt.savefig(os.path.join(self.new_path, f"x_attenuation.png"))

        plt.close()

        plt.figure()
        plt.plot(self.lz.mean(axis = -1).mean(axis = -1), self.It_avg.mean(axis = -1).mean(axis = -1), 'b.')
        plt.plot(self.lz[self.sampled_slices, ...].mean(axis = -1).mean(axis = -1), self.I_rel.mean(axis = -1).mean(axis = -1), color='red', label = 'Line fit')
        plt.title(f"Attenuation profile. Channel: {self.channel}")
        plt.xlabel(f"lz {self.muvi_info['units']}")
        plt.ylabel("Intensity (Cts)")
        plt.legend()
        plt.savefig(os.path.join(self.new_path, f"lz_attenuation.png"))

        plt.close()

        plt.figure()
        plt.plot(self.sy.mean(axis = 0).mean(axis = 1), self.It_avg.mean(0).mean(1), 'b.')
        plt.plot(self.sy.mean(axis = 0).mean(axis = 1), self.I_rel.mean(axis = 0).mean(axis = 1), color = 'red')
        plt.xlabel('sy (radians)')
        plt.ylabel('Intensity (Cts)')
        plt.savefig(os.path.join(self.new_path, f"y_spline.png"))

        plt.close()

        plt.figure()
        plt.imshow(self.spline_y.mean(axis = 0))
        plt.title("spline averaged along z-axis")
        plt.xlabel("Nx")
        plt.ylabel("Ny")
        plt.savefig(os.path.join(self.new_path, f"y_spline_spread.png"))
        plt.close()

        return


    def rms_error(self, txt_file):
        #Computing RMS
        squared_diff = (self.I_init[self.sampled_slices, ...] - self.I_rel) ** 2
        mean_squared_diff = np.mean(squared_diff)
        rms = np.sqrt(mean_squared_diff)

        #Compute temporal RMS
        squared_diff_t = (self.It_avg[self.sampled_slices, ...] - self.I_rel) ** 2
        mean_squared_diff_t = np.mean(squared_diff_t)
        rms_temporal = np.sqrt(mean_squared_diff_t)

        signal_amplitude = np.max(self.I_init) - np.min(self.I_init)
        n_e = signal_amplitude * (self.cal_info['de'])
        shot_noise = np.sqrt(n_e)

        # Prepare list to export errors to text file
        errors_ls = [f"Extinction coefficient dye: {self.opt_params[0]/self.C:.6f} (micrograms*{self.cal_info['units']})^-1 L",
                     f"Absorption coefficient dye (incident light): {self.opt_params[0]:.6f} ({self.vol_info['units']}^-1)",
                     f"Slices: {self.len_slices}/{self.vol_info['Nz']}",
                     f"RMS (Cts): {rms}",
                     f"Temp RMS (Cts): {rms_temporal}",
                     f"Shot noise (Cts): {shot_noise/self.cal_info['de']}",
                     f"Laser shot-to-shot noise (Cts): {signal_amplitude * self.p_n}"
                    ]

        with open(txt_file, 'a') as f:
            #Write each line from the data list to the file
            for line in errors_ls:
                f.write(line + '\n')
            f.write('\n')

        return

    #Initialize grid
    def initialize_grid(self,):
        from .geometry.volume import rectilinear_grid
        
        X = rectilinear_grid(
            np.arange(self.vol_info['Nx']) + 0.5,
            np.arange(self.vol_info['Ny']) + 0.5,
            np.arange(self.vol_info['Nz']) + 0.5
            )

        distortion = get_distortion_model(self.vol_info)
        self.Xc = distortion.convert(X, 'index-xyz', 'physical')

        return

    def load_cine_data(self,):
        if self.channel == 0:
            #UV default in current setup, uv channel corresponds to frame 0
            initial_frame = 0
        else:
            #Green channel default, green channel corresponds to frame 1
            initial_frame = 1

        grid = self.initialize_grid()
        self.video = Cine(filename = self.cine, output_bits = 16, remap = False)
        frames_arr = np.arange(initial_frame, self.vol_info['Nz'] * self.vol_info['channels'] + initial_frame, self.vol_info['channels']) + self.vol_info['offset']
        #Initalizing intensity array
        print(f"Sampling {self.vol_info['vols']} volumes...")
        I_measured = np.zeros((self.vol_info['vols'],) +  self.Xc.shape[:-1])
        for j in range(self.vol_info['vols']):
            for k, frame in enumerate(frames_arr + self.vol_info['Ns'] * j):
                I_measured[j, k, ...] = self.video.get_frame(i = frame).astype("f")

        self.I_init = I_measured[0, ...]
        #Temporally averaged intensities
        self.It_avg = I_measured.mean(axis = 0)
        
        x, y, z = self.Xc[..., 0], self.Xc[..., 1], self.Xc[..., 2]
        self.lx = (self.vol_info['df_x']  - x) * np.sqrt(1 + (y**2 + z**2)/(self.vol_info['dx'] + x)**2)
        self.lz = (self.vol_info['df_z'] - z) * np.sqrt(1 + (y**2 + x**2)/(self.vol_info['dz'] + z)**2)
        self.gx = (self.vol_info['dx'])/(self.vol_info['dx'] - x)
        self.sy = y/(self.vol_info['dx'] - x)
        self.z = z

        return


    def optimize_intensities(self,):
        def objective_function(p):
            #Computing spline functions for intensity variations along y and along z
            self.spline_y = CubicSpline(self.sy_reduced, p[1:(self.spline_points_y+1)])(self.sy[self.sampled_slices, ...])
            #print(f"p1: {len(p[1:(self.spline_points_y+1)])}")
            self.spline_lz = CubicSpline(self.lz_reduced, p[-self.spline_points_lz:])(self.lz[self.sampled_slices, ...])
            #print(f"p2: {len(p[-self.spline_points_z:])}")
            #Computing relative intensity
            self.I_rel = (self.vol_info['black_level'] + np.exp(-p[0] * self.lx[self.sampled_slices, ...]) * self.spline_y * self.spline_lz *
                          self.gx[self.sampled_slices, ...])               
            #Computing relative intensity without dye attenuation
            self.I_0 = self.vol_info['black_level'] + self.spline_y * self.spline_lz * self.gx[self.sampled_slices, ...]

            return (self.It_avg[self.sampled_slices, ...] - self.I_rel)**2

        def U(p):
            return np.sum(objective_function(p))

        self.opt_params = optimize.minimize(U, x0 = self.init_guess, method='Powell').x

        self.plot_fit()

        return


    def corrected_intensity(self,  skip_array = [264, 64, 32, 1], spline_points_y = 20, spline_points_lz = 5):
        #Prepare to create a text file for the computed rms errors
        print(f"Computing intenisty correction for Channel {self.channel}")
        txt_file = os.path.join(self.new_path, f"rms_errors.txt")
        with open(txt_file, "w+") as f:
            f.write(f"Channel: {self.channel}, Total Volumes: {self.cal_info['vols']} \n")
            f.write("\n")

        self.load_cine_data()
        self.init_guess = [self.A]
        self.I_rel = 0
        self.I_0 = 0

        self.plot_cine_data()

        #Preparing y-spline guess
        self.spline_points_y = spline_points_y
        self.spline_indices_y = np.linspace(0, self.vol_info['Ny'] - 1, spline_points_y, dtype=int)
        Iy = self.It_avg[0, self.spline_indices_y, 0]
        self.init_guess.extend(Iy)
        self.sy_reduced = self.sy[0, :, 0][self.spline_indices_y]
        #Preparing lz-spline guess
        self.spline_points_lz = spline_points_lz
        self.spline_indices_lz = np.linspace(0, self.vol_info['Nz'] - 1, spline_points_lz, dtype=int)
        #[::-1] ensures that lz is strictly decreasing, in our data lz increases as Nz decreases, check FIG for further details in H2C-SVLIF Diego Tapia Silva et. al
        Iz = self.It_avg[self.spline_indices_lz, 0, 0][::-1]
        self.init_guess.extend(Iz)
        self.lz_reduced = self.lz[:, 0, 0][::-1][self.spline_indices_lz]
        #Print total parameters that will be used for optimizer
        print(f"{len(self.init_guess)} parameters for initial guess.")

        for i, skip_z in enumerate(skip_array):
            self.sampled_slices = np.arange(0, self.vol_info['Nz'], 1)[::skip_z]
            self.len_slices =  len(self.sampled_slices)
            print(f"Sampled slices: {self.sampled_slices}")
            print(f"Computing relative intensity and optimized parameters for {self.len_slices}/{self.vol_info['Nz']} thick slices...")
            self.optimize_intensities()
            self.init_guess = self.opt_params
            self.rms_error(txt_file)

        I0_median = 0.5 * (np.max(self.I_0) - np.min(self.I_0))
        I0_normalized = 1/(self.I_0/I0_median)

        if self.vol_info['flip_y'] == True:
            I0_normalized = I0_normalized[:, ::-1, :]
        
        print(f"Temporary outfile: {os.path.join(self.new_path, 'I0.npy')}")
        np.save(os.path.join(self.new_path, "I0.npy"), I0_normalized)

        #Update parameters in the xml file:
        #if self.channel == 0:
        #    self.info['A1'] = round(float(self.opt_params[0]), 6)
        #else:
        #    self.info['A2'] = round(float(self.opt_params[0]), 6) 
        
        
        #self.vol_info.update(self.vol_info)
        #self.vol_info.to_file(self.setup_xml)
        
        #print('XML file updated')

class TargetCalibrationModel:
    '''
    A class used for determining distortion correction parameters from a pickle file.

    Methods
    -------
    '''
    def __init__(self, pickle_file, setup_xml):
        df = pd.read_pickle(pickle_file)
        df = df[df.frame == 0]
        self.muvi_info = VolumeProperties()
        self.muvi_info.update_from_file(setup_xml)

        up, vp, wp = np.array(df['x']), np.array(df['y']), np.array(df['z'])
        self.Up = np.column_stack((up,vp,wp))
        self.setup_xml = setup_xml

        if self.muvi_info['units'] == 'cm':
            self.F = 0.1
        elif self.muvi_info['units'] == 'mm':
            self.F = 1
        else:
            raise ValueError("Select either cm or mm for the units.")
    
    def select_nearest_points(self, X, center, num_points=40):
        distances = np.linalg.norm(X - center, axis=1)
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:num_points]
        nearest_points = X[nearest_indices]
        
        return nearest_points
        
    def subgrid(self, X, bounds = [(195, 315), (230, 290)]):
        #Bounding grid along x and y
        center = np.mean(X, axis=0)
        Xb = X[(X[:, 0] > bounds[0][0]) & (X[:, 0] < bounds[0][1])]
        Xb = Xb[(Xb[:, 1] > bounds[1][0]) & (Xb[:, 1] < bounds[1][1])]
        
        plt.figure()
        
        plt.scatter(center[...,0], center[...,1], color='red')
        plt.plot(X[...,0], X[...,1], '.', color='green')
        plt.plot(Xb[...,0], Xb[...,1], '.', color='blue')
        
        plt.xlabel('Nx')
        plt.ylabel('Ny')
        
        plt.hlines(bounds[1][0], xmin=X[...,0].min(), xmax=X[...,0].max(), colors='r', linestyles='dashed')
        plt.hlines(bounds[1][1], xmin=X[...,0].min(), xmax=X[...,0].max(), colors='r', linestyles='dashed')
        
        plt.vlines(bounds[0][0], ymin=X[...,1].min(), ymax=X[...,1].max(), colors='r', linestyles='dashed')
        plt.vlines(bounds[0][1], ymin=X[...,1].min(), ymax=X[...,1].max(), colors='r', linestyles='dashed')
        
        plt.savefig('bounded_grid.png')

        Xs = self.select_nearest_points(Xb, center)

        fig = plt.figure()
        plt.scatter(center[0], center[1], center[2], color='red')
        plt.scatter(Xs[...,0], Xs[...,1], Xs[...,2], color='blue')
        plt.savefig('subgrid.png')
        
        return  Xs

    
    def optimize_distortion_parameters(self, Up, initial_guess, spacing):
        def objective_function(p):
            muvi_info = VolumeProperties(
                Lx=float(p[0]),
                Ly=float(p[0]),
                Lz=float(p[1]),
                dx=float(p[2]),
                dz=float(p[3]),
                Nx =self.muvi_info['Nx'],
                Ny =self.muvi_info['Ny'],
                Nz =self.muvi_info['Nz'],
            )

            distortion = get_distortion_model(muvi_info)
            Xr = distortion.convert(Up, 'index-xyz', 'physical')
            X = rot.from_euler('zyx', [p[4], p[5], p[6]], degrees = True).inv().apply(Xr) + [p[7], p[8], p[9]]

            return -(np.cos((np.pi*X[:,0])/spacing)*np.cos((np.pi*X[:,1])/spacing))**2 + ((np.pi*X[:,2])/spacing)**2 
        
        def U(p):
            return np.sum(objective_function(p))
        
        def dzdx(p):
            return p[3] - p[2]  
        
        def dzLx(p):
            return p[3] - p[1] 
        
        if self.muvi_info['units'] == 'cm':
            F = 1
        elif self.muvi_info['units'] == 'mm':
            F = 0.1
        else:
            raise ValueError("Set units to either cm or mm")

        constraints = [ {'type': 'ineq', 'fun': dzdx}, {'type': 'ineq', 'fun': dzLx}]

        self.opt_params = optimize.minimize(U, x0=initial_guess, method='L-BFGS-B', constraints=constraints).x
        print(U(self.opt_params)/Up.shape[0])

    def parameters(self,):
        iterations = 100
        params = np.zeros(10)
        U=0
        for i in range(0, iterations):
            initial_guess = [
            np.random.randint(150, 180)*self.F, #Lx
            np.random.randint(150, 180)*self.F, #Lz
            np.random.randint(1100, 1500)*self.F, #dx
            np.random.randint(1300, 1600)*self.F, #dz
            np.random.randint(-1, 1)*self.F, #az
            np.random.randint(40, 50)*self.F, #ay
            np.random.randint(-10, 10)*self.F, #ax
            np.random.randint(-20, 20)*self.F, #xo
            np.random.randint(-20, 20)*self.F, #yo
            np.random.randint(-20, 20)*self.F, #zo
            ]
            #print(initial_guess)
            Xs = self.subgrid(self.Up)
            #Optimize distortion parameters
            self.optimize_distortion_parameters(Xs, initial_guess, spacing = 5*self.F)
            params += self.opt_params
        
        print(f"Oprimized parameters: {params/(i+1)}")

        #Updating XML file
        vol_info = VolumeProperties(
            Lx=round(float(params[0]),2),
            Ly=round(float(params[0]),2),
            Lz=round(float(params[1]),2),
            dx=round(float(params[2]),2),
            dz=round(float(params[3]),2),
        )
        self.muvi_info.update(vol_info)
        self.muvi_info.to_file(self.setup_xml)
        
        print('XML file updated')
