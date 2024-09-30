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

def adj_frames_filter(data_frame, start_frame):
        """
        -Filters data (either pandas data frame) keeping only particles that appear in two adjacent frames.
        This is used particularly for particle tracking data because particles can dissapear and reappear between frames.

        Args:
            data_frame: pandas DataFrame (e.g. from trackpy). 
            start_frame: starting frame for filtering.
        Returns:
            Filtered trajectories
        """
        #Determine end_frame 
        end_frame = start_frame + 1

        #Define frames to be considered
        frames = np.arange(start_frame, end_frame + 1)
        filtered_data = data_frame[data_frame['frame'].isin(frames)]
        df_adj_frames = filtered_data[filtered_data['frame'].isin(frames)]

        #Identify particles and their frame counts
        particle_frame_counts = df_adj_frames.groupby('particle')['frame'].nunique()
        valid_particles = particle_frame_counts[particle_frame_counts == len(frames)].index

        #Filter duplicates to keep only those particles appearing in all frames
        duplicates = df_adj_frames[df_adj_frames['particle'].isin(valid_particles)]

        corrected_coords = ['xc', 'yc', 'zc']
        uncorrected_coords = ['x', 'y', 'z']

        #Find existing corrected coordinates
        existing_corrected = [coord for coord in corrected_coords if coord in data_frame.columns]

        #If at least one corrected coordinate is found, use it
        if existing_corrected:
            coords = existing_corrected
        else:
            # Find existing uncorrected coordinates if no corrected coordinates are found
            existing_uncorrected = [coord for coord in uncorrected_coords if coord in data_frame.columns]
            if existing_uncorrected:
                coords = existing_uncorrected
            else:
                raise ValueError("Neither corrected ('xc', 'yc', 'zc') nor uncorrected ('x', 'y', 'z') coordinates are present in the DataFrame.")
        
        num_dims = len(coords)

        if num_dims <2:
            raise ValueError(f"num_dims must be either 2 or 3. Current num_dims: {num_dims}. With coordinates: {coords}")

        #Check if preferred coordinates are present
        coords = corrected_coords if all(coord in data_frame.columns for coord in corrected_coords) else uncorrected_coords

        #Extract and sort trajectories
        particles0 = duplicates[duplicates['frame'] == start_frame]['particle']
        particles1 = duplicates[duplicates['frame'] == start_frame + 1]['particle']
        traj0 = duplicates[duplicates['frame'] == start_frame][coords].to_numpy()
        traj1 = duplicates[duplicates['frame'] == start_frame + 1][coords].to_numpy()

        sorted_indices0 = particles0.argsort()
        sorted_indices1 = particles1.argsort()

        traj0 = traj0[sorted_indices0]
        traj1 = traj1[sorted_indices1]

        return np.stack((traj0, traj1), axis=0)
    
def select_frames(data_frame, start_frame = None, end_frame = None):
    """
    -Filters data (either pandas data frame) keeping only particles that appear in the range [start_frame, end_frame]

    Args:
        data_frame: pandas DataFrame (e.g. from trackpy). 
        start_frame: starting frame for filtering.
        end_frame: end frame for filtering.

    Returns:
        Filtered DataFrame
    """

    if start_frame is None:
        start_frame = data_frame['frame'].min()

    if end_frame is None:
        end_frame = data_frame['frame'].max()
    
    elif end_frame <= start_frame:
        raise ValueError(f"'start_frame' ({start_frame}) must precede 'end_frame' ({end_frame}).")
    
    return data_frame[(data_frame['frame'] >= start_frame) & (data_frame['frame'] <= end_frame)]    