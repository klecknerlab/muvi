#!/usr/bin/python3
#
# Copyright 2024 Diego Tapia Silva
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
""" 
    An example for the intensity correction for multi channel
    intensity correction.
    For multi channel, process can be parallilzed.
"""
import os
import time
import numpy as np 
from multiprocessing import Pool 
from muvi.calibration import IntensityModel

cines = ["gr24dot08ugL_uv4dot41mgL.cine", "gr366ugL_uv0mgL.cine"]
channels = [0, 1]
muvi_xml = "muvi_setup.xml"
cal_json = "calibration_setup.json"
Parallelize = False

def main():
    startTime = time.time()
    I = []
    tfn = 'intensity_normalized.npy'
    ofn = 'muvi_intensity_correction.npy'
    
    if len(cines) != len(channels):
            raise ValueError("Number of channels must match the number of cine files.")
    
    if len(cines) not in [1, 2]:
        raise ValueError("LIF calibration uses a MIN/MAX of ONE/TWO channels. Data should load from a MAX of TWO separate CINES")
    
    if Parallelize and len(channels) == 2:
        print(f"Parallizing intensity correction")
        with Pool() as pool:
            pool.map(intensity_correction, zip(channels, cines, [tfn] * len(channels)))
    
    elif Parallelize and len(channels) == 1:
        raise ValueError('Parallelization requires TWO channels')
    
    else:
        [intensity_correction(args) for args in zip(channels, cines, [tfn] * len(channels))]         
        
    parent_dir = os.path.dirname(cines[0])
    cines_dir = [os.path.join(parent_dir, os.path.splitext(cine)[0]) for cine in cines]
    channels_dir = [os.path.join(cine_dir, f'channel{channel}') for cine_dir, channel in zip(cines_dir, channels)]
    
    #Append corrected intensity arrays to a list
    [I.append(np.load(os.path.join(channel_dir, tfn))) for channel_dir in channels_dir]
    if len(channels) == 2:
        #Stack arrays 
        I = np.stack((I[0], I[1]), axis = -1)
    else:
        I = I[0]
    #Save muvi_intensity_correction.py 
    np.save(os.path.join(parent_dir, ofn), I)
    #Remove the temporary files 'cp_normalized.npy'  
    [os.remove(os.path.join(channel_dir, tfn)) for channel_dir in channels_dir]
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

def intensity_correction(args):
    channel, cine, tfn = args
    intensity = IntensityModel(channel = channel, cine = cine, setup_xml = muvi_xml, setup_json= cal_json)
    intensity.corrected_intensity(ofn = tfn, skip_array = [256, 128, 64, 32, 1], spline_points_sy = 36, spline_points_lz = 5)


if __name__ == "__main__":
    main()