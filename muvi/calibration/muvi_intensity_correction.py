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
""" An example for the intensity correction for two channels using
    two separate cine files.
"""
import os
import time
import numpy as np 
from muvi.calibration import IntensityModel

cines = ["gr24dot08ugL_uv4dot41mgL.cine", "gr366ugL_uv0mgL.cine"]
muvi_xml = "muvi_setup.xml"
cal_json = "calibration_setup.json"

tfn = 'cp_normalized.npy'
ofn = 'muvi_intensity_correction.npy'

def main():
    startTime = time.time()
    intensity_correction(startTime)

def intensity_correction(startTime):
    I = []
    for channel, cine in enumerate(cines):
        intensity = IntensityModel(channel, cine, setup_xml = muvi_xml, setup_json= cal_json)
        intensity.corrected_intensity(ofn = tfn, skip_array=[256])
        parent_dir = os.path.dirname(cine)
        cine_dir = os.path.join(parent_dir, os.path.splitext(cine)[0])
        channel_dir = os.path.join(cine_dir, f'channel{channel}')
        I.append(np.load(os.path.join(channel_dir, tfn)))

        if channel == 1:
            I = np.stack((I[0], I[1]), axis = -1)
            np.save(os.path.join(parent_dir, ofn), I)

        np.save(os.path.join(parent_dir, ofn), I)
        os.remove(os.path.join(channel_dir, tfn))

    print(f"Outfile: {os.path.join(parent_dir, ofn)}")
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

if __name__ == "__main__":
    main()