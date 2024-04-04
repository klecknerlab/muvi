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
import muvi
import time
import os
import numpy as np
import argparse

def main():
    startTime = time.time()
    intensity_correction(startTime)

def intensity_correction(startTime):
    parser = argparse.ArgumentParser(description='Correct intensities from Cine files and output to a numpy array')
    parser.add_argument('infile', type=str, help='Input CINE files', nargs='*')
    parser.add_argument('--muvi_setup', type=str, help='XML file to use for muvi conversion parameters', default='muvi_setup.xml',  nargs='?')

    args = parser.parse_args()
    ofn = 'muvi_intensity_correction.npy'

    I = []
    for channel, cine in enumerate(args.infile):
        intensity = muvi.IntensityModel(channel, cine, setup_xml = args.muvi_setup)
        intensity.calibration.corrected_intensity()
        parent_dir = os.path.dirname(cine)
        cine_dir = os.path.join(parent_dir, os.path.splitext(cine)[0])
        channel_dir = os.path.join(cine_dir, f'channel{channel}')
        intensity_fn = 'I0.npy'
        I.append(np.load(os.path.join(channel_dir, intensity_fn)))

        if channel == 1:
            I = np.stack((I[0], I[1]), axis = -1)
            np.save(os.path.join(parent_dir, ofn), I)

        np.save(os.path.join(parent_dir, ofn), I)
        os.remove(os.path.join(channel_dir, intensity_fn))

    print(f"Outfile: {os.path.join(parent_dir, ofn)}")
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

if __name__ == "__main__":
    main()