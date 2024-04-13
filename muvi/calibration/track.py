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
from muvi.calibration import TrackingModel
import time
import numpy as np
import argparse

def main():
    startTime = time.time()
    track(startTime)

def track(startTime):  
    parser = argparse.ArgumentParser(description='Tracking of particles from a VTI file')
    parser.add_argument('vti_file', type=str, help='Input VTI file')
    parser.add_argument('muvi_setup', type=str, help='XML file to use for muvi conversion parameters')
    parser.add_argument('calibration_setup', type=str, help='JSON file to use for distortion and intensity correction')

    args = parser.parse_args()
    tracks = TrackingModel(vti=args.vti_file, setup_xml=args.muvi_setup, setup_json=args.calibration_setup)
    tracks.vti_tracks()
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

if __name__ == "__main__":
    main()