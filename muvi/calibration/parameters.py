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
    distortion_parameters(startTime)

def distortion_parameters(startTime):
    parser = argparse.ArgumentParser(description='Recover distortion parameters using a pickle file')
    parser.add_argument('infile', type=str, help='Pickle file', nargs=1)
    parser.add_argument('--muvi_setup', type=str, help='XML file to use for muvi conversion parameters', default='muvi_setup.xml',  nargs='?')

    args = parser.parse_args()
    print(args.infile[0])
    calibration = muvi.calibration.TargetCalibrationModel(pickle_file = args.infile[0], setup_xml = args.muvi_setup)
    calibration.parameters()
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

if __name__ == "__main__":
    main()