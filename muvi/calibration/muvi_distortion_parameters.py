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
""" An example to recover distortion parameters {Lx, Ly, Lz, dx, dz}.
"""
from muvi.calibration import TargetCalibrationModel
import time

pickle = ".\sphere\channel1\muvi_track.pickle"
cal_json = "calibration_setup.json"
muvi_xml = "muvi_setup.xml"

def main():
    startTime = time.time()
    distortion_parameters(startTime)

def distortion_parameters(startTime):
    calibration = TargetCalibrationModel(pickle_file = pickle, setup_xml = muvi_xml, setup_json= cal_json)
    calibration.parameters()
    
    executionTime = (time.time() - startTime)
    print(f'execution time: {executionTime} seconds, {executionTime/60} minutes, {executionTime/3600} hours')

if __name__ == "__main__":
    main()