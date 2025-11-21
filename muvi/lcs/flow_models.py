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

def double_gyre(R, T):
    A = 0.1
    ε = 0.25
    ω = np.pi / 5
    
    
    a = ε * np.sin(ω * T)
    b = 1 - 2 * ε * np.sin(ω * T)
    f = a * R[..., 0]**2 + b * R[..., 0]
    dfdR = 2 * a *R[..., 0] + b
    
    u = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * R[..., 1])
    v = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * R[..., 1]) * dfdR
    
    U = np.zeros_like(R)
    U[..., 0] = u
    U[..., 1] = v
    
    return U


def abc_flow(R, T):
    A = np.sqrt(3)
    B = np.sqrt(2)
    C = 1
    
    u =  (A + 0.5 * T * np.sin(np.pi * T)) * np.sin(R[..., 2]) + C * np.cos(R[..., 1])
    v = B * np.sin(R[..., 0]) + (A + 0.5 * T * np.sin(np.pi * T)) * np.cos(R[..., 2])
    w = C * np.sin(R[..., 1]) + B * np.cos(R[..., 0])
    
    U = np.zeros_like(R)
    U[..., 0] = u
    U[..., 1] = v
    U[..., 2] = w
    
    return U