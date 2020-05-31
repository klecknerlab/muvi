#!/usr/bin/python3
#
# Copyright 2020 Dustin Kleckner
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

from muvi.readers.cine import Cine
import sys
import matplotlib.pyplot as plt
import numpy as np

offset_frame = None
num_frames = 100
max_level = 0.1
num_bins = 1000
skip = 100

video = Cine(sys.argv[1], gamma=1.0, output_bits=16)
if offset_frame is None:
    offset_frame = (len(video) - num_frames*skip) // 2

frames = np.array(video[offset_frame:offset_frame + num_frames*skip:skip]) / (2**16 - 1)

bins = np.logspace(-3, 0, num_bins + 1)
hist, bins = np.histogram(frames.flat, bins=bins)

total = np.cumsum(hist)
total = total / float(total[-1])

bin_top = bins[1:]
plt.loglog(bin_top, 1-total)
plt.xlabel('Relative intensity')
plt.ylabel('Fration above intensity')
plt.xlim(1E-3, 1)
plt.ylim(1E-5, 1)

print('---- Fraction Clipped ----')
for level in (np.arange(10)+1)*1E-3:
    i = np.where(bin_top >= level)[0][0]
    print('dark_level = %.3f: %.1f%%' % (level, total[i] * 100))

plt.show()
