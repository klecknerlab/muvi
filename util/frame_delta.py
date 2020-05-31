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

num_frames = 1500
rel_diff = 0.8

video = Cine(sys.argv[1], gamma=1.0, output_bits=16)


f0 = video[0].astype('f')
f1 = video[1].astype('f')

diff = []
fn = np.arange(1, num_frames)

for i in fn:
    f2 = video[i+1].astype('f')
    # diff.append((f1-f0).std())
    diff.append((2*f1 - f0 - f2).std())
    f0, f1 = f1, f2


print('--- Frames with High Activity ---')

j = None
haf = fn[np.where(diff > np.max(diff) * rel_diff)]
for i in haf:
    p = '   %5d' % i
    if j:
        p += ' (\u0394=%d)' % (i-j)
    print(p)
    j = i

print('--- Estimated Parameters (from first two maximum) ---')
Ns = haf[1] - haf[0]
print('      Ns = %d' % Ns)

frame_sizes = np.concatenate([2**np.arange(5, 12), 3 * 2**np.arange(4, 10)])
# frame_sizes.sort()

Nz = frame_sizes[np.where(frame_sizes < Ns)].max()

print('      Nz = %d' % Nz)

offset = (haf[1] + haf[0] - Nz) // 2
while offset > Ns: offset -= Ns

print('  offset = %d' % offset)

plt.plot(fn, diff)
plt.xlabel('Frame number')
plt.ylabel('Rel. frame change')
plt.show()
