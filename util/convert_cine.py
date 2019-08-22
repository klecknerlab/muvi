#!/usr/bin/python3
#
# Copyright 2019 Dustin Kleckner
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
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Convert a CINE file to a .h5 movie')
parser.add_argument('infile', type=str, help='Input CINE file', nargs=1)
parser.add_argument('outfile', type=str, help='Output HDF5 file', nargs='?', default=None)
parser.add_argument('-v', '--volume', type=int, help='Number of frames per volume (should be <= # per scan)', default=512)
parser.add_argument('-s', '--scan', type=int, help='Number of frames per scan', default=None)
parser.add_argument('-o', '--offset', type=int, help='Offset to first valid frame', default=0)
parser.add_argument('-c', '--clip', type=int, help='Noise clip value (default: 80; good for 10 bit images)', default=80)
parser.add_argument('-m', '--max', type=int, help='Max value after clipping (default: use full dynamic range', default=1024-80)
parser.add_argument('-g', '--gamma', help='If specified, apply 2.0 gamma correction', default=False, action='store_true')
parser.add_argument('-n', '--number', type=int, help='Number of volumes to output (default: all)', default=None)

args = parser.parse_args()

ifn = args.infile[0]
ofn = args.outfile

if ofn is None:
    bfn, ext = os.path.splitext(ifn)
    ofn = bfn + '.muv'

print(ifn, '=>', ofn)
vol = muvi.CineMovie(ifn, fpv=args.volume, offset=0, fps=args.scan, info=None, clip=args.clip, top=args.max, gamma=args.gamma)
vol.save(ofn, print_status=True, end=args.number)
