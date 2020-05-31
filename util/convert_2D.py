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

import muvi
import os, sys
import argparse

parser = argparse.ArgumentParser(description='Convert a CINE file to a VTI movie')
parser.add_argument('infile', type=str, help='Input CINE file', nargs=1)
parser.add_argument('outfile', type=str, help='Output VTI file', nargs='?', default=None)
parser.add_argument('-x', '--xml', type=str, help='XML file to use for conversion parameters', default=None)
parser.add_argument('-s', '--start', type=int, help='First volume index to convert (default: 0)', default=0)
parser.add_argument('-e', '--end', type=int, help='Last volume index to convert (default: all volumes converted)', default=None)

args = parser.parse_args()

ifn = args.infile[0]
ofn = args.outfile

vm = muvi.open_3D_movie(ifn, setup_xml = args.xml)

if ofn is None:
    bfn, ext = os.path.splitext(ifn)
    ofn = bfn + '.vti'

print(ifn, '=>', ofn)
print('Info fields')
print('-'*40)
for key, val in vm.info.items():
    print('%15s: %s' % (key, repr(val)))
# vol = muvi.CineMovie(ifn, fpv=args.volume, offset=0, fps=args.scan, info=None, clip=args.clip, top=args.max, gamma=args.gamma)
vm.save(ofn, print_status=True, start=args.start, end=args.end)
