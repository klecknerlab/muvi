#!/usr/bin/python3
#
# Copyright 2024 Dustin Kleckner
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

import os, sys
os.environ['PYTHON_JULIACALL_THREADS'] = "auto"
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = "yes"


try:
    # raise Exception("Testing Numba")
    import juliacall
    HAS_JULIA = True

except:
    HAS_JULIA = False

    try: 
        import numba
        HAS_NUMBA = True
        import warnings
        warnings.warn('Numba is being depreciated from the MUVI library -- recommended replacement in "juliacall"')
    except:
        HAS_NUMBA = False
        raise RuntimeError('Either julicall or numba must be installed to use the MUVI library')

if HAS_JULIA:
    jl = juliacall.Main
    jl.include(os.path.join(os.path.split(__file__)[0], 'accel.jl'))
    unpack_10b = jl.unpack_10b
    decompress_blocks = jl.decompress_blocks
    # print("Julia Libraries successfully compiled!")

    def writeText(s, start, cap_height, line_height, output, atlas):
        x0 = 0.0
        y0 = -cap_height
        width = 0.0
        height = cap_height
        i = start

        N = len(atlas)
        M = len(output)

        for c in s:
            u = ord(c)

            if u > N: # Outside the covered range of glyphs
                continue

            if u == 10: # line break
                x0 = 0.0
                y0 -= line_height
                height += line_height
                width = max(width, x0)
                continue

            advance = atlas[u, 0]
            if advance < 0.0: # This is an undefined character!
                continue

            if atlas[u, 1] > -1000.0: # if this is whitespace, left = -2**16
                output[i, 2] = atlas[u, 1] + x0
                output[i, 3] = atlas[u, 2] + y0
                output[i, 4] = atlas[u, 3]
                output[i, 5] = atlas[u, 4]
                output[i, 6] = atlas[u, 5]
                output[i, 7] = atlas[u, 6]
                i += 1

            x0 += advance

            if i >= M:
                break

        width = max(width, x0)

        for j in range(start, i):
            output[j, 0] = width
            output[j, 1] = height
            output[j, 3] += height

        return i
    

    
elif HAS_NUMBA:
    @numba.njit(cache=True)
    def writeText (s, start, cap_height, line_height, output, atlas):
        x0 = 0.0
        y0 = -cap_height
        width = 0.0
        height = cap_height
        i = start

        N = len(atlas)
        M = len(output)

        for c in s:
            u = ord(c)

            if u > N: # Outside the covered range of glyphs
                continue

            if u == 10: # line break
                x0 = 0.0
                y0 -= line_height
                height += line_height
                width = max(width, x0)
                continue

            advance = atlas[u, 0]
            if advance < 0.0: # This is an undefined character!
                continue

            if atlas[u, 1] > -1000.0: # if this is whitespace, left = -2**16
                output[i, 2] = atlas[u, 1] + x0
                output[i, 3] = atlas[u, 2] + y0
                output[i, 4] = atlas[u, 3]
                output[i, 5] = atlas[u, 4]
                output[i, 6] = atlas[u, 5]
                output[i, 7] = atlas[u, 6]
                i += 1

            x0 += advance

            if i >= M:
                break

        width = max(width, x0)

        for j in range(start, i):
            output[j, 0] = width
            output[j, 1] = height
            output[j, 3] += height

        return i

    # A simple homegrown LZ4 decompresser, which uses JIT through numba.
    # It's really fast!  Typically performance is several times that of the LZ4
    #   library!  A few GB/s of decompression speed is easily acheivable on
    #   commodity hardware, primarily due to excuting in parallel.
    @numba.jit(nopython=True, parallel=True, cache=True)
    def decompress_blocks(input, block_size, last_block_size, block_ends, output):
        num_blocks = len(block_ends)

        for p in numba.prange(num_blocks):
            if p == 0:
                i = numba.uint64(0)
            else:
                i = numba.uint64(block_ends[p - numba.uint(1)])

            block_end = numba.uint64(block_ends[p])
            j = numba.uint64(block_size * p)

            if (p == (num_blocks - numba.uint8(1))):
                end = j + numba.uint64(last_block_size)
            else:
                end = j + numba.uint64(block_size)

            while ((j < end) and (i < block_end)):
                t1 = numba.uint16((input[i] & 0xF0) >> 4)
                t2 = numba.uint16((input[i] & 0x0F) + 4)
                i += numba.uint8(1)

                if (t1 == 15):
                    while input[i] == 255:
                        t1 += numba.uint8(input[i])
                        i += numba.uint8(1)

                    t1 += numba.uint8(input[i])
                    i += numba.uint8(1)

                for n in range(t1):
                    output[j] = input[i]
                    i += numba.uint8(1)
                    j += numba.uint8(1)

                if (j >= end): break

                off = numba.uint16(input[i]) + (numba.uint16(input[i+1]) << 8)
                i += numba.uint8(2)

                if (t2 == 19):
                    while input[i] == 255:
                        t2 += numba.uint8(input[i])
                        i += numba.uint8(1)

                    t2 += numba.uint8(input[i])
                    i += numba.uint8(1)

                for n in range(t2):
                    output[j] = output[j - off]
                    j += numba.uint8(1)


    @numba.jit(nopython=True, parallel=True, cache=True)
    def unpack_10b(input, tone_map, output, offset=0, stride=1):
        # 10 bits data => 4 points packed into 5 bytes
        for block in numba.prange(len(input) // 5):
            i = block * 5
            val0 = (numba.uint16(input[i+0] & 0xFF) << 2) + (numba.uint16(input[i+1]) >> 6)
            val1 = (numba.uint16(input[i+1] & 0x3F) << 4) + (numba.uint16(input[i+2]) >> 4)
            val2 = (numba.uint16(input[i+2] & 0x0F) << 6) + (numba.uint16(input[i+3]) >> 2)
            val3 = (numba.uint16(input[i+3] & 0x03) << 8) + (numba.uint16(input[i+4]) >> 0)

            j = offset + (block * 4 * stride)
            output[j           ] = tone_map[val0]
            output[j +   stride] = tone_map[val1]
            output[j + 2*stride] = tone_map[val2]
            output[j + 3*stride] = tone_map[val3]

#Benchmarking; set to true to print uncompress speed.
if False:
    import time

    _decompress_blocks = decompress_blocks

    def decompress_blocks(*args):
        start = time.time()
        _decompress_blocks(*args)
        el = time.time() - start
        print(f'Decompress in {"Julia" if HAS_JULIA else "Numba"}: {el*1000:.1f}')