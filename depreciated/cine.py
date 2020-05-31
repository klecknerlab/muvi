#!/usr/bin/env python

# Reader for CINE files produced by Vision Research Phantom Software

################################################################################

# This code was originally developed (w/o a license) for the Irvine lab at the
#  university of Chicago.  Moved to an Apache License (with permision) when
#  ported to Python 3 in 2019 by Dustin Kleckner (dkleckner@ucmerced.edu).

################################################################################

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

################################################################################


import sys
import os
import time
import struct
import numpy as np
from threading import Lock
import datetime
import hashlib


FRACTION_MASK = (2**32-1)
MAX_INT = 2**32

#Define types for fields
BYTE = 'B'
WORD = 'H'
INT16 = 'h'
SHORT = 'h'
BOOL = 'i'
DWORD = 'I'
UINT = 'I'
LONG = 'l'
INT = 'l'
FLOAT = 'f'
DOUBLE = 'd'
TIME64 = 'Q'
RECT = '4i'
WBGAIN = '2f'
IMFILTER = '28i'


#Define Header fields
TAGGED_FIELDS = {
    1000: ('ang_dig_sigs', ''),
    1001: ('image_time_total', TIME64),
    1002: ('image_time_only', TIME64),
    1003: ('exposure_only', DWORD),
    1004: ('range_data', ''),
    1005: ('binsig', ''),
    1006: ('anasig', ''),
    1007: ('undocumented', '')}  # 1007 exists in my files, but is not in documentation I can find

HEADER_FIELDS = [
    ('type', '2s'),
    ('header_size', WORD),
    ('compression', WORD),
    ('version', WORD),
    ('first_movie_image', LONG),
    ('total_image_count', DWORD),
    ('first_image_no', LONG),
    ('image_count', DWORD),
    ('off_image_header', DWORD),
    ('off_setup', DWORD),
    ('off_image_offsets', DWORD),
    ('trigger_time', TIME64),
]

BITMAP_INFO_FIELDS = [
    ('bi_size', DWORD),
    ('bi_width', LONG),
    ('bi_height', LONG),
    ('bi_planes', WORD),
    ('bi_bit_count', WORD),
    ('bi_compression', DWORD),
    ('bi_image_size', DWORD),
    ('bi_x_pels_per_meter', LONG),
    ('bi_y_pels_per_meter', LONG),
    ('bi_clr_used', DWORD),
    ('bi_clr_important', DWORD),
]

SETUP_FIELDS = [
    ('frame_rate_16', WORD),
    ('shutter_16', WORD),
    ('post_trigger_16', WORD),
    ('frame_delay_16', WORD),
    ('aspect_ratio', WORD),
    ('contrast_16', WORD),
    ('bright_16', WORD),
    ('rotate_16', BYTE),
    ('time_annotation', BYTE),
    ('trig_cine', BYTE),
    ('trig_frame', BYTE),
    ('shutter_on', BYTE),
    ('description_old', '121s'),  # Guessed at length... because it isn't documented!  This seems to work.
    ('mark', '2s'),
    ('length', WORD),
    ('binning', WORD),
    ('sig_option', WORD),
    ('bin_channels', SHORT),
    ('samples_per_image', BYTE)] + \
    [('bin_name%d' % i, '11s') for i in range(8)] + [
        ('ana_option', WORD),
        ('ana_channels', SHORT),
        ('res_6', BYTE),
        ('ana_board', BYTE)] + \
    [('ch_option%d' % i, SHORT) for i in range(8)] + \
    [('ana_gain%d' % i, FLOAT) for i in range(8)] + \
    [('ana_unit%d' % i, '6s') for i in range(8)] + \
    [('ana_name%d' % i, '11s') for i in range(8)] + [
    ('i_first_image', LONG),
    ('dw_image_count', DWORD),
    ('n_q_factor', SHORT),
    ('w_cine_file_type', WORD)] + \
    [('sz_cine_path%d' % i, '65s') for i in range(4)] + [
    ('b_mains_freq', WORD),
    ('b_time_code', BYTE),
    ('b_priority', BYTE),
    ('w_leap_sec_dy', DOUBLE),
    ('d_delay_tc', DOUBLE),
    ('d_delay_pps', DOUBLE),
    ('gen_bits', WORD),
    ('res_1', INT16),  # Manual says INT, but this is clearly wrong!
    ('res_2', INT16),
    ('res_3', INT16),
    ('im_width', WORD),
    ('im_height', WORD),
    ('edr_shutter_16', WORD),
    ('serial', UINT),
    ('saturation', INT),
    ('res_5', BYTE),
    ('auto_exposure', UINT),
    ('b_flip_h', BOOL),
    ('b_flip_v', BOOL),
    ('grid', UINT),
    ('frame_rate', UINT),
    ('shutter', UINT),
    ('edr_shutter', UINT),
    ('post_trigger', UINT),
    ('frame_delay', UINT),
    ('b_enable_color', BOOL),
    ('camera_version', UINT),
    ('firmware_version', UINT),
    ('software_version', UINT),
    ('recording_time_zone', INT),
    ('cfa', UINT),
    ('bright', INT),
    ('contrast', INT),
    ('gamma', INT),
    ('reserved1', UINT),
    ('auto_exp_level', UINT),
    ('auto_exp_speed', UINT),
    ('auto_exp_rect', RECT),
    ('wb_gain', '8f'),
    ('rotate', INT),
    ('wb_view', WBGAIN),
    ('real_bpp', UINT),
    ('conv_8_min', UINT),
    ('conv_8_max', UINT),
    ('filter_code', INT),
    ('filter_param', INT),
    ('uf', IMFILTER),
    ('black_cal_sver', UINT),
    ('white_cal_sver', UINT),
    ('gray_cal_sver', UINT),
    ('b_stamp_time', BOOL),
    ('sound_dest', UINT),
    ('frp_steps', UINT),
    ] + [('frp_img_nr%d' % i, INT) for i in range(16)] + \
        [('frp_rate%d' % i, UINT) for i in range(16)] + \
        [('frp_exp%d' % i, UINT) for i in range(16)] + [
    ('mc_cnt', INT),
    ] + [('mc_percent%d' % i, FLOAT) for i in range(64)] + [
    ('ci_calib', UINT),
    ('calib_width', UINT),
    ('calib_height', UINT),
    ('calib_rate', UINT),
    ('calib_exp', UINT),
    ('calib_edr', UINT),
    ('calib_temp', UINT),
    ] + [('header_serial%d' % i, UINT) for i in range(4)] + [
    ('range_code', UINT),
    ('range_size', UINT),
    ('decimation', UINT),
    ('master_serial', UINT),
    ('sensor', UINT),
    ('shutter_ns', UINT),
    ('edr_shutter_ns', UINT),
    ('frame_delay_ns', UINT),
    ('im_pos_xacq', UINT),
    ('im_pos_yacq', UINT),
    ('im_width_acq', UINT),
    ('im_height_acq', UINT),
    ('description', '4096s')
]


T64_F = lambda x: int(x) / 2.**32
T64_F_ms = lambda x: '%.3f' % (float(x.rstrip('L')) / 2.**32)
T64_S = lambda s: lambda t: time.strftime(s, time.localtime(float(t.rstrip('L'))/2.**32))


#Processing the data in chunks keeps it in the L2 catch of the processor, increasing speed for large arrays by ~50%
CHUNK_SIZE = 6 * 10**5 #Should be divisible by 3, 4 and 5!  This seems to be near-optimal.

def ten2sixteen(a):
    b = np.zeros(a.size//5*4, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2, a3, a4) = [a[j+i:j+CHUNK_SIZE:5].astype('u2') for i in range(5)]

        k = j//5 * 4
        k2 = k + CHUNK_SIZE//5 * 4

        b[k+0:k2:4] = ((a0 & 0b11111111) << 2) + ((a1 & 0b11000000) >> 6)
        b[k+1:k2:4] = ((a1 & 0b00111111) << 4) + ((a2 & 0b11110000) >> 4)
        b[k+2:k2:4] = ((a2 & 0b00001111) << 6) + ((a3 & 0b11111100) >> 2)
        b[k+3:k2:4] = ((a3 & 0b00000011) << 8) + ((a4 & 0b11111111) >> 0)

    return b

def sixteen2ten(b):
    a = np.zeros(b.size//4*5, dtype='u1')

    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1, b2, b3) = [b[j+i:j+CHUNK_SIZE:4] for i in range(4)]

        k = j//4 * 5
        k2 = k + CHUNK_SIZE//4 * 5

        a[k+0:k2:5] =                              ((b0 & 0b1111111100) >> 2)
        a[k+1:k2:5] = ((b0 & 0b0000000011) << 6) + ((b1 & 0b1111110000) >> 4)
        a[k+2:k2:5] = ((b1 & 0b0000001111) << 4) + ((b2 & 0b1111000000) >> 6)
        a[k+3:k2:5] = ((b2 & 0b0000111111) << 2) + ((b3 & 0b1100000000) >> 8)
        a[k+4:k2:5] = ((b3 & 0b0011111111) << 0)

    return a

def twelve2sixteen(a):
    b = np.zeros(a.size//3*2, dtype='u2')

    for j in range(0, len(a), CHUNK_SIZE):
        (a0, a1, a2) = [a[j+i:j+CHUNK_SIZE:3].astype('u2') for i in range(3)]

        k = j//3 * 2
        k2 = k + CHUNK_SIZE//3 * 2

        b[k+0:k2:2] = ((a0 & 0xFF) << 4) + ((a1 & 0xF0) >> 4)
        b[k+1:k2:2] = ((a1 & 0x0F) << 8) + ((a2 & 0xFF) >> 0)

    return b


def sixteen2twelve(b):
    a = np.zeros(b.size//2*3, dtype='u1')

    for j in range(0, len(a), CHUNK_SIZE):
        (b0, b1) = [b[j+i:j+CHUNK_SIZE:2] for i in range(2)]

        k = j//2 * 3
        k2 = k + CHUNK_SIZE//2 * 3

        a[k+0:k2:3] =                       ((b0 & 0xFF0) >> 4)
        a[k+1:k2:3] = ((b0 & 0x00F) << 4) + ((b1 & 0xF00) >> 8)
        a[k+2:k2:3] = ((b1 & 0x0FF) << 0)

    return a


class Cine(object):
    '''Class for reading Vision Research CINE files, e.g. from Phantom cameras.

    Supports indexing, so frame can be accessed like list items, and ``len``
    returns the number of frames.  Iteration is also supported.

    Cine objects also use locks for file reading, allowing cine objects to
    be shared safely by several threads.

    Parameters
    ---------
    filename : string
        Source filename
    '''

    def __init__(self, fn):
        self.f = open(fn, 'rb')
        self.fn = fn

        self.read_header(HEADER_FIELDS)
        self.read_header(BITMAP_INFO_FIELDS, self.off_image_header)
        self.read_header(SETUP_FIELDS, self.off_setup)
        self.image_locations = self.unpack('%dQ' % self.image_count, self.off_image_offsets)
        if type(self.image_locations) not in (list, tuple):
            self.image_locations = [self.image_locations]

        self.width = self.bi_width
        self.height = self.bi_height

        self.file_lock = Lock()  # Allows Cine object to be accessed from multiple threads!

        self._hash = None


    def gamma_corrected_frame(self, frame_number, bottom_clip=0, top_clip=None, gamma=2.2):
        '''Return a frame as a gamma corrected 'u1' array, suitable for saving
        to a standard image.

        Output is equal to: ``255 * ((original - bottom_clip) / (top_clip - bottom_clip))**(1/gamma)``

        Parameters
        ----------
        frame_number : integer
        gamma : float (default: 2.2)
            The gamma correction to apply
        top_clip : integer (default: 0)
        bottom_clip : integer (default: 2**real_bpp)

        Returns
        -------
        frame : numpy array (dtype='u1')

        '''
        if top_clip is None: top_clip = 2**self.real_bpp

        return (255 * np.clip(((self.get_frame(frame_number) - bottom_clip) / float(top_clip - bottom_clip)), 0, 1)**(1.0/gamma)).astype('u1')



    def get_frame(self, frame_number):
        '''Get a frame from the cine file.

        Parameters
        ----------
        frame_number : integer

        Returns
        -------
        frame : numpy array (dtype='u1' or 'u2', depending on bit depth)
        '''

        self.file_lock.acquire()

        image_start = self.image_locations[frame_number]
        annotation_size = self.unpack(DWORD, image_start)
        annotation = self.unpack('%db' % (annotation_size - 8))
        image_size = self.unpack(DWORD)

        #self.f.seek(image_start + annotation_size-8)
        data_type = 'u1' if self.bi_bit_count in (8, 24) else 'u2'

        actual_bits = image_size * 8 // (self.width * self.height)

        if actual_bits in (10, 12):
            data_type = 'u1'

        self.f.seek(image_start + annotation_size)

        frame = np.frombuffer(self.f.read(image_size), data_type)

        if (actual_bits == 10):
           frame = ten2sixteen(frame)
        elif (actual_bits == 12):
           frame = twelve2sixteen(frame)

        # if (actual_bits % 8):
        #     raise ValueError('Data should be byte aligned, packed data not supported' % actual_bits)

        frame = frame.reshape(self.height, self.width)[::-1]

        if actual_bits in (10, 12):
           frame = frame[::-1, :]  # Don't know why it works this way, but it does...

        self.file_lock.release()

        return frame


    def __len__(self):
        return self.image_count

    len = __len__


    def __getitem__(self, key):
        if type(key) == slice:
            return map(self.get_frame, range(self.image_count)[key])

        return self.get_frame(key)


    def get_time(self, frame_number):
        '''Get the time of a specific frame.

        Parameters
        ----------
        frame_number : integer

        Returns
        -------
        time : float
            Time from start in seconds.'''
        return float(frame_number) / self.frame_rate


    def get_fps(self):
        '''Get the frames per second of the movie.

        Returns
        -------
        fps : int
        '''
        return self.frame_rate


    def __iter__(self):
        self._iter_current_frame = -1
        return self

    def next(self):
        self._iter_current_frame += 1
        if self._iter_current_frame >= self.image_count:
            raise StopIteration
        else:
            return self.get_frame(self._iter_current_frame)

    def close(self):
        '''Closes the cine file.'''
        self.f.close()




    #---------------------------------------------------------------------------
    #These functions are not meant to be used externally, and so are
    #   undocumented.
    #---------------------------------------------------------------------------
    def unpack(self, fs, offset=None):
        if offset is not None:
            self.f.seek(offset)
        s = struct.Struct('<' + fs)
        vals = s.unpack(self.f.read(s.size))
        if len(vals) == 1:
            return vals[0]
        else:
            return vals


    def read_tagged_blocks(self):
        if not self.off_setup + self.length < self.off_image_offsets:
            return
        next_tag_exists = True
        next_tag_offset = 0
        while next_tag_exists:
            block_size, next_tag_exists = self._read_tag_block(next_tag_offset)
            next_tag_offset += block_size


    def _read_tag_block(self, off_set):
        self.file_lock.acquire()
        self.f.seek(self.off_setup + self.length + off_set)
        block_size = self.unpack(DWORD)
        b_type = self.unpack(WORD)
        more_tags = self.unpack(WORD)

        if b_type == 1004:
            # docs say to ignore range data
            # it seems to be a poison flag, if see this, give up tag parsing
            return block_size, 0

        try:
            d_name, d_type = TAGGED_FIELDS[b_type]

        except KeyError:
            #            print 'unknown type, find an updated version of file spec', b_type
            return block_size, more_tags

        if d_type == '':
            #            print "can't deal with  <" + d_name + "> tagged data"
            return block_size, more_tags

        s_tmp = struct.Struct('<' + d_type)
        if (block_size-8) % s_tmp.size != 0:
            #            print 'something is wrong with your data types'
            return block_size, more_tags

        d_count = (block_size-8)//(s_tmp.size)

        data = self.unpack('%d' % d_count + d_type)
        if not isinstance(data, tuple):
            # fix up data due to design choice in self.unpack
            data = (data, )

        # parse time
        if b_type == 1002 or b_type == 1001:
            data = [(datetime.datetime.fromtimestamp(d >> 32), float((FRACTION_MASK & d))/MAX_INT) for d in data]
        # convert exposure to seconds
        if b_type == 1003:
            data = [float(d)/(MAX_INT) for d in data]

        setattr(self, d_name, data)

        self.file_lock.release()
        return block_size, more_tags

    def read_header(self, fields, offset=0):
        self.f.seek(offset)
        for name, format in fields:
            setattr(self, name, self.unpack(format))




    def __unicode__(self):
        return self.fn

    def __str__(self):
        return unicode(self).encode('utf-8')

    __repr__ = __unicode__

    @property
    def trigger_time_p(self):
        '''Returns the time of the trigger, tuple of (datatime_object, fraction_in_ns)'''
        return datetime.datetime.fromtimestamp(self.trigger_time >> 32), float(FRACTION_MASK & self.trigger_time)/(MAX_INT)

    @property
    def hash(self):
        if self._hash is None:
            self._hash_fun()
        return self._hash

    def __hash__(self):
        return int(self.hash, base=16)

    def _hash_fun(self):
        """
        Generates the md5 hash of the header of the file.  Here the
        header is defined as everything before the first image starts.

        This includes all of the meta-data (including the plethora of
        time stamps) so this will be unique.
        """
        # get the file lock (so we don't screw up any other reads)
        self.file_lock.acquire()

        self.f.seek(0)
        max_loc = self.image_locations[0]
        md5 = hashlib.md5()

        chunk_size = 128*md5.block_size
        chunk_count = (max_loc//chunk_size) + 1

        for j in range(chunk_count):
            md5.update(self.f.read(128*md5.block_size))

        self._hash = md5.hexdigest()

        self.file_lock.release()

    def __eq__(self, other):
        return self.hash == other.hash

    def __ne__(self, other):
        return not self == other
