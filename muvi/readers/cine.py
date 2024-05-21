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

import numpy as np
# import numba
from .. import accel
import struct
from threading import Lock

FRACTION_MASK = (2**32-1)
MAX_INT = 2**32

uint8_t = 'B'
uint16_t = 'H'
uint32_t = 'I'
uint64_t = 'Q'
int16_t = 'h'
int32_t = 'l'
bool32_t = 'i'
float_t = 'f'
double_t = 'd'
TIME64 = 'Q'
RECT = '4i'
WBGAIN = '2f'
IMFILTER = '28i'

MAXLENDESCRIPTION = 4096
MAXLENDESCRIPTION_OLD = 121
OLDMAXFILENAME = 65

T64_F = lambda x: int(x) / 2.**32
T64_F_ms = lambda x: '%.3f' % (float(x.rstrip('L')) / 2.**32)
T64_S = lambda s: lambda t: time.strftime(s, time.localtime(float(t.rstrip('L'))/2.**32))

# #Processing the data in chunks keeps it in the L2 catch of the processor, increasing speed for large arrays by ~50%
# CHUNK_SIZE = 6 * 10**5 #Should be divisible by 3, 4 and 5!  This seems to be near-optimal.
#
# def ten2sixteen(a):
#     b = np.zeros(a.size//5*4, dtype='u2')
#
#     for j in range(0, len(a), CHUNK_SIZE):
#         (a0, a1, a2, a3, a4) = [a[j+i:j+CHUNK_SIZE:5].astype('u2') for i in range(5)]
#
#         k = j//5 * 4
#         k2 = k + CHUNK_SIZE//5 * 4
#
#         b[k+0:k2:4] = ((a0 & 0b11111111) << 2) + ((a1 & 0b11000000) >> 6)
#         b[k+1:k2:4] = ((a1 & 0b00111111) << 4) + ((a2 & 0b11110000) >> 4)
#         b[k+2:k2:4] = ((a2 & 0b00001111) << 6) + ((a3 & 0b11111100) >> 2)
#         b[k+3:k2:4] = ((a3 & 0b00000011) << 8) + ((a4 & 0b11111111) >> 0)
#
#     return b

# LUT used to convert 10 bit packed values to 16 bit.
LinLUT = np.array([
      2,    5,    6,    7,    8,    9,   10,   11,
     12,   13,   14,   15,   16,   17,   17,   18,
     19,   20,   21,   22,   23,   24,   25,   26,
     27,   28,   29,   30,   31,   32,   33,   33,
     34,   35,   36,   37,   38,   39,   40,   41,
     42,   43,   44,   45,   46,   47,   48,   48,
     49,   50,   51,   52,   53,   54,   55,   56,
     57,   58,   59,   60,   61,   62,   63,   63,
     64,   65,   66,   67,   68,   69,   70,   71,
     72,   73,   74,   75,   76,   77,   78,   79,
     79,   80,   81,   82,   83,   84,   85,   86,
     87,   88,   89,   90,   91,   92,   93,   94,
     94,   95,   96,   97,   98,   99,  100,  101,
    102,  103,  104,  105,  106,  107,  108,  109,
    110,  110,  111,  112,  113,  114,  115,  116,
    117,  118,  119,  120,  121,  122,  123,  124,
    125,  125,  126,  127,  128,  129,  130,  131,
    132,  133,  134,  135,  136,  137,  137,  138,
    139,  140,  141,  142,  143,  144,  145,  146,
    147,  148,  149,  150,  151,  152,  153,  154,
    156,  157,  158,  159,  160,  161,  162,  163,
    164,  165,  167,  168,  169,  170,  171,  172,
    173,  175,  176,  177,  178,  179,  181,  182,
    183,  184,  186,  187,  188,  189,  191,  192,
    193,  194,  196,  197,  198,  200,  201,  202,
    204,  205,  206,  208,  209,  210,  212,  213,
    215,  216,  217,  219,  220,  222,  223,  225,
    226,  227,  229,  230,  232,  233,  235,  236,
    238,  239,  241,  242,  244,  245,  247,  249,
    250,  252,  253,  255,  257,  258,  260,  261,
    263,  265,  266,  268,  270,  271,  273,  275,
    276,  278,  280,  281,  283,  285,  287,  288,
    290,  292,  294,  295,  297,  299,  301,  302,
    304,  306,  308,  310,  312,  313,  315,  317,
    319,  321,  323,  325,  327,  328,  330,  332,
    334,  336,  338,  340,  342,  344,  346,  348,
    350,  352,  354,  356,  358,  360,  362,  364,
    366,  368,  370,  372,  374,  377,  379,  381,
    383,  385,  387,  389,  391,  394,  396,  398,
    400,  402,  404,  407,  409,  411,  413,  416,
    418,  420,  422,  425,  427,  429,  431,  434,
    436,  438,  441,  443,  445,  448,  450,  452,
    455,  457,  459,  462,  464,  467,  469,  472,
    474,  476,  479,  481,  484,  486,  489,  491,
    494,  496,  499,  501,  504,  506,  509,  511,
    514,  517,  519,  522,  524,  527,  529,  532,
    535,  537,  540,  543,  545,  548,  551,  553,
    556,  559,  561,  564,  567,  570,  572,  575,
    578,  581,  583,  586,  589,  592,  594,  597,
    600,  603,  606,  609,  611,  614,  617,  620,
    623,  626,  629,  632,  635,  637,  640,  643,
    646,  649,  652,  655,  658,  661,  664,  667,
    670,  673,  676,  679,  682,  685,  688,  691,
    694,  698,  701,  704,  707,  710,  713,  716,
    719,  722,  726,  729,  732,  735,  738,  742,
    745,  748,  751,  754,  758,  761,  764,  767,
    771,  774,  777,  781,  784,  787,  790,  794,
    797,  800,  804,  807,  811,  814,  817,  821,
    824,  828,  831,  834,  838,  841,  845,  848,
    852,  855,  859,  862,  866,  869,  873,  876,
    880,  883,  887,  890,  894,  898,  901,  905,
    908,  912,  916,  919,  923,  927,  930,  934,
    938,  941,  945,  949,  952,  956,  960,  964,
    967,  971,  975,  979,  982,  986,  990,  994,
    998, 1001, 1005, 1009, 1013, 1017, 1021, 1025,
   1028, 1032, 1036, 1040, 1044, 1048, 1052, 1056,
   1060, 1064, 1068, 1072, 1076, 1080, 1084, 1088,
   1092, 1096, 1100, 1104, 1108, 1112, 1116, 1120,
   1124, 1128, 1132, 1137, 1141, 1145, 1149, 1153,
   1157, 1162, 1166, 1170, 1174, 1178, 1183, 1187,
   1191, 1195, 1200, 1204, 1208, 1212, 1217, 1221,
   1225, 1230, 1234, 1238, 1243, 1247, 1251, 1256,
   1260, 1264, 1269, 1273, 1278, 1282, 1287, 1291,
   1295, 1300, 1304, 1309, 1313, 1318, 1322, 1327,
   1331, 1336, 1340, 1345, 1350, 1354, 1359, 1363,
   1368, 1372, 1377, 1382, 1386, 1391, 1396, 1400,
   1405, 1410, 1414, 1419, 1424, 1428, 1433, 1438,
   1443, 1447, 1452, 1457, 1462, 1466, 1471, 1476,
   1481, 1486, 1490, 1495, 1500, 1505, 1510, 1515,
   1520, 1524, 1529, 1534, 1539, 1544, 1549, 1554,
   1559, 1564, 1569, 1574, 1579, 1584, 1589, 1594,
   1599, 1604, 1609, 1614, 1619, 1624, 1629, 1634,
   1639, 1644, 1649, 1655, 1660, 1665, 1670, 1675,
   1680, 1685, 1691, 1696, 1701, 1706, 1711, 1717,
   1722, 1727, 1732, 1738, 1743, 1748, 1753, 1759,
   1764, 1769, 1775, 1780, 1785, 1791, 1796, 1801,
   1807, 1812, 1818, 1823, 1828, 1834, 1839, 1845,
   1850, 1856, 1861, 1867, 1872, 1878, 1883, 1889,
   1894, 1900, 1905, 1911, 1916, 1922, 1927, 1933,
   1939, 1944, 1950, 1956, 1961, 1967, 1972, 1978,
   1984, 1989, 1995, 2001, 2007, 2012, 2018, 2024,
   2030, 2035, 2041, 2047, 2053, 2058, 2064, 2070,
   2076, 2082, 2087, 2093, 2099, 2105, 2111, 2117,
   2123, 2129, 2135, 2140, 2146, 2152, 2158, 2164,
   2170, 2176, 2182, 2188, 2194, 2200, 2206, 2212,
   2218, 2224, 2231, 2237, 2243, 2249, 2255, 2261,
   2267, 2273, 2279, 2286, 2292, 2298, 2304, 2310,
   2317, 2323, 2329, 2335, 2341, 2348, 2354, 2360,
   2366, 2373, 2379, 2385, 2392, 2398, 2404, 2411,
   2417, 2423, 2430, 2436, 2443, 2449, 2455, 2462,
   2468, 2475, 2481, 2488, 2494, 2501, 2507, 2514,
   2520, 2527, 2533, 2540, 2546, 2553, 2559, 2566,
   2572, 2579, 2586, 2592, 2599, 2605, 2612, 2619,
   2625, 2632, 2639, 2645, 2652, 2659, 2666, 2672,
   2679, 2686, 2693, 2699, 2706, 2713, 2720, 2726,
   2733, 2740, 2747, 2754, 2761, 2767, 2774, 2781,
   2788, 2795, 2802, 2809, 2816, 2823, 2830, 2837,
   2844, 2850, 2857, 2864, 2871, 2878, 2885, 2893,
   2900, 2907, 2914, 2921, 2928, 2935, 2942, 2949,
   2956, 2963, 2970, 2978, 2985, 2992, 2999, 3006,
   3013, 3021, 3028, 3035, 3042, 3049, 3057, 3064,
   3071, 3078, 3086, 3093, 3100, 3108, 3115, 3122,
   3130, 3137, 3144, 3152, 3159, 3166, 3174, 3181,
   3189, 3196, 3204, 3211, 3218, 3226, 3233, 3241,
   3248, 3256, 3263, 3271, 3278, 3286, 3294, 3301,
   3309, 3316, 3324, 3331, 3339, 3347, 3354, 3362,
   3370, 3377, 3385, 3393, 3400, 3408, 3416, 3423,
   3431, 3439, 3447, 3454, 3462, 3470, 3478, 3486,
   3493, 3501, 3509, 3517, 3525, 3533, 3540, 3548,
   3556, 3564, 3572, 3580, 3588, 3596, 3604, 3612,
   3620, 3628, 3636, 3644, 3652, 3660, 3668, 3676,
   3684, 3692, 3700, 3708, 3716, 3724, 3732, 3740,
   3749, 3757, 3765, 3773, 3781, 3789, 3798, 3806,
   3814, 3822, 3830, 3839, 3847, 3855, 3863, 3872,
   3880, 3888, 3897, 3905, 3913, 3922, 3930, 3938,
   3947, 3955, 3963, 3972, 3980, 3989, 3997, 4006,
   4014, 4022, 4031, 4039, 4048, 4056, 4064, 4095,
   4095, 4095, 4095, 4095, 4095, 4095, 4095, 4095
], dtype='u2')


class Cine:
    _MUVI_SUPPORTS_TONE_MAP = True

    _HEADER_FIELDS = (
        ('Type', '2s'),
        ('HeaderSize', uint16_t),
        ('Compression', uint16_t),
        ('Version', uint16_t),
        ('FirstMovieImage', int32_t),
        ('TotalImageCount', uint32_t),
        ('FirstImageNo', int32_t),
        ('ImageCount', uint32_t),
        ('OffImageHeader', uint32_t),
        ('OffSetup', uint32_t),
        ('OffImageOffsets', uint32_t),
        ('TriggerTime', TIME64),
    )

    _BITMAP_INFO_FIELDS = (
        ('Size', uint32_t),
        ('Width', int32_t),
        ('Height', int32_t),
        ('Planes', uint16_t),
        ('BitCount', uint16_t),
        ('Compression', uint32_t),
        ('SizeImage', uint32_t),
        ('XPelsPerMeter', int32_t),
        ('YPelsPerMeter', int32_t),
        ('ClrUsed', uint32_t),
        ('ClrImportant', uint32_t),
    )

    _SETUP_FIELDS = (
        ('FrameRate16', uint16_t),
        ('Shutter16', uint16_t),
        ('PostTrigger16', uint16_t),
        ('FrameDelay16', uint16_t),
        ('AspectRatio', uint16_t),
        ('Res7', uint16_t),
        ('Res8', uint16_t),
        ('Res9', uint8_t),
        ('Res10', uint8_t),
        ('Res11', uint8_t),
        ('TrigFrame', uint8_t),
        ('Res12', uint8_t),
        ('DescriptionOld', str(MAXLENDESCRIPTION_OLD) + 's'),
        ('Mark', uint16_t),
        ('Length', uint16_t),
        ('Res13', uint16_t),
        ('SigOption', uint16_t),
        ('BinChannels', int16_t),
        ('SamplesPerImage', uint8_t),
        ('BinName', 8 * '11s'),
        ('AnaOption', uint16_t),
        ('AnaChannels', int16_t),
        ('Res6', uint8_t),
        ('AnaBoard', uint8_t),
        ('ChOption', 8 * int16_t),
        ('AnaGain', 8 *  float_t),
        ('AnaUnit', 8 * '6s'),
        ('AnaName', 8 * '11s'),
        ('lFirstImage', int32_t),
        ('dwImageCount', uint32_t),
        ('nQFactor', int16_t),
        ('wCineFileType', uint16_t),
        ('szCinePath', 4 * (str(OLDMAXFILENAME) + 's')),
        ('Res14', uint16_t),
        ('Res15', uint8_t),
        ('Res16', uint8_t),
        ('Res17', uint16_t),
        ('Res18', double_t),
        ('Res19', double_t),
        ('Res20', uint16_t),
        ('Res1', int32_t),
        ('Res2', int32_t),
        ('Res3', int32_t),
        ('ImWidth', uint16_t),
        ('ImHeight', uint16_t),
        ('EDRShutter16', uint16_t),
        ('Serial', uint32_t),
        ('Saturation', int32_t),
        ('Res5', uint8_t),
        ('AutoExposure', uint32_t),
        ('bFlipH', bool32_t),
        ('bFlipV', bool32_t),
        ('Grid', uint32_t),
        ('FrameRate', uint32_t),
        ('Shutter', uint32_t),
        ('EDRShutter', uint32_t),
        ('PostTrigger', uint32_t),
        ('FrameDelay', uint32_t),
        ('bEnableColor', bool32_t),
        ('CameraVersion', uint32_t),
        ('FirmwareVersion', uint32_t),
        ('SoftwareVersion', uint32_t),
        ('RecordingTimeZone', int32_t),
        ('CFA', uint32_t),
        ('Bright', int32_t),
        ('Contrast', int32_t),
        ('Gamma', int32_t),
        ('Res21', uint32_t),
        ('AutoExpLevel', uint32_t),
        ('AutoExpSpeed', uint32_t),
        ('AutoExpRect', RECT),
        ('WBGgain', 4 * WBGAIN),
        ('Rotate', int32_t),
        ('WBView', WBGAIN),
        ('RealBPP', uint32_t),
        ('Conv8Min', uint32_t),
        ('Conv8Max', uint32_t),
        ('FilterCode', int32_t),
        ('FilterParam', int32_t),
        ('UF', IMFILTER),
        ('BlackCalSver', uint32_t),
        ('WhiteCalSver', uint32_t),
        ('GrayCalSver', uint32_t),
        ('bStampTime', bool32_t),
        ('SoundDest', uint32_t),
        ('FRPSteps', uint32_t),
        ('FRPImgNr', 16 * int32_t),
        ('FRPRate', 16 * uint32_t),
        ('FRPExp', 16 * uint32_t),
        ('FRPImageNr', 16 * int32_t),
        ('FRPRate', 16 * uint32_t),
        ('FRPExp', 16 * uint32_t),
        ('MCCnt', int32_t),
        ('MCPercent', 64 * float_t),
        ('CICalib', uint32_t),
        ('CalibWidth', uint32_t),
        ('CalibHeight', uint32_t),
        ('CalibRate', uint32_t),
        ('CalibExp', uint32_t),
        ('CalibEDR', uint32_t),
        ('CalibTemp', uint32_t),
        ('HeadSerial', 4 * uint32_t),
        ('RangeCode', uint32_t),
        ('RangeSize', uint32_t),
        ('Decimation', uint32_t),
        ('MasterSerial', uint32_t),
        ('Sensor', uint32_t),
        ('ShutterNs', uint32_t),
        ('EDRShutterNs', uint32_t),
        ('FrameDelayNs', uint32_t),
        ('ImPosXacq', uint32_t),
        ('ImPosYacq', uint32_t),
        ('ImWidthAcq', uint32_t),
        ('ImHeightAcq', uint32_t),
        ('Description', str(MAXLENDESCRIPTION) + 's'),
        ('RisingEdge', bool32_t),
        ('FilterTime', uint32_t),
        ('LongReady', bool32_t),
        ('ShutterOff', bool32_t),
        ('Res4', '16s'),
        ('bMetaWB', bool32_t),
        ('Hue', int32_t),
        ('BlackLevel', int32_t),
        ('WhiteLevel', int32_t),
        ('LensDescription', '256s'),
        ('LensAperture', float_t),
        ('LensFocusDistance', float_t),
        ('LensFocalLength', float_t),
        ('fOffset', float_t),
        ('fGain', float_t),
        ('fSaturation', float_t),
        ('fHue', float_t),
        ('fGamma', float_t),
        ('fGammaR', float_t),
        ('fGammaB', float_t),
        ('fFlat', float_t),
        ('fPedestalR', float_t),
        ('fPedestalG', float_t),
        ('fPedestalB', float_t),
        ('fChroma', float_t),
        ('ToneLabel', '256s'),
        ('TonePoints', int32_t),
        ('fTone', 4 * float_t),
    )

    def __init__(self, filename, output_bits=8, black_level=64,
                 white_level=4064, gamma=1.0, dark_clip=0.0, remap=True):
        '''Open a Vision Research Cine file.

        This reader does not return the raw values in the file, but rather
        converts it to normalized values in a user-defined way.

        Parameters
        ----------
        filename : str

        Keywords
        --------
        output_bits : int (default: 8)
            The number of bits in the output array.  Should be 8, 16, or 32.
            8 and 16 will produce 'u1' or 'u2' output, while 32 bit output
            is single precision float.
        black_level : int (default: 64)
            The level in the decoded cine file which is considered "black";
            this value will be converted to 0 in the output.  This value is
            standard for Phantom cameras, and should not need to be changed.
        white_level : int (default: 4064)
            The level in the decoded cine file which is considered "white";
            this value will be converted to 2^bits - 1 in the output.  This
            value is standard for Phantom cameras, and should not need to be
            changed.  (For floating point, it will be converted to 0 - 1)
        gamma : float (default: 1.0)
            The gamma value of the extracted images.  The actual intensity of
            the pixel is proportional to (output value)^(gamma)
        dark_clip : float (default: 0.0)
            Relative brightnesses below this value are converted to 0 in the
            output.  In other words, the clip value in the raw file is given
            by (black_level * (1 - dark_clip) + white_level * dark_clip).
            Note that this clip is applied *before* gamma correction!
        remap : bool (default: True)
            If False, ignore black_level, white_level, gamma, and dark_clip,
            and instead return the raw values from the CINE file.  Note that 
            you must have 16 bit output for this!  (The raw values are 
            between 0 and 4095.)
        '''
        self._file = open(filename, 'rb')
        self.filename = filename
        self.black_level = black_level
        self.white_level = white_level
        self.gamma = gamma
        self.dark_clip = dark_clip

        # Read the headers
        self.header = self._read_header(self._HEADER_FIELDS)
        if self.header['Version'] != 1:
            raise ValueError('version of CINE file should be 1 (found %d)' % self.header['Version'])

        self._file.seek(self.header['OffImageHeader'])
        self.bi = self._read_header(self._BITMAP_INFO_FIELDS)

        self._file.seek(self.header['OffSetup'])
        self.setup = self._read_header(self._SETUP_FIELDS)

        # Find the image locations
        self._file.seek(self.header['OffImageOffsets'])
        self.image_locations = self._unpack(self.header['ImageCount'] * uint64_t)

        # For convenience
        self.width = self.bi['Width']
        self.height = self.bi['Height']
        self.internal_bit_depth = self.setup['RealBPP']
        self.image_bytes = self.bi['SizeImage']

        # This reader supports 10 bit packed... so lets see if thats what we
        #   have here.
        if self.bi['Compression'] != 256:
            raise ValueError('this library only supports 10 bit packed CINE files')

        # Allows Cine object to be accessed from multiple threads
        self.file_lock = Lock()

        # Set up tone map and output info
        if output_bits == 8:
            self.output_type = 'u1'
            output_max = 2**output_bits - 1
        elif output_bits == 16:
            self.output_type = 'u2'
            output_max = 2**output_bits - 1
        elif output_bits == 32:
            self.output_type = 'f'
            output_max = 1.0
        else:
            raise ValueError('output_bits shoud be 8, 16, or 32 (found %s)' % output_bits)

        if remap:
            rel_map = (LinLUT.astype('f') - black_level) / (white_level - black_level)
            rel_map[np.where(rel_map < dark_clip)] = 0
            self.tone_map = (np.clip(rel_map, 0.0, 1.0)**(1./gamma) * output_max)
            if output_bits != 32:
                self.tone_map = (self.tone_map + 0.5).astype(self.output_type)
        else:
            if output_bits != 16:
                raise ValueError('Must be in 16 bit output mode if the data is not remapped!')
            self.tone_map = LinLUT

    def _unpack(self, format):
        s = struct.Struct('<' + format)
        vals = s.unpack(self._file.read(s.size))
        return vals[0] if len(vals) == 1 else vals

    def _read_header(self, fields, offset=None):
        return {name: self._unpack(format) for name, format in fields}

    def __getitem__(self, key):
        if type(key) == slice:
            return [self.get_frame(i) for i in range(len(self))[key] ]
        else:
            if key < 0:
                key = len(self) + key
            return self.get_frame(key)

    def __len__(self):
        return len(self.image_locations)

    def get_frame(self, i, raw=False):
        '''Get a single frame from the CINE

        Parameters
        ----------
        i : int
            Frame number

        Keywords
        --------
        raw : bool (default: False)
            If true, don't do the tone mapping, just return the raw data

        Returns
        -------
        frame : numpy array
            The tone mapped frame
        '''

        if self.internal_bit_depth == 10:
            self.file_lock.acquire()

            self._file.seek(self.image_locations[i])

            annotation_size = self._unpack(uint32_t)
            annotation = self._unpack('%ds' % (annotation_size - 8))
            image_size = self._unpack(uint32_t)

            raw_frame = np.frombuffer(self._file.read(self.image_bytes), dtype='u1')
            self.file_lock.release()

            if raw:
                return raw_frame
            else:
                frame = np.empty((self.height, self.width), dtype=self.output_type)
                accel.unpack_10b(raw_frame, self.tone_map, frame.reshape(-1))
                return frame

        else:
            raise ValueError('this library only supports 10 bit packed files')

    def pack_into_array(self, i, arr, offset=0, stride=1):
        '''Get a single frame from the Cine, and pack into an existing array.

        Parameters
        ----------
        i : int
            Frame number
        arr : numpy array
            Array in which to pack

        Keywords
        --------
        offset : int (default: 0)
            The integer offset to the start of the packing.  The array is
            flattened before accessing, so this is always a single number
        stride : int (default: 1)
            The spacing between values in the array.  Usually 1, but may be
            different if, for example, you are packing data into a single
            color channel of a multichannel image.
        '''
        if self.internal_bit_depth == 10:
            self.file_lock.acquire()

            self._file.seek(self.image_locations[i])

            annotation_size = self._unpack(uint32_t)
            annotation = self._unpack('%ds' % (annotation_size - 8))
            image_size = self._unpack(uint32_t)

            raw_frame = np.frombuffer(self._file.read(self.image_bytes), dtype='u1')

            self.file_lock.release()

            accel.unpack_10b(raw_frame, self.tone_map, arr.reshape(-1), offset=offset, stride=stride)

        else:
            raise ValueError('this library only supports 10 bit packed files')

    def close(self):
        if hasattr(self, 'f'):
            self._file.close()
        del self._file

    def get_frame_times(self):
        '''Read the image time block from the Cine file.  Will raise an error
        if the timestamp block isn't found.  (This shouldn't be a problem
        with files saved from recent versions of the software.)

        Returns
        -------
        timestamps : array of 64 bit unsigned integers
            The timestamp of each frame.  First 32 bits are the unix time stamp
            in seconds; final 32 bits are the fractional seconds bit.  (I.e.
            the whole timestamp is a fixed point number with 32 decimal bits.)
        '''
        self.file_lock.acquire()

        self._file.seek(self.header['OffSetup'] + self.setup['Length'])

        while self._file.tell() < self.header['OffImageOffsets']:
            block_size, block_type, res = struct.unpack('<LHH', self._file.read(8))
            # print(block_size, type)

            dat = self._file.read(block_size-8)
            if block_type == 1002: # Time stamp block!
                self.file_lock.release()
                return np.fromstring(dat, dtype='u8')
        else:
            self.file_lock.release()
            raise ValueError("Cine file does not contain timestamp block!")



    def __del__(self):
        self._file.close()


if __name__ == "__main__":
    import sys
    cine = Cine(sys.argv[1], dark_clip=10E-3, gamma=2.0, output_bits=8)

    N = 128
    frame_offset = 560*10
    i0 = 75200

    channels = 3
    vol = np.empty((N, cine.height, cine.width, channels), dtype='u1')
    frame_size = np.prod(vol.shape[1:])
    print(frame_size)

    for channel in range(3):
        start = i0 + channel * frame_offset
        for z in range(N):
            # vol[i] = cine[i + start]
            cine.pack_into_array(start + z, vol, offset=z*frame_size+channel, stride=channels)


    from muvi import VolumetricMovie
    from muvi.view.qtview import view_volume

    vm = VolumetricMovie([vol])
    vm.save('test.vti')

    view_volume(vm)

    # frames = np.array(cine[32000:33000])
    # print(frames.shape, frames.dtype)
    # mi = frames.mean((1, 2))
    # frame = frames[np.argmax(mi)]
    #
    # import matplotlib.pyplot as plt
    # import time
    #
    # print('Fraction empty: %.2f%%' % ((frame==0).mean()*100))
    #
    #
    # plt.subplot(121)
    # plt.imshow(frame)
    # plt.colorbar()
    #
    # plt.subplot(122)
    # plt.hist(frame.reshape(-1), bins=np.arange(50))
    # plt.yscale('log')
    # plt.show()


    # for header, name in (('header', 'Header'), ('bi', 'Bitmap Info'), ('setup', 'Camera Setup')):
    #     print('-'*79)
    #     print(name)
    #     print('-'*79)
    #
    #     for k, v in getattr(cine, header).items():
    #         v = str(v)
    #         if len(v) > 59:
    #             v = v[:56] + '...'
    #         print('%18s: %s' % (k, v))
    #
    # print(cine.image_locations[:20])
