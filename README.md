# Setup

## Mac / Linux

### Dependencies

In order to run the tools, you will need several Python packages installed, including:
  * numpy
  * pyQt5
  * pyopengl
  * lz4
  * numba

This is easiest to do with some sort of package manager; the utilities are being developed using Anaconda (https://www.anaconda.com/).  Note that the only Python 3.X is supported.

Assuming you have Anaconda installed, you can get the require packages with:

```shell
$ conda install numba lz4 pyopengl pyqt
```

### Installing the Package in Developer Mode


```shell
$ git clone https://github.com/klecknerlab/muvi
$ cd muvi
$ python setup.py develop
```

### Installing as a Regular Package

*This is not recommended, code is under active development!*

```shell
$ git clone https://github.com/klecknerlab/muvi
$ cd muvi
$ python setup.py install
```

## Windows

You should be able to download the package directly from the Git repo and run `setup.py` as above.  

Alternatively, if you have Atom (https://atom.io/) installed, you can use it to clone the repository and install this way.

To do this, from within Atom:
 - `Ctrl+Shift+P` (Note: `Command+Shift+P` on Mac)
 - Enter `Github: Clone` into the dialog, and hit enter
 - Add the address of this page (`https://github.com/klecknerlab/muvi.git`)
 - Select a folder, and navigate there in an Anaconda prompt
 - Run `setup.py` as above.

---

# Usage

## Simple Example

To create and view an example volume:

```shell
$ cd [MUVI DIR]/util
$ python generate_gyroid.py
$ python view.py gyroid.vti
```

![](gyroid.png)

Alternatively, there is a sample frame from a real experiment in the same directory, it can be viewed with:

```shell
$ cd [MUVI DIR]/util
$ python view.py ../samples/sample_frame.vti
```

## Converting a 2D Movie to 3D

To convert a Phantom CINE file to compressed VTI volume, you need to first generate a XML file which contains the VolumeProperties info.  This will allow you to define the number of frames per volume and other important properties.  To do this, copy the `samples/muvi_setup.xml` file (copied below) to the same directory as your 2D movie source files.  If you leave the name as is, it will be automatically used by every file conversion in that directory.  Alternatively, if you give it the same name as your source file (apart from the extension), this will be used for that specific file only.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<VolumeProperties>
    <!-- Number of frames in a volume -->
    <int name="Nz">256</int>
    <!-- Nx/Ny don't need to be specified, determined automatically -->

    <!-- Number of frames in a scan, must satisfy Ns >= Nz -->
    <int name="Ns">300</int>

    <!-- Dark clip value; default works well for Cine files -->
    <float name="dark_clip">0.005</float>

    <!-- If desired, you can also limit the upper end brightness.  This is the
    level in the raw file which is converted to the brightest value.  The
    default below is the correct value for Phantom cameras.  In most situations
    it should probably be left alone.  Note that you would also need to adjust
    the dark_clip if you change the white_level!
    -->
    <!-- <int name="white_level">4064</int>-->

    <!-- Gamma correction.  If gamma = 2, stored value is sqrt of input -->
    <float name="gamma">2.0</float>

    <!-- Physical size of volume on each axis -->
    <int name="Lx">100</int>
    <int name="Ly">100</int>
    <int name="Lz">100</int>

    <!-- Uncomment these lines if distortion correction will be used.  Note
        that the units of dx/dz are the same as L -->
    <!--
        <int name="dx">200</int>
        <int name="dz">200</int>
    -->

    <!-- Units of L -->
    <str name="units">mm</str>
</VolumeProperties>
```

Additionally, there are two utilities in the `util` directory which are useful for determining the properties of Cine files:

* `cine_histo.py`: This will sample the input cine file and create a histogram of the brightness levels.  Can be used to determine the appropriate `dark_clip` level.  (Although the default is usually sufficient.)
* `frame_delta.py`: This will analyze the first 1500 frames to try to find the "turn-around" of the laser scanner.  Can be used to determine the number of frames in a scan (`Ns`) and the offset (`offset`).

Once you've defined the volume properties, you can should be able to view your data in two ways:
1. Use `view.py` directly on the source file.  In general, this is rather slow, as it converts on the fly.  However, this can be very useful to verify your setup.
2. Run `convert_2D.py` on the input file (see below), and then view the resulting `.vti` conversion.

```shell
$ cd [MUVI DIR]/util
$ python convert_2D.py [INPUT FILE] [OUTPUT FILE]
$ python view.py [OUTPUT FILE]
```

By default, the output filename is the same as the input with an `.vti`
extension.  There are also more options in the conversion utility, which you
can view with:

```shell
$ cd [MUVI DIR]/util
$ python convert_2D.py --help
usage: convert_2D.py [-h] [-x XML] [-n NUMBER] infile [outfile]

Convert a CINE file to a VTI movie

positional arguments:
  infile                Input CINE file
  outfile               Output VTI file

optional arguments:
  -h, --help            show this help message and exit
  -x XML, --xml XML     XML file to use for conversion parameters
  -n NUMBER, --number NUMBER
                        Number of volumes to output (default: all)
```

# Features

This project is currently in active development.
A number of features are currently planned:

- [x] Volume rendering
- [ ] Isosurface viewing
- [x] VTI file writing
- [x] VTI file reading
- [ ] VTI reading from other sources
- [x] CINE conversion
- [ ] SEQ conversion
- [ ] Conversion GUI
- [ ] Display perspective correction
- [x] Multichannel support (partial)
- [ ] Support for perspective in old "S4D" format  
- [ ] Perspective correction on volumes in memory (for external processing)
- [ ] Image export (high res)
- [ ] Movie export
- [ ] Multithreaded display code (to avoid hang ups)
- [ ] Labelled axes
- [ ] Standalone Mac App
- [ ] Standalone Windows App
- [ ] Create Wiki with practical examples



---

# License

Copyright 2020 Dustin Kleckner

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
