Module muvi
===========

Sub-modules
-----------
* muvi.cine
* muvi.sparse
* muvi.view

Functions
---------

    
`align_bdy(a, b)`
:   

    
`ceil_div(a, b)`
:   

    
`numpy_to_python_val(obj)`
:   

    
`open_4D_movie(fn, file_type=None, *args, **kwargs)`
:   

Classes
-------

`CineMovie(fn, fpv=512, offset=0, fps=None, info=None, clip=80, top=100, gamma=False)`
:   Generic class for working with volumetric movies.
    
    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
    
    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).
    
    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.
    
    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".
    
    Time scales are always specified in terms of seconds.
    
    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.
    
    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type
    
    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*
    
    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    ### Ancestors (in MRO)

    * muvi.VolumetricMovie

    ### Methods

    `close(self)`
    :

    `get_volume(self, i)`
    :

`HDF5Movie(fn, group='/VolumetricMovie')`
:   Generic class for working with volumetric movies.
    
    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
    
    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).
    
    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.
    
    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".
    
    Time scales are always specified in terms of seconds.
    
    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.
    
    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type
    
    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*
    
    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    ### Ancestors (in MRO)

    * muvi.VolumetricMovie

    ### Methods

    `close(self)`
    :

    `get_volume(self, i)`
    :

`MuviMovie(fn, info={})`
:   Generic class for working with volumetric movies.
    
    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
    
    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).
    
    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.
    
    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".
    
    Time scales are always specified in terms of seconds.
    
    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.
    
    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type
    
    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*
    
    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    ### Ancestors (in MRO)

    * muvi.VolumetricMovie

    ### Methods

    `close(self)`
    :

    `get_volume(self, i)`
    :

`SparseMovie(fn, group='/VolumetricMovie')`
:   Generic class for working with volumetric movies.
    
    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
    
    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).
    
    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.
    
    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".
    
    Time scales are always specified in terms of seconds.
    
    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.
    
    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type
    
    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*
    
    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    ### Ancestors (in MRO)

    * muvi.VolumetricMovie

    ### Methods

    `get_volume(self, i)`
    :

`VolumetricMovie(volumes, info={}, name='[VOLUME IN MEMORY]')`
:   Generic class for working with volumetric movies.
    
    Attributes
    ----------
    volumes : an iterable which contains the volumetric data.  Should support
        ``len``, be addressable by index, and return a numpy array
    info : a dictionary of metadata parameters, including perspective distortion
        and scale information; see documentation of recognized parameters below
    computed_info: a dictionary of metadata paremeters which are computed from
        the other parameters.  These parameters are not saved if the volume
        is written to disk
    metadata : a dicionary of arbitrary user metadata, will be saved as a JSON
        object if the volume is written to disk
    name : a string used to identify the volume; if loaded from a disk this
        should be the filename; otherwise it will be ``[VOLUME IN MEMORY]``.
    
    Members of info, computed_info, and metadata will be accesible as attributes
    of the class.  To alter these attibutes, however, the underlying
    dictionaries should be altered, rather than modifying the attributes of the
    class itself (which will not have the desired effect with regards to
    saving the volume).
    
    Each volume should be either 3 or 4 dimensional, where in the later case the
    fourth axis is the color dimension, typically specifying 1--4 planes.
    
    Length scales in the volume are specified in terms of an abitrary physical
    unit, whose scale is specified via "Lunit".
    
    Time scales are always specified in terms of seconds.
    
    The valid info parameters are documented below.  Each parameter must be
    expressable as a floating point number.
    
    General Info Parameters
    -----------------------
    - Lunit : the size of the distance unit in meters (e.g. Lunit = 1E-3 if the
        physical unit is mm (default), or 25.4E-3 if the unit is inches)
    - VPS : volumes per second
    - FPS : frames per second (if not specified, assumed to be VPS * [z depth];
        this assumes no dead time in the scan, which is unlikely!)
    - Lx, Ly, Lz : total length of each axis; *should be specified only for non-
        distorted volumes (or non-distorted axes)*
    - shape : tuple of ints.  The shape of each volume, automatically determined
        from the first volume.  Has 3-4 elemeters, of the form:
        (depth, height, width [, channels])
    - dtype : numpy data type
    
    Info Parameters for Scanning Slope-Distortion
    ---------------------------------------------
    - Dz : Displacement of camera from center of the volume.  Usually negative,
        since the camera should always be in the negative-z direction relative
        to the volume
    - Dx, Dy: Displacement of the scanning sheet axis from the center of the
        volume.  Only one should be specified, depending on the relevant axis
    - m1x : the slope of the ray leading the right edge of the volume.  Can be
        computed as -Lx / (2*Dz).
    - m1z : the slop of the ray leading to the back edge of the volume.  Can be
        computed as -Lz / (2*[Dx/Dy]), assuming the laser scanner is normal to
        the camera axis (as is normally the case).
    - m0z : the slope of the ray leading to the front edge of the volume.  If
        not specified, assumed to be -m1z.  *This should only be directly
        specified for scanning at an oblique angle*
    
    The additional parameters m0x, m0y, and m1y are determined automatically
    from the volume size.  These will be made available in the ``computed_info``
    attributes, or as attributes of the main class itself.

    ### Descendants

    * muvi.SparseMovie
    * muvi.CineMovie
    * muvi.MuviMovie
    * muvi.HDF5Movie

    ### Methods

    `get_info(self)`
    :

    `get_volume(self, i)`
    :

    `save(self, fn, file_type=None, print_status=False, start=0, end=None, skip=1, **kwargs)`
    :

    `validate_info(self)`
    :

`VolumetricMovieError(*args, **kwargs)`
:   Common base class for all non-exit exceptions.

    ### Ancestors (in MRO)

    * builtins.Exception
    * builtins.BaseException

`enum_range(obj, **kwargs)`
:   

    ### Ancestors (in MRO)

    * muvi.stat_range

    ### Methods

    `get_next(self)`
    :

`stat_range(start, stop=None, step=1, pre_message='', post_message='', length=40)`
:   

    ### Descendants

    * muvi.enum_range

    ### Methods

    `get_next(self)`
    :

    `update(self)`
    :