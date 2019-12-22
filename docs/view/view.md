Module muvi.view.view
=====================
This file contains the openGL routines for drawing a volume, but does not
actually handle any of window managment.

The qtview module contains an example of a routine that uses these functions.

Functions
---------

    
`in_module_dir(fn)`
:   

Classes
-------

`LogValue(val, minval=None, maxval=None, logbase=2, steps_per_base=2)`
:   

    ### Methods

    `dec(self)`
    :

    `inc(self)`
    :

`Value(val, minval=None, maxval=None, step=1)`
:   

    ### Methods

    `dec(self)`
    :

    `inc(self)`
    :

`View(volume=None, R=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]), center=None, fov=30, X0=array([0., 0., 0.]), X1=None, width=100, height=100, os_width=1500, os_height=1000, scale=None)`
:   An object which represents and renders views of a 3D volumes.
    
    Parameters
    ----------
    volume : 3D volumetric movie to be displayed.
    R : [3, 3] rotation matrix; will be automatically converted to orthonormal.
    center : [3] vector, center of view.
    fov : vertical field of view (default: 45).  If <= 0, flat projection is
            used instead.
    scale : overall scale of object in display.  If set to 1 / volume height,
            will fill the vertical axis of the window
            (default: None, set on first display.)
    X0 : [3] vector, lower edge of volume view (default: [0, 0, 0]).
    X1 : [3] vector, upper edge of volume view (default: None, set on
            first display to be entire volume)
    width : int (default: 100), the width of the view
    height : int (default: 100), the height of the view
    os_width : int (default: 1000), the width of the off-screen view (used for screenshots)
    os_height : int (default: 1000), the height of the off-screen view

    ### Methods

    `attach_volume(self, volume)`
    :   Attach a VolumetricMovie to the view.

    `autoscale(self)`
    :   Sets limits of volume and scale if not already defined.

    `draw(self, z0=0.01, z1=10.0, offscreen=False, save_image=False)`
    :   Draw the volume, creating all objects as required.  Has two
        parameters, but normally the defaults are fine.
        
        Parameters
        ----------
        z0 : float, the front clip plane (default: 0.01)
        z1 : float, the back clip plane (default: 10.0)
        offscreen : bool, if True, renders to the offscreen buffer instead of
           the onscreen one.  (Usually used for screenshots/movies.)

    `fov_correction(self)`
    :   Computes the half height of the viewport in OpenGL units.

    `frame(self, frame)`
    :

    `mouse_move(self, x, y, dx, dy)`
    :   Handles mouse move events, rotating the volume accordingly.
        
        Parameters
        ----------
        dx : the x motion since the last event in viewport pixels.
        dy : the y motion since the last event in viewport pixels.
        
        Note that the buttons_pressed property should also be directly
        updated by the window manager.

    `resize(self, width, height)`
    :   Convenience command to update the width and height.

    `rot_x(self, a)`
    :   Rotate view around x axis by a given angle (in radians).

    `rot_y(self, a)`
    :   Rotate view around y axis by a given angle (in radians).

    `rot_z(self, a)`
    :   Rotate view around z axis by a given angle (in radians).

    `select_volume_shader(self, show_isosurface=True, color_function='rrr', opacity_function='r', perspective_model='uncorrected', show_grid=False)`
    :   Compiles the volume render shader with the desired options.
        
        Parameters
        ----------
        show_isosurface : bool (default: True).  If True, isosurface is
            displated.
        color_function : string (default: 'rrr').  Select the function used to
            translate the values in the raw image data to RGB.
        opacity_function : string (default: 'r').  Select the channel used for
            opacity data.
        perspective_model : string (default: 'uncorrected').  Select the
            perspective correction model used by the renderer.  Currently
            no other options are implemented
        show_grid : bool (default: False).  If true, show grid on top of
            isosurface.
        
        Note that the last three options are passed directly to
        ``VolumeShader.compile``, which contains more details on their
        effects.

    `units_per_pixel(self)`
    :   Returns the viewport scale in image units per pixel.

    `update_uniforms(self, **kwargs)`
    :   Function to update uniforms associated with volume rendering shader.
        
        The variable names and values should be passed as keyword arguments.
        
        See ``volume_shadre.glsl`` for a list of valid parameters.

    `update_view_settings(self, **kwargs)`
    :   Updates the view settings, recompiling the shader if needed.
        
        Accepts keyword arguments which are passed either to ``update_uniforms``
        or ``select_volume_shader``, as needed.  This is a convenience function
        which treats all view variables the same to ease front end creation.

`VolumeShader(source)`
:   Creates a volume shader from an appropriate GLSL fragment shader.
    
    The Python code handles inserting the appropriate perspective correction
    functions, textures, and key variables to provide consistent implementation.
    
    Parameters
    ----------
    source : filename for external fragment shader code, or string with shader
        code.  If the input contains a line ending, it will assume it is code,
        otherwise it is treated as a filename.
    
    Note that no code will not get compiled until the "compile" method is
        called.

    ### Methods

    `compile(self, color_function='rrr', opacity_function='r', perspective_model='uncorrected', defines='')`
    :   Compile and return a shader with the options.
        
        Parameters
        ----------
        color_function : length 3 string with GLSL swizzle specifiction.  I.e.
            "rgb" would copy the color channels as is , and "rrr" would set R/G/B
            to all be based on the red channel.  (default: "rrr")
        opacity_function : length 1 string with GLSL swizzle specification for opacity
            source channel.  (default: "r")
        perspective_model : string with model name.  Currently not
            implemeneted.  (default: "uncorrected")
        defines : string which gets appended to the beginning of the code.
            Intended to be used to pass defines which control the render
            pathway.
        
        Returns
        -------
        shader : ShaderProgram object with compiled code.  May be reused from
            previous compiles.