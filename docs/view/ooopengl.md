Module muvi.view.ooopengl
=========================

Functions
---------

    
`dot(X, Y)`
:   

    
`mag(X)`
:   

    
`norm(X)`
:   

    
`normalize_basis(V)`
:   Converts basis to be right handed and orthonormal.
    
    Parameters
    ----------
    V : [3, 3] array
    
    Returns
    -------
    V : [3, 3] array

    
`raise_nice_compile_errors(err)`
:   

    
`rot_x(a, V=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]))`
:   

    
`rot_y(a, V=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]))`
:   

    
`rot_z(a, V=array([[1., 0., 0.],
       [0., 1., 0.],
       [0., 0., 1.]]))`
:   

    
`texture_from_array(arr, format=None, **kwargs)`
:   Create an OpenGL Texture Object from a numpy array.
    
    Parameters
    ----------
    arr : Numpy array.  Should have dimensions [y, x, d] or [z, y, x, d].  The
            last dimension is *always* the depth, which is used to determine
            the internal fromat (I, IA, RGB, or RGBA)
    format : directly specify the format.  Can be used to switch between (e.g.)
            IA and RG.  (default: determined from shape)
    
    Any extra keyword arguments are passed directly to Texture.

Classes
-------

`FrameBufferObject(width=100, height=100, depth=True, depth_type=GL_DEPTH_COMPONENT24, **kwargs)`
:   OpenGL FrameBuffer object
    
    Parameters
    ----------
    width : int
    height : int
    depth : if True, depth buffer will be attached.
    depth_type : OpenGL depth buffer type.  (default: GL_DEPTH_COMPONENT24)
    
    Any extra parameters are passed directly to the texture creation; in
    particular format and data_type.
    
    Note that creating a framebuffer will cause it to be bound.  If this is
    not desired then you must bind something else.

    ### Methods

    `bind(self)`
    :   Bind the framebuffer.

    `delete(self)`
    :   Delete the framebuffer and child objects.

    `resize(self, width, height)`
    :   Resize to new width and height.
        
        For safety, this actually destroys and recreates the framebuffer and
        all attachments.

`ShaderProgram(vertex_shader=None, fragment_shader=None, geometry_shader=None, uniforms={}, verify=True)`
:   OpenGL shader program object
    
    Parameters
    ----------
    vertex_shader : a string with the vertex shader (default: None).
    fragment_shader : a string with the fragment shader (default: None).
    geometry_shader : a string with the geometry shader (default: None).
    uniforms : if specified, set uniforms from a dictionary.
    verify : if True, verify code after setting uniforms (default: True)

    ### Methods

    `bind(self)`
    :   Use shader.

    `delete(self)`
    :   Delete shader object.

    `set_uniforms(self, **kwargs)`
    :   Set uniform values for shader as keyword arguments.
        
        Does not check that the shader is current -- be sure to bind it first!

`Texture(size, format=GL_RGBA, data_type=GL_UNSIGNED_BYTE, mag_filter=GL_LINEAR, min_filter=GL_LINEAR, wrap_s=None, wrap_t=None, wrap_r=None, wrap=None, target=None, source=None, internal_format=None)`
:   OpenGL Texture Object
    
    Parameters
    ----------
    size : the shape tuple as (h, w) or (d, h, w)
    format : texture color format (default: GL_RGBA)
    data_type : texture data type (default: GL_UNSIGNED_BYTE)
    mag_filter : magnification filter (default: GL_LINEAR)
    min_filter : minification filter (default: GL_LINEAR)
    target : type of texture (default: GL_TEXTURE_2D or GL_TEXTURE_3D, depending on shape)
    source : optional data source, i.e. byte stream of numpy array (default: None)
    wrap_s : texture wrap type in s (x) (default: None, not set)
    wrap_t : texture wrap type in t (y) (default: None, not set)
    wrap_r : texture wrap type in r (z) (default: None, not set)
    wrap : texture wrap for all coodinates (default: None); if wrap_s/t/r are
        set they override this.
    
    If creating from a numpy texture, using texture_from_array is easier!
    
    Note that creating a texture will cause it to be bound.

    ### Methods

    `bind(self)`
    :   Bind the texture.

    `delete(self)`
    :   Delete the texture id.

    `replace(self, arr)`
    :   Replace the texture data with a new numpy array.
        
        Will also bind the texture to the current texture unit.