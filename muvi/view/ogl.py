# from muvi.view.ooopengl import ShaderProgram

try:
    # this fails in <= 2020 versions of Python on OS X 11.x
    from OpenGL import GL
except ImportError:
    # print('Patching ctypes for OS X 11.x...')
    from ctypes import util
    orig_util_find_library = util.find_library
    def new_util_find_library(name):
        res = orig_util_find_library(name)
        if res: return res
        return f'/System/Library/Frameworks/{name}.framework/{name}'
    util.find_library = new_util_find_library

    from OpenGL import GL


from OpenGL.GL import shaders
import numpy as np
import numba
import re
import ctypes
import os
import json
import base64
from PIL import Image

#--------------------------------------------------------
# Some basic vector operations -- pretty self explanatory
#--------------------------------------------------------


def mag(X):
    return np.sqrt(dot(X, X))

def mag1(X):
    return np.sqrt(dot(X, X))[..., np.newaxis]

def norm(X):
    return np.asarray(X) / mag1(X)

def dot(X, Y):
    return (np.asarray(X)*Y).sum(-1)

def dot1(X, Y):
    return (np.asarray(X)*Y).sum(-1)[..., np.newaxis]

cross = np.cross

GL_VEC_TYPES = {
    GL.GL_SAMPLER_1D: ('i4', (1,), "GL_SAMPLER_1D", GL.GL_INT, GL.glUniform1iv),
    GL.GL_SAMPLER_2D: ('i4', (1,), "GL_SAMPLER_2D", GL.GL_INT, GL.glUniform1iv),
    GL.GL_SAMPLER_2D_RECT: ('i4', (1,), "GL_SAMPLER_2D_RECT", GL.GL_INT, GL.glUniform1iv),
    GL.GL_SAMPLER_3D: ('i4', (1,), "GL_SAMPLER_3D", GL.GL_INT, GL.glUniform1iv),
}

for gl_var_type, np_dtype in [('GL_FLOAT', 'f'), ('GL_DOUBLE', 'd'), \
        ('GL_INT', 'i4'), ('GL_UNSIGNED_INT', 'u4')]:
    for gl_shape, np_shape in [('', (1, )), \
            ('_VEC2', (2, )), ('_VEC3', (3, )), ('_VEC4', (4, )), \
            ('_MAT2', (2, 2)), ('_MAT3', (3, 3)), ('_MAT4', (4, 4))]:

        gl_type = gl_var_type + gl_shape
        const = getattr(GL, gl_type, None)
        if const is None:
            continue

        func_type = 'Matrix' if len(np_shape) > 1 else ''

        ut = np_dtype[:1]
        if ut == 'u':
            ut = 'ui'
        uf = getattr(GL, 'glUniform' + func_type + str(np_shape[0]) + ut + 'v', None)

        if uf is None:
            # print(gl_type, 'glUniform' + func_type + str(np_shape[0]) + np_dtype[:1] + 'v')
            continue
        # Matrix uniform set values require an additional "transpose" parameter
        # We can just transpose the numpy matrix, so we won't use this!
        if len(np_shape) > 1:
            def _uf(i, n, v, uf=uf):
                uf(i, n, GL.GL_FALSE, v)
        else:
            _uf = uf

        GL_VEC_TYPES[const] = (np_dtype, np_shape, gl_type, getattr(GL, gl_var_type), _uf)


def raise_nice_compile_errors(err):
    if err.args[0].startswith('Shader compile failure'):
        label = {
            GL.GL_VERTEX_SHADER: "Vertex ",
            GL.GL_FRAGMENT_SHADER: "Fragment ",
            GL.GL_GEOMETRY_SHADER: "Geometry "
        }.get(err.args[2], "")
        errstr = "%sshader compile failure.\n" % label

        marked_lns = []
        for ln, msg in re.findall(r'ERROR:\s+[0-9]+:([0-9]+):\s+(.+?)\\n', err.args[0]):
            ln = int(ln)
            marked_lns.append(ln)
            errstr = errstr + "%5d: %s\n" % (ln, msg)

        errstr = errstr + "-"*79+"\n"
        errstr = errstr + "  source\n"
        errstr = errstr + "-"*79+"\n"

        for ln, s in enumerate(err.args[1][0].decode('utf-8').splitlines()):
            ln += 1
            if ln in marked_lns: errstr = errstr + "%4d | %s\n" % (ln, s)

        errstr = errstr + "-"*79+"\n"

        err = RuntimeError(errstr)

    raise err

class UniformError(Exception):
    pass

CURRENT_SHADER_ID = None

def useProgram(index):
    global CURRENT_SHADER_ID

    GL.glUseProgram(index)
    CURRENT_SHADER_ID = index

class ShaderLinkError(Exception):
    pass

class ShaderProgram:
    def __init__(self, vertexShader=None, fragmentShader=None, geometryShader=None):
        self.id = GL.glCreateProgram()

        s = []

        try:
            if vertexShader is not None:
                s.append(shaders.compileShader(vertexShader, GL.GL_VERTEX_SHADER))
            if fragmentShader is not None:
                s.append(shaders.compileShader(fragmentShader, GL.GL_FRAGMENT_SHADER))
            if geometryShader is not None:
                s.append(shaders.compileShader(geometryShader, GL.GL_GEOMETRY_SHADER))
        except RuntimeError as err:
            raise_nice_compile_errors(err)

        for ss in s:
            GL.glAttachShader(self.id, ss)

        GL.glLinkProgram(self.id)

        if GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS) == GL.GL_FALSE:
            raise ShaderLinkError(GL.glGetProgramInfoLog(self.id).decode('utf8'))

        # Delete references to shader objects; they will be kept in memory
        # so long as the program still exists.
        for ss in s:
            GL.glDeleteShader(ss)


        # Set up the uniform information
        self._uniform = {}
        self._uniformNames = set()
        self._delayedUniforms = {}
        # Used to store uniform values set when the shader is unbound --
        #   automatically updated upon binding!

        for i in range(GL.glGetProgramiv(self.id, GL.GL_ACTIVE_UNIFORMS)):
            name, n_elem, t = GL.glGetActiveUniform(self.id, i)
            name = name.decode('utf8')
            np_dtype, np_shape, gl_type, gl_var_type, uf = GL_VEC_TYPES[t]
            np_size = n_elem * np.prod(np_shape)
            loc = GL.glGetUniformLocation(self.id, name)
            self._uniform[name] = (loc, n_elem, np_dtype, np_size, np_shape, uf)
            self._uniformNames.add(name)

        # !! New version handles attributes explicitly, rather than getting them
        #  from shader info !!

        # # Set up the vertex attribute information
        # self._attrib = {}
        # # Note that glGetActiveAttrib doesn't have a nice pyopengl binding,
        # #  so we have to work a little harder ):
        # bufSize = GL.glGetProgramiv(self.id, GL.GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
        # length = GL.GLint()
        # size = GL.GLint()
        # t = GL.GLenum()
        # name = (GL.GLchar * bufSize)()
        #
        # for i in range(GL.glGetProgramiv(self.id, GL.GL_ACTIVE_ATTRIBUTES)):
        #     GL.glGetActiveAttrib(self.id, i, bufSize, length, size, t, name)
        #     np_dtype, np_shape, gl_type, gl_var_type, uf = GL_VEC_TYPES[t.value]
        #     ns = name.value.decode('utf8')
        #
        #     loc = GL.glGetAttribLocation(self.id, ns)
        #     items = int(size.value * np.prod(np_shape))
        #
        #     self._attrib[ns] = (loc, np_dtype, items)

    def bind(self):
        '''Use shader, and update uniforms that were set previously.'''
        useProgram(self.id)

        if self._delayedUniforms:
            for k, v in self._delayedUniforms.items():
                self._setUniform(k, v)

            self._delayedUniforms = {}

    def unbind(self):
        useProgram(0)

    def __enter__(self):
        self.bind()
        return self

    def __exit__(self, type, value, traceback):
        self.unbind()

    def delete(self):
        '''Delete shader object.'''
        if hasattr(self, 'id'):
            GL.glDeleteProgram(self.id)
            del self.id

    def _setUniform(self, key, val):
        loc, n_elem, np_dtype, np_size, np_shape, uf = self._uniform[key]

        val = np.asarray(val, np_dtype, order='C')
        if val.size != np_size:
            raise UniformError(f"Value for uniform '{key}' must have {np_size} total elements (found {val.size})")

        uf(loc, n_elem, val)

    def __setitem__(self, key, val, ignore=False):
        if key not in self._uniformNames:
            if ignore:
                return
            else:
                raise UniformError(f"'{key}' is not a valid uniform variable for this shader")

        if self.id != CURRENT_SHADER_ID:
            self._delayedUniforms[key] = val
        else:
            self._setUniform(key, val)

    def update(self, uniforms, ignore=False):
        for k, v in uniforms.items():
            self.__setitem__(k, v, ignore)

    def __getitem__(self, key):
        if self.id != CURRENT_SHADER_ID:
            raise UniformError('you must bind the shader before getting uniforms!')

        prop = self._uniform.get(key, None)
        if prop is None:
            raise UniformError(f"'{key}' is not a valid uniform variable for this shader")

        loc, n_elem, np_dtype, np_size, np_shape, uf = prop

        # Note: the get function hasn't been optimized for speed (i.e. the
        # read function is not cached).  The assumption is that this is used
        # infrequently, or only in testing.

        ut = np_dtype[:1]
        if ut == 'u':
            ut = 'ui'

        func = getattr(GL, 'glGetUniform' + ut + 'v')
        if n_elem != 1:
            np_shape = (n_elem, ) + np_shape
        dat = np.ones(np_shape, dtype=np_dtype, order='C')
        func(self.id, loc, ctypes.c_void_p(dat.ctypes.data))

        return dat

    def keys(self):
        return self._uniformNames

    def validate(self):
        GL.glValidateProgram(self.id)

        validation = GL.glGetProgramiv(self.id, GL.GL_VALIDATE_STATUS)
        if validation == GL.GL_FALSE:
            raise RuntimeError(
                """Validation failure (%s): %s"""%(
                validation,
                shaders.glGetProgramInfoLog(self.id),
            ))

        link_status = GL.glGetProgramiv(self.id, GL.GL_LINK_STATUS)
        if link_status == GL.GL_FALSE:
            raise RuntimeError(
                """Link failure (%s): %s"""%(
                link_status,
                shaders.glGetProgramInfoLog(self.id),
            ))


class AttributeError(Exception):
    pass


class VertexArray:
    def __init__(self, data, numElements=None, usage=GL.GL_DYNAMIC_DRAW):
        if isinstance(data, np.ndarray):
            spec = data.dtype
        elif isinstance(data, np.dtype):
            spec = data
        else:
            raise ValueError('first argument should be numpy data type or array')

        self.id = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self.id)

        self.buffer = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)

        self.dtype = spec
        self.itemsize = spec.itemsize

        if isinstance(data, np.ndarray):
            GL.glBufferData(GL.GL_ARRAY_BUFFER, data, usage)
            self._len = len(data)
        else:
            GL.glBufferData(GL.GL_ARRAY_BUFFER, spec.itemsize * numElements, None, usage)
            self._len = numElements
            if numElements is None:
                raise ValueError('if VertexArray is created with numpy data type, must specify number of elements as second argument')

        for loc, name in enumerate(spec.names):
            if name.startswith('_'):
                continue

            dt, offset = spec.fields[name]
            glType = npToGl(dt.base)
            num = int(np.prod(dt.shape, dtype='i'))

            GL.glEnableVertexAttribArray(loc)

            if glType in {GL.GL_INT, GL.GL_UNSIGNED_INT}:
                GL.glVertexAttribIPointer(loc, num, glType, self.itemsize,
                    GL.GLvoidp(offset))
            else:
                GL.glVertexAttribPointer(loc, num, glType, GL.GL_FALSE,
                    self.itemsize, GL.GLvoidp(offset))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def __len__(self):
        return self.elements

    def update(self, data, start=0):
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, start * self.itemsize, data)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def delete(self):
        if hasattr(self, 'id'):
            GL.glDeleteVertexArrays(1, [self.id])
            del self.id

        if hasattr(self, 'buffer'):
            GL.glDeleteBuffers(1, [self.buffer])
            del self.buffer

        if hasattr(self, 'elements'):
            GL.glDeleteBuffers(1, [self.elements])
            del self.elements

    def attachElements(self, elements, mode=GL.GL_TRIANGLES, usage=GL.GL_STATIC_DRAW):
        if hasattr(self, 'elements'):
            GL.glDeleteBuffers(1, [self.elements])

        self.elements = GL.glGenBuffers(1)
        self.mode = mode
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.elements)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, elements, usage)
        self.numElements = elements.size
        self.elementsType = npToGl(elements.dtype)
        self.elementsItemSize = elements.itemsize

    def draw(self, num=None, offset=0):
        if not hasattr(self, 'elements'):
            raise ValueError('to draw, first attach elements!')

        GL.glBindVertexArray(self.id)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.elements)
        if num is None:
            num = self.numElements - offset
        GL.glDrawElements(self.mode, num, self.elementsType, GL.GLvoidp(offset*self.elementsItemSize))

    def drawArrays(self, mode, first, count):
        GL.glBindVertexArray(self.id)
        GL.glDrawArrays(mode, first, count)


_npToGl = {
    np.dtype('int8'): GL.GL_BYTE,
    np.dtype('uint8'): GL.GL_UNSIGNED_BYTE,
    np.dtype('int16'): GL.GL_SHORT,
    np.dtype('uint16'): GL.GL_UNSIGNED_SHORT,
    np.dtype('int32'): GL.GL_INT,
    np.dtype('uint32'): GL.GL_UNSIGNED_INT,
    np.dtype('f'): GL.GL_FLOAT
}

def npToGl(t):
    t = np.dtype(t)
    if t not in _npToGl:
        raise ValueError('Data type of array should be one of: [%s] (found %s)', (','.join(_npToGl.keys()), t))
    else:
        return _npToGl[t]


_glToNp = {v:k for k, v in _npToGl.items()}

def glToNp(t):
    if t not in _glToNp:
        raise ValueError('Data type of array should be one of: [%s] (found %s)', (','.join(dts.keys()), arr.dtype))
    else:
        return _glToNp[t]


class Texture:
    '''OpenGL Texture Object

    Parameters
    ----------
    size : the shape tuple as (h, w) or (d, h, w)
    format : texture color format (default: GL_RGBA)
    dataType : texture data type (default: GL_UNSIGNED_BYTE)
    magFilter : magnification filter (default: GL_LINEAR)
    minFilter : minification filter (default: GL_LINEAR)
    target : type of texture (default: depends on shape)
    source : optional data source, i.e. byte stream of numpy array (default: None)
    wrapS : texture wrap type in s (x) (default: None, not set)
    wrapT : texture wrap type in t (y) (default: None, not set)
    wrapR : texture wrap type in r (z) (default: None, not set)
    wrap : texture wrap for all coodinates (default: None); if wrapS/T/R are
        set they override this.

    If creating from a numpy texture, using textureFromArray is easier!

    Note that creating a texture will cause it to be bound.
    '''

    def __init__(self, size, format=GL.GL_RGBA, dataType=GL.GL_UNSIGNED_BYTE,
                 magFilter=GL.GL_LINEAR, minFilter=GL.GL_LINEAR, wrapS=None,
                 wrapT=None, wrapR=None, wrap=None, target=None, source=None,
                 internalFormat=None):

        # print(size)
        if target is None:
            if len(size) == 1:
                target = GL.GL_TEXTURE_1D
            elif len(size) == 2:
                target = GL.GL_TEXTURE_2D
            elif len(size) == 3:
                target = GL.GL_TEXTURE_3D
            else:
                raise ValueError('size should have 1--3 elements (found %s)' % len(size))

        self.target = target

        if wrapS is None:
            wrapS = wrap
        if wrapT is None:
            wrapT = wrap
        if wrapR is None:
            wrapR = wrap

        self.id = GL.glGenTextures(1)

        # Let the user set the active texture
        # glActiveTexture(GL_TEXTURE0)

        self.bind()
        self.format = format
        self.dataType = dataType
        self.size = size
        if internalFormat is None:
            self.internalFormat = format
        else:
            self.internalFormat = internalFormat

        if len(size) == 1:
            GL.glTexImage1D(self.target, 0, self.internalFormat, size[0], 0,
                         format, dataType, source)
        elif len(size) == 2:
            GL.glTexImage2D(self.target, 0, self.internalFormat, size[0],
                         size[1], 0, format, dataType, source)
        elif len(size) == 3:
            GL.glTexImage3D(self.target, 0, self.internalFormat, size[0],
                         size[1], size[2], 0, format, dataType, source)
        else: raise ValueError('size should have 1--3 elements (found %s)' % len(size))

        if magFilter is not None:
            GL.glTexParameteri(self.target, GL.GL_TEXTURE_MAG_FILTER, magFilter)
        if minFilter is not None:
            GL.glTexParameteri(self.target, GL.GL_TEXTURE_MIN_FILTER, minFilter)
        if wrapS is not None:
            GL.glTexParameteri(self.target, GL.GL_TEXTURE_WRAP_S, wrapS)
        if len(size) >= 2 and wrapT is not None:
            GL.glTexParameteri(self.target, GL.GL_TEXTURE_WRAP_T, wrapT)
        if len(size) >= 3 and wrapR is not None:
            GL.glTexParameteri(self.target, GL.GL_TEXTURE_WRAP_R, wrapR)

    def bind(self):
        '''Bind the texture.'''
        GL.glBindTexture(self.target, self.id)

    def delete(self):
        '''Delete the texture id.'''
        GL.glDeleteTextures([self.id])

    def replace(self, arr):
        '''Replace the texture data with a new numpy array.

        Will also bind the texture to the current texture unit.'''
        self.bind()

        # print(self.size, arr.shape)
        if type(arr) == bytes:
            pass
            # if (np.prod(self.size) != len(arr)):
                # raise ValueError('input buffer must have sime size as texture!')
        else:
            if (self.size != arr.shape[:len(self.size)][::-1]):
                raise ValueError('new array must have same shape as texture!')

        if len(self.size) == 1:
            GL.glTexSubImage1D(self.target, 0, 0, self.size[0],
                         self.format, self.dataType, arr)


        if len(self.size) == 2:
            GL.glTexImage2D(self.target, 0, self.internal_format,
                         self.size[0], self.size[1], 0, self.format,
                         self.dataType, arr)

        elif len(self.size) == 3:
            # glTexImage3D(self.target, 0, self.internal_format,
            #              self.size[0], self.size[1], self.size[2], 0,
            #              self.format, self.data_type, arr)
            GL.glTexSubImage3D(self.target, 0, 0, 0, 0,
                         self.size[0], self.size[1], self.size[2],
                         self.format, self.dataType, arr)

        # If we don't flush, the update may not show up until the next frame!
        GL.glFlush()


    def __array__(self):
        type = glToNp(self.dataType)

        channels = {
            GL.GL_RED: 1,
            GL.GL_RG: 2,
            GL.GL_RGB: 3,
            GL.GL_RGBA: 4,
            GL.GL_DEPTH_COMPONENT: 1
        }

        if self.format not in channels:
            raise ValueError('to convert to a numpy array, texture format must (RED, RG, RGB, RGBA, or DEPTH_COMPONENT)')

        channels = channels[self.format]
        dtype = glToNp(self.dataType)

        shape = self.size[::-1]
        if channels > 1: shape += (channels, )

        data = np.empty(shape, dtype=dtype)

        self.bind()
        GL.glGetTexImage(self.target, 0, self.format, self.dataType, data)

        return data


def textureFromArray(arr, format=None, **kwargs):
    '''Create an OpenGL Texture Object from a numpy array.

    Parameters
    ----------
    arr : Numpy array.  Should have dimensions [y, x, d] or [z, y, x, d].  The
            last dimension is *always* the depth, which is used to determine
            the internal fromat (R, RG, RGB, or RGBA)
    format : directly specify the format.  Can be used to switch between (e.g.)
            RG and IA.  (default: determined from shape)

    Any extra keyword arguments are passed directly to Texture.
    '''

    formats = {
        1: GL.GL_RED,
        2: GL.GL_RG,
        3: GL.GL_RGB,
        4: GL.GL_RGBA
    }

    dataType = npToGl(arr.dtype)
    if format is None:
        if arr.shape[-1] not in formats:
            raise ValueError('Last dimension should have length 1-4, indicating number of planes (found %s)', (arr.shape[-1]))
        format = formats[arr.shape[-1]]

    arr = np.asarray(arr, order='c')

    return Texture(arr.shape[:-1][::-1], format=format, dataType=dataType,
                   source=arr, **kwargs)


class FrameBuffer:
    '''OpenGL FrameBuffer object

    Parameters
    ----------
    width : int
    height : int
    depth : bool
        If True, depth buffer will be attached.
    depthType : OpenGL depth buffer type (default: GL_DEPTH_COMPONENT24)
    depthTexture : bool
        If True, the depth buffer is rendered to a texture, instead of a
        renderbuffer.  This is a bit slower, but allows you to access the
        depth in a late stage.

    Any extra parameters are passed directly to the texture creation; in
    particular format and dataType.

    Note that creating a framebuffer will cause it to be bound.  If this is
    not desired then you must bind something else.
    '''

    def __init__(self, width=100, height=100, depth=True,
                 depthType=GL.GL_DEPTH_COMPONENT24, depthTexture=False,
                 **kwargs):
        self.textureKwargs = kwargs.copy()
        self.depth = bool(depth)
        self.depthType = depthType
        self._depthTexture = bool(depthTexture)
        if self._depthTexture:
            self.depthTextureKwargs = kwargs.copy()
            self.depthTextureKwargs['internalFormat'] = depthType
            self.depthTextureKwargs['format'] = GL.GL_DEPTH_COMPONENT
            self.depthTextureKwargs['dataType'] = GL.GL_FLOAT

        self.width = None
        self.height = None
        self.resize(width, height)

    def resize(self, width, height):
        '''Resize to new width and height.

        For safety, this actually destroys and recreates the framebuffer and
        all attachments.
        '''
        #Check to make sure we actually changed something.
        if self.width == width and self.height == height: return
        # print(width, height)

        self.delete()

        self.width = width
        self.height = height

        self.id = GL.glGenFramebuffers(1)
        self.bind()

        self.texture = Texture((self.width, self.height), **self.textureKwargs)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                               self.texture.target, self.texture.id, 0)

        if self.depth:
            if self._depthTexture:
                self.depthTexture = Texture((self.width, self.height),
                    **self.depthTextureKwargs)
                GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_ATTACHMENT, self.depthTexture.target,
                    self.depthTexture.id, 0)
            else:
                self.depthId = GL.glGenRenderbuffers(1)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depthId)
                GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, self.depthType,
                    self.width, self.height)
                GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, 0)
                GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER,
                    GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depthId)

        status = GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER)
        if status != GL.GL_FRAMEBUFFER_COMPLETE:
            errs = dict((int(globals()[item]), item)
                        for item in globals() if 'GL_FRAMEBUFFER_' in item)
            raise RuntimeError("Framebuffer status: %s" % errs.get(status, status))

        self.aspect = self.width / self.height

    def bind(self):
        '''Bind the framebuffer.'''
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.id)

    def delete(self):
        '''Delete the framebuffer and child objects.'''
        if hasattr(self, 'depth_id'):
            GL.glDeleteRenderbuffers(1, [self.depthId])
        if hasattr(self, 'texture'):
            self.texture.delete()
        if hasattr(self, 'depthTexture'):
            self.depthTexture.delete()
        if hasattr(self, 'id'):
            GL.glDeleteFramebuffers(1, [self.id])

    def __array__(self):
        return self.texture.__array__()


class TextRenderer:
    # From shader:
    # layout (location = 0) in vec3 anchor;
    # layout (location = 1) in vec4 glyph; // x/y = container, z/w = offset relative to bottom-left corner of container
    # layout (location = 2) in vec4 atlas; // x/y = x0/y0, z/w = x1/y1
    # layout (location = 3) in vec2 padding;
    # layout (location = 4) in vec3 baseline;
    # layout (location = 5) in vec3 up;
    # layout (location = 6) in uint flags;
    vertexType = np.dtype(dict(
        names =   [  'anchor',    'glyph',    'atlas',  'padding', 'baseline',       'up',  'flags', '_glyphRender'],
        #                 0:3,        3:7,       7:11,      11:13,      13:16,      16:19,    19:20,           3:11
        formats = ['3float32', '4float32', '4float32', '2float32', '3float32', '3float32', 'uint32',     '8float32'],
        offsets = [         0,         12,         28,         44,         52,         64,       76,             12],
        itemsize = 80
    ))

    def __init__(self, basename):
        '''Create a text renderer object.

        Parameters
        ----------
        basename : str
            A path/filename to the font information (.png and .font_info).
            The extension is ignored, and replaced with the required types.
        '''
        self.basename = os.path.splitext(basename)[0]
        with open(self.basename + '.font_info', 'rt') as f:
            info = json.load(f)

        geometry = np.frombuffer(base64.b64decode(info['geometry']),
            dtype='i4').reshape(-1, 8)

        self.atlas = np.full((geometry[:, 0].max()+1, 7),
            -1, dtype='f')
        self.atlas[geometry[:, 0]] = geometry[:, 1:8]
        self.atlas[geometry[:, 0], :3] *= 1./65536
        self.atlas[geometry[:, 0], 3:] += 0.5
        self.lineHeight = info['lineHeight']
        self.pixelsPerEm = info['pixelsPerEm']
        self.pixelRange = info.get('pixelRange', 2.0)
        M = self.atlas[ord('M')]
        self.capHeight = abs(M[6] - M[4]) / self.pixelsPerEm
        # print(self.capHeight)

        img = np.array(Image.open(self.basename + '.png'))
        self.texture = textureFromArray(img, target=GL.GL_TEXTURE_RECTANGLE)

    def write(self, va, anchor, text, flags=0, padding=1.0, lineSpacing=1.0,
            baseline=np.array([1, 0, 0], 'f'), up=np.array([0, 1, 0], 'f'),
            start=0):
        end = _writeText(text, start, self.capHeight,
            self.lineHeight*lineSpacing, va['_glyphRender'], self.atlas)
        va['flags'][start:end] = flags
        va['anchor'][start:end] = anchor
        va['baseline'][start:end] = baseline
        va['up'][start:end] = up
        va['padding'][start:end] = padding

        return end

    def delete(self):
        if hasattr(self, 'texture'):
            self.texture.delete()
            del self.texture
            del self.atlas

@numba.njit(cache=True)
def _writeText (s, start, cap_height, line_height, output, atlas):
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
