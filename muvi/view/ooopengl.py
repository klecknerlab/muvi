#!/usr/bin/python3
#
# Copyright 2018 Dustin Kleckner
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

# Reference: http://qt.apidoc.info/5.2.0/qtopengl/qtopengl-framebufferobject2-example.html


import numpy as np
from OpenGL.GL import *
from OpenGL.GL import shaders
import re

#--------------------------------------------------------
# Some basic vector operations -- pretty self explanatory
#--------------------------------------------------------


def mag(X):
    return np.sqrt(np.dot(X, X))

def norm(X):
    return np.asarray(X) / mag(X)

def dot(X, Y):
    return (np.asarray(X)*Y).sum(-1)[..., np.newaxis]

def normalize_basis(V):
    '''Converts basis to be right handed and orthonormal.

    Parameters
    ----------
    V : [3, 3] array

    Returns
    -------
    V : [3, 3] array'''
    V = np.array(V)
    V[0] = norm(V[0])
    V[1] = norm(V[1] - dot(V[1], V[0]) * V[0])
    V[2] = np.cross(V[0], V[1])
    return V

def rot_x(a, V=np.eye(3)):
    U = np.array(V) #This will make a copy even if it is already an array
    U[..., 1] = np.cos(a) * V[..., 1] - np.sin(a) * V[..., 2]
    U[..., 2] = np.cos(a) * V[..., 2] + np.sin(a) * V[..., 1]
    return U

def rot_y(a, V=np.eye(3)):
    U = np.array(V) #This will make a copy even if it is already an array
    U[..., 2] = np.cos(a) * V[..., 2] - np.sin(a) * V[..., 0]
    U[..., 0] = np.cos(a) * V[..., 0] + np.sin(a) * V[..., 2]
    return U

def rot_z(a, V=np.eye(3)):
    U = np.array(V) #This will make a copy even if it is already an array
    U[..., 0] = np.cos(a) * V[..., 0] - np.sin(a) * V[..., 1]
    U[..., 1] = np.cos(a) * V[..., 1] + np.sin(a) * V[..., 0]
    return U


#----------------------------------------------
# Object oriented versions of OpenGL primitives
#----------------------------------------------

class Texture(object):
    '''OpenGL Texture Object

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
    '''

    def __init__(self, size, format=GL_RGBA, data_type=GL_UNSIGNED_BYTE,
                 mag_filter=GL_LINEAR, min_filter=GL_LINEAR, wrap_s=None,
                 wrap_t=None, wrap_r=None, wrap=None, target=None, source=None,
                 internal_format=None):

        # print(size)
        if target is None:
            if len(size) == 2:
                target = GL_TEXTURE_2D
            elif len(size) == 3:
                target = GL_TEXTURE_3D
            else: raise ValueError('size should have 2 or 3 elements (found %s)' % len(size))

        self.target = target

        if wrap_s is None: wrap_s = wrap
        if wrap_t is None: wrap_t = wrap
        if wrap_r is None: wrap_r = wrap

        self.id = glGenTextures(1)
        glActiveTexture(GL_TEXTURE0)
        self.bind()
        self.format = format
        self.data_type = data_type
        self.size = size
        if internal_format is None:
            self.internal_format = format
        else:
            self.internal_format = internal_format

        if len(size) == 2:
            glTexImage2D(self.target, 0, self.internal_format, size[0], size[1], 0, format,
                         data_type, source)
        elif len(size) == 3:
            glTexImage3D(self.target, 0, self.internal_format, size[0], size[1], size[2], 0,
                         format, data_type, source)
        else: raise ValueError('size should have 2 or 3 elements (found %s)' % len(size))

        if mag_filter is not None:
            glTexParameteri(self.target, GL_TEXTURE_MAG_FILTER, mag_filter)
        if min_filter is not None:
            glTexParameteri(self.target, GL_TEXTURE_MIN_FILTER, min_filter)
        if wrap_s is not None:
            glTexParameteri(self.target, GL_TEXTURE_WRAP_S, wrap_s)
        if wrap_t is not None:
            glTexParameteri(self.target, GL_TEXTURE_WRAP_T, wrap_t)
        if len(size)==3 and wrap_r is not None:
            glTexParameteri(self.target, GL_TEXTURE_WRAP_R, wrap_r)


    def bind(self):
        '''Bind the texture.'''
        glBindTexture(self.target, self.id)


    def delete(self):
        '''Delete the texture id.'''
        glDeleteTextures([self.id])


    def replace(self, arr):
        '''Replace the texture data with a new numpy array.

        Will also bind the texture to the current texture unit.'''

        glBindTexture(self.target, self.id)

        # print(self.size, arr.shape)
        if (self.size != arr.shape[:len(self.size)][::-1]):
            raise ValueError('new array must have same shape as texture!')

        if len(self.size) == 2:
            glTexImage2D(self.target, 0, self.internal_format, self.size[0], self.size[1], 0, self.format,
                         self.data_type, arr)

        elif len(self.size) == 3:
            glTexImage3D(self.target, 0, self.internal_format, self.size[0], self.size[1], self.size[2], 0,
                         self.format, self.data_type, arr)


def texture_from_array(arr, format=None, **kwargs):
    '''Create an OpenGL Texture Object from a numpy array.

    Parameters
    ----------
    arr : Numpy array.  Should have dimensions [y, x, d] or [z, y, x, d].  The
            last dimension is *always* the depth, which is used to determine
            the internal fromat (I, IA, RGB, or RGBA)
    format : directly specify the format.  Can be used to switch between (e.g.)
            IA and RG.  (default: determined from shape)

    Any extra keyword arguments are passed directly to Texture.
    '''

    dts = {
        np.dtype('int8'): GL_BYTE,
        np.dtype('uint8'): GL_UNSIGNED_BYTE,
        np.dtype('int16'): GL_SHORT,
        np.dtype('uint16'): GL_UNSIGNED_SHORT,
        np.dtype('int32'): GL_INT,
        np.dtype('uint32'): GL_UNSIGNED_INT,
        np.dtype('f'): GL_FLOAT
    }
    formats = {
        1: GL_RED,
        2: GL_RG,
        3: GL_RGB,
        4: GL_RGBA
    }

    if arr.dtype not in dts:
        raise ValueError('Data type of array should be one of: [%s] (found %s)', (','.join(dts.keys()), arr.dtype))

    if arr.ndim not in (3, 4):
        raise ValueError('Array should have dimensions [y, x, d] or [z, y, x, d] (found %s dimensions)' % (arr.ndim))

    if format is None:
        if arr.shape[-1] not in formats:
            raise ValueError('Last dimension should have length 1-4, indicating number of planes (found %s)', (arr.shape[-1]))
        format = formats[arr.shape[-1]]

    arr = np.ascontiguousarray(arr)


    return Texture(arr.shape[:-1][::-1], format=format, data_type=dts[arr.dtype],
                   source=arr, **kwargs)


class FrameBufferObject(object):
    '''OpenGL FrameBuffer object

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
    '''

    def __init__(self, width=100, height=100, depth=True,
                 depth_type=GL_DEPTH_COMPONENT24, **kwargs):
        self.texture_kwargs = kwargs
        self.depth = depth
        self.depth_type = depth_type
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

        self.id = glGenFramebuffers(1)
        self.bind()

        self.texture = Texture((self.width, self.height), **self.texture_kwargs)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               self.texture.target, self.texture.id, 0)

        if self.depth:
            self.depth_id = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.depth_id)
            glRenderbufferStorage(GL_RENDERBUFFER, self.depth_type, width, height)
            glBindRenderbuffer(GL_RENDERBUFFER, 0)
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                                      GL_RENDERBUFFER, self.depth_id)

        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            errs = dict((int(globals()[item]), item)
                        for item in globals() if 'GL_FRAMEBUFFER_' in item)
            raise RuntimeError("Framebuffer status: %s" % errs.get(status, status))


    def bind(self):
        '''Bind the framebuffer.'''
        glBindFramebuffer(GL_FRAMEBUFFER, self.id)

    def delete(self):
        '''Delete the framebuffer and child objects.'''
        if hasattr(self, 'depth_id'): glDeleteRenderbuffers(1, [self.depth_id])
        if hasattr(self, 'texture'): self.texture.delete()
        if hasattr(self, 'id'): glDeleteFramebuffers(1, [self.id])


_glUniforms = {
    (int,   1):glUniform1i,
    (int,   2):glUniform2i,
    (int,   3):glUniform3i,
    (int,   4):glUniform4i,
    (float, 1):glUniform1f,
    (float, 2):glUniform2f,
    (float, 3):glUniform3f,
    (float, 4):glUniform4f,
}

def raise_nice_compile_errors(err):
    # for arg in err.args:
    #     print(arg)
    #     print('-'*80)
    if err.args[0].startswith('Shader compile failure'):
        label = {
            GL_VERTEX_SHADER: "Vertex ",
            GL_FRAGMENT_SHADER: "Fragment ",
            GL_GEOMETRY_SHADER: "Geometry "
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
            # errstr = errstr + "%s%4d | %s\n" % ('!' if ln in marked_lns else ' ', ln, s)
            if ln in marked_lns: errstr = errstr + "%4d | %s\n" % (ln, s)


        errstr = errstr + "-"*79+"\n"

        err = RuntimeError(errstr)

    raise err

class ShaderProgram(object):
    '''OpenGL shader program object

    Parameters
    ----------
    vertex_shader : a string with the vertex shader (default: None).
    fragment_shader : a string with the fragment shader (default: None).
    geometry_shader : a string with the geometry shader (default: None).
    uniforms : if specified, set uniforms from a dictionary.
    verify : if True, verify code after setting uniforms (default: True)
    '''

    def __init__(self, vertex_shader=None, fragment_shader=None, geometry_shader=None, uniforms={}, verify=True):
        self.id = glCreateProgram()
        self.uloc = {}

        s = []

        try:
            if vertex_shader is not None:
                s.append(shaders.compileShader(vertex_shader, GL_VERTEX_SHADER))
            if fragment_shader is not None:
                s.append(shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
            if geometry_shader is not None:
                s.append(shaders.compileShader(geometry_shader, GL_GEOMETRY_SHADER))
        except RuntimeError as err:
            raise_nice_compile_errors(err)

        for ss in s:
            glAttachShader(self.id, ss)

        glLinkProgram(self.id)

        if uniforms:
            self.bind()
            self.set_uniforms(**uniforms)
            shaders.glUseProgram(0)

        if verify:
            glValidateProgram(self.id)

            validation = glGetProgramiv(self.id, GL_VALIDATE_STATUS)
            if validation == GL_FALSE:
                raise RuntimeError(
                    """Validation failure (%s): %s"""%(
                    validation,
                    shaders.glGetProgramInfoLog(self.id),
                ))

            link_status = glGetProgramiv(self.id, GL_LINK_STATUS)
            if link_status == GL_FALSE:
                raise RuntimeError(
                    """Link failure (%s): %s"""%(
                    link_status,
                    shaders.glGetProgramInfoLog(self.id),
                ))

        # Delete references to shader objects; they will be kept in memory
        # so long as the program still exists.
        for ss in s:
            glDeleteShader(ss)

    def set_uniforms(self, **kwargs):
        '''Set uniform values for shader as keyword arguments.

        Does not check that the shader is current -- be sure to bind it first!
        '''

        for key, val in kwargs.items():
            if key not in self.uloc:
                self.uloc[key] = glGetUniformLocation(self.id, key)

            #
            if hasattr(val, 'val'): val = val.val
            # print('%20s:%s' % (key, repr(val)))

            val = np.asarray(val)
            if not val.shape: val.shape = (1,)

            if len(val) > 4: raise ValueError('at most 4 values can be used for set_uniforms')
            if val.dtype in ('u1', 'u2', 'u4', 'u8', 'i1', 'i2', 'i4', 'i8'): dt = int
            elif val.dtype in ('f', 'd'): dt = float
            else: raise ValueError('values for set_uniforms should be ints or floats')

            # print(key, val)
            _glUniforms[dt, len(val)](self.uloc[key], *val)

    def bind(self):
        '''Use shader.'''
        glUseProgram(self.id)

    def delete(self):
        '''Delete shader object.'''
        glDeleteProgram(self.id)
