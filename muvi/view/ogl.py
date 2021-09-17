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
import re
import ctypes


#--------------------------------------------------------
# Some basic vector operations -- pretty self explanatory
#--------------------------------------------------------


def mag(X):
    return np.sqrt(np.dot(X, X))

def mag1(X):
    return np.sqrt(np.dot(X, X))[..., np.newaxis]

def norm(X):
    return np.asarray(X) / mag(X)

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

        # Set up the vertex attribute information
        self._attrib = {}

        # Note that glGetActiveAttrib doesn't have a nice pyopengl binding,
        #  so we have to work a little harder ):
        bufSize = GL.glGetProgramiv(self.id, GL.GL_ACTIVE_ATTRIBUTE_MAX_LENGTH)
        length = GL.GLint()
        size = GL.GLint()
        t = GL.GLenum()
        name = (GL.GLchar * bufSize)()

        for i in range(GL.glGetProgramiv(self.id, GL.GL_ACTIVE_ATTRIBUTES)):
            GL.glGetActiveAttrib(self.id, i, bufSize, length, size, t, name)
            np_dtype, np_shape, gl_type, gl_var_type, uf = GL_VEC_TYPES[t.value]
            ns = name.value.decode('utf8')

            loc = GL.glGetAttribLocation(self.id, ns)
            items = int(size.value * np.prod(np_shape))

            self._attrib[ns] = (loc, np_dtype, items)

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


# class Buffer:
#     def __init__(self, dtype, shape, target=GL.GL_ARRAY_BUFFER,
#             usage=GL.GL_DYNAMIC_DRAW):
#         '''Initiliaze an openGL buffer, intended to be mapped to a numpy array
#
#         Parameters
#         ----------
#         dtype : numpy data type
#             The data type of the array
#         shape : integer, or tuple of integers
#             The shape of the array -- if 2 or more dimensional, only the
#             first index is used for indexing and assigment
#
#         Keywords
#         --------
#         target : GLenum (default: GL.GL_ARRAY_BUFFER)
#             The target parameters pass to the OpenGL buffer object.
#         usage : GLenum (default: GL.GL_DYNAMIC_DRAW)
#             The usage parameter passed to the OpenGL buffer object.
#
#         The resulting object can be written to like an array, using slice
#         notation.  (Indeed, this is the preferred method for altering it!)
#
#         Generally speaking, the first index is the number of elements, and the
#         subsequent indices is used for vector or matrix types.  If you set a
#         slice of the buffer with an array, the only thing that matters is that
#         the total number of elements matches!
#         '''
#
#         self.dtype = np.dtype(dtype)
#
#         if isinstance(shape, int):
#             self.shape = (shape, )
#         else:
#             self.shape = tuple(shape)
#
#         self.target = target
#         self.id = GL.glGenBuffers(1)
#         self.numElem = int(np.prod(self.shape[1:]))
#         self.bytesPerVertex = self.numElem * self.dtype.itemsize
#
#         GL.glBindBuffer(self.target, self.id)
#         nbytes = self.dtype.itemsize * int(np.prod(self.shape))
#         GL.glBufferData(self.target, nbytes, None, usage)
#
#     def __len__(self):
#         return self.shape[0]
#
#     def __setitem__(self, key, val):
#         if isinstance(key, int):
#             key = slice(key, key + 1)
#         if not isinstance(key, slice):
#             raise TypeError("buffer indices must be integers or slices, not tuple")
#
#         if key.step not in (None, 1):
#             raise ValueError("only step=1 slices are supported for setting the buffer")
#
#         val = np.asarray(val, dtype=self.dtype, order='C')
#         if val.size % self.numElem:
#             raise ValueError(f"The number of elements in this array ({val.size}) is not divisible by the number of elements in the buffer ({self.numElem})")
#         n = val.size // self.numElem
#
#         start, end = key.start, key.stop
#
#         if start is None:
#             if end is None:
#                 start = 0
#             else:
#                 start = end - n
#         elif (end is not None) and (end - start != n):
#             raise ValueError(f"Incompatible number of items in array -- can't set!")
#
#         # print(self.id)
#         GL.glBindBuffer(self.target, self.id)
#         GL.glBufferSubData(self.target, start*self.bytesPerVertex, val)
#
#     def _set(self, val, start=0):
#         '''Set the value of the buffer using an array -- no checking of sizes,
#         etc. is performed, so make sure you know what you are doing!'''
#
#         val = np.asarray(val, dtype=self.dtype, order='C')
#         GL.glBindBuffer(self.target, self.id)
#         GL.glBufferSubData(self.target, start*self.bytesPerVertex, val)
#
#     def bind(self):
#         GL.glBindBuffer(self.target, self.id)
#
#     def delete(self):
#         GL.glDeleteBuffers(1, [self.id])


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
            if name.startswith('__'):
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


# class VertexArray:
#     def __init__(self, spec, numVert, packed=False, usage=GL.GL_STATIC_DRAW):
#         '''An object to contain an OpenGL VertexArray.
#
#         Paramaters
#         ----------
#         spec : ShaderProgram object, or dictionary
#             If a ShaderProgram object, attributes are copied over from that
#             program, otherwise a dictionary of the form:
#                 {name1: (loc1, dtype1, num1), name2: (loc2, dtype2, num2), ...}
#         numVert : int
#             The number of vertices to store in the array
#
#         Keywords
#         --------
#         packed : bool (default: False)
#             If True, the array data is stored as a single buffer, and the
#             vertex array can be accessed directly like a Buffer object.
#             Requires all attributes to have the same data type!
#         usage : GLenum (default: GL.GL_DYNAMIC_DRAW)
#             The usage parameter passed to the OpenGL buffer object.
#
#         Examples
#         --------
#
#         >>> arr = VertexArray({'position': (0, 'f', 3), 'color': (0, 'f', 4)}, 20)
#         >>> arr['position'] = np.random.rand((10, 3))
#
#         Create an array, and set the first 10 vertex positions
#
#         >>> arr['position', -10:] = numpy.random.rand((10, 3))
#
#         Set the last ten vertex positions
#
#         >>> arr.drawArrays(GL.GL_TRIANGLES, 0, 9)
#
#         Draw 3 triangles in array order
#
#         >>> arr.attachElements(np.random.randint(0, 20, (15, 3)))
#         >>> arr.drawElements(GL.GL_TRIANGLES, 30)
#
#         Draws 10 triangles, with ordering determined by array.  (Note: we
#         created 15, but only drew 10!)
#
#         >>> arr2 = VertexArray({'position': (0, 'f', 3), 'color': (0, 'f', 4)}, 20, packed=True)
#         >>> arr2[:] = np.random.rand((15, 6))
#
#         Create a packed array, setting position *and* color of the first 15
#         elements.
#
#         '''
#
#         if hasattr(spec, "_attrib"):
#             spec = {k:v for k, v in spec._attrib.items()
#                         if not k.startswith("gl_")}
#
#         self.id = GL.glGenVertexArrays(1)
#         self.packed = bool(packed)
#         GL.glBindVertexArray(self.id)
#
#         if self.packed:
#             attrib = [(loc, name, dtype, items) for name, (loc, dtype, items)
#                 in spec.items()]
#             attrib.sort()
#             totalItems = 0
#             dtype = attrib[0][2]
#             self.loc = {}
#
#             for (loc, name, dt, items) in attrib:
#                 if dtype != dt:
#                     raise ValueError('For a packed array, all items must have same data type!')
#                 self.loc[name] = (loc, totalItems, items)
#                 totalItems += items
#
#             self.buffer = Buffer(dtype, (numVert, totalItems))
#             glType = npToGl(dtype)
#
#             for name, (loc, start, items) in self.loc.items():
#                 GL.glEnableVertexAttribArray(loc)
#                 GL.glVertexAttribPointer(loc, items, glType,  GL.GL_FALSE,
#                     self.buffer.bytesPerVertex,
#                     GL.GLvoidp(self.buffer.dtype.itemsize * start))
#
#         else:
#             self.buffers = {}
#
#             for name, (loc, dtype, items) in spec.items():
#                 buffer = Buffer(dtype, (numVert, items))
#                 self.buffers[name] = buffer
#                 GL.glEnableVertexAttribArray(loc)
#                 GL.glVertexAttribPointer(loc, items, npToGl(dtype),
#                     GL.GL_FALSE, 0, None)
#
#         GL.glBindVertexArray(0)
#
#     def drawArrays(self, mode, first, count):
#         '''Bind the Array and draw vertices using glDrawArrays.
#
#         See GL.glDrawArrays for more info.'''
#         GL.glBindVertexArray(self.id)
#         GL.glDrawArrays(mode, first, count)
#
#     def drawElements(self, mode, count=None):
#         if not hasattr(self, 'elements'):
#             raise ValueError("You must create elements using the 'attachElements' method before calling 'drawElements'")
#         GL.glBindVertexArray(self.id)
#         self.elements.bind()
#         if count is None:
#             count = len(self.elements)
#         GL.glDrawElements(mode, count, self.elementType, GL.GLvoidp(0))
#
#     def attachElements(self, elem, dtype='u4'):
#         if isinstance(elem, np.ndarray):
#             self.elements = Buffer(dtype, elem.size,
#                 target=GL.GL_ELEMENT_ARRAY_BUFFER)
#             self.elements[:] = elem
#         else:
#             self.elements = Buffer(dtype, elem,
#                 target=GL.GL_ELEMENT_ARRAY_BUFFER)
#         self.elementType = npToGl(dtype)
#
#     def __setitem__(self, key, val):
#         if self.packed:
#             self.buffer.__setitem__(key, val)
#         elif isinstance(key, str):
#             self.buffers[key][:] = val
#         elif isinstance(key, tuple) and len(key) == 2:
#             self.buffers[key[0]].__setitem__(key[1], val)
#         else:
#             raise ValueError('Key for unpacked array should be string or string, slice')
#
#     def delete(self):
#         if hasattr(self, 'id'):
#             GL.glDeleteVertexArrays(1, [self.id])
#             del self.id
#
#         if hasattr(self, 'buffers'):
#             for buffer in self.buffers.values():
#                 buffer.delete()
#
#         if hasattr(self, 'buffer'):
#             self.buffer.delete()
#
#         if hasattr(self, 'elements'):
#             self.elements.delete()


# class VertexArray:
#     def __init__(self, _parent, _usage=GL.GL_STATIC_DRAW, _packed=None,
#         _refreshable=True, **kwargs):
#         '''Initiliaze a VertexArray object
#
#         Parameters
#         ----------
#         _parent : ShaderProgram
#             The shader program this will draw to.  Used to identify name
#             attributes
#
#         Keywords
#         --------
#         _usage : GLenum (default: GL_STATIC_DRAW)
#             The usage type of the buffers
#         _packed : None or numpy array (default: None)
#             If specified, the data is assumed to close packed with each
#             attribute following the order indicated by the shader.  In this
#             case, the data type of the array *MUST* match that of the shader,
#             and all data must have the same type.
#         _refreshable : bool (default: False)
#             If True, keep a reference to the data array to allow the vertices
#             to be refreshed.  (In this case, _usage should most probably be
#             GL_DYNAMIC_DRAW or similar.)
#
#         If _packed is not specified, additional keywords are mapped to input
#         attributes in the shader.  All attributes must be specified, or an
#         error will be raised.
#         '''
#         self._attrib = {}
#         for k, v in _parent._attrib.items():
#             if k.startswith('gl_'):
#                 continue
#             self._attrib[k] = v
#
#         self.id = GL.glGenVertexArrays(1)
#         self.bind()
#         n = 1 if _packed is not None else len(self._attrib)
#         self.vb_ids = GL.glGenBuffers(n)
#         if n == 1: # Usually if n=1, it does not return a list... force it to!
#             self.vb_ids = (self.vb_ids,)
#         self._len = None
#
#         self.refreshable = bool(_refreshable)
#         self.usage = _usage
#
#         if _packed is not None:
#             if not isinstance(_packed, np.ndarray) or _packed.ndim != 2:
#                 raise AttributeError('_packed input must be 2D numpy array')
#
#             offset = 0
#             self._len, stride = _packed.shape
#             stride *= _packed.itemsize
#
#             # print(_packed.shape, _packed.nbytes)
#
#             GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vb_ids[0])
#             GL.glBufferData(GL.GL_ARRAY_BUFFER, _packed, _usage)
#
#             if self.refreshable:
#                 self.sources = (_packed,)
#
#             for i, (loc, items, gl_var_type, np_dtype) in \
#                     enumerate(sorted(self._attrib.values())):
#                 if np_dtype != _packed.dtype:
#                     raise AttributeError(f'_packed input must have same data type as shader input attributes ({_packed.dtype} != {np_dtype})')
#                 GL.glVertexAttribPointer(loc, items, gl_var_type, GL.GL_FALSE,
#                     stride, GL.GLvoidp(offset))
#                 GL.glEnableVertexAttribArray(loc)
#                 # print(i, loc, items, stride, offset)
#                 offset += items * _packed.itemsize
#
#         else:
#             if self._attrib.keys() != kwargs.keys():
#                 raise AttributeError(f'the attributes of this vertex array object must match shader specification!\n(shader has: {tuple(self._attrib.keys())})')
#
#             if self.refreshable:
#                 self.sources = []
#
#             for i, (name, val) in enumerate(kwargs.items()):
#                 loc, items, gl_var_type, np_dtype = self._attrib[name]
#                 val = np.asarray(val, dtype=np_dtype, order='C')
#                 val = val.reshape(-1, items)
#                 if self._len is None:
#                     self._len = len(val)
#                 if len(val) != self._len:
#                     raise AttributeError(f'each attribute must have the same number of values!')
#
#                 if self.refreshable:
#                     self.sources.append(val)
#                 GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vb_ids[i])
#                 GL.glBufferData(GL.GL_ARRAY_BUFFER, val, _usage)
#                 GL.glEnableVertexAttribArray(loc)
#                 GL.glVertexAttribPointer(loc, items, gl_var_type, GL.GL_FALSE, 0, GL.GLvoidp(0))
#
#         self.unbind()
#         GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
#
#     def refresh(self, n=None):
#         if not self.refreshable:
#             raise ValueError('this Vertex Array was created with _refreshable = False')
#         else:
#             # self.bind()
#             for i, source in enumerate(self.sources):
#                 GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vb_ids[i])
#                 if n is None:
#                     GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, source)
#                 else:
#                     GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, source[:n])
#
#     def bind(self):
#         GL.glBindVertexArray(self.id)
#         if hasattr(self, 'element_id'):
#             GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.element_id)
#
#     def unbind(self):
#         GL.glBindVertexArray(0)
#         if hasattr(self, 'element_id'):
#             GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
#
#     def attachElements(self, data, element_type=GL.GL_TRIANGLES):
#         self.element_id = GL.glGenBuffers(1)
#         self.element_type = element_type
#
#         data = np.asarray(data, dtype='u4', order='C')
#         self.n_elem = data.size
#
#         GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.element_id)
#         GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, data, GL.GL_STATIC_DRAW)
#         GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, 0)
#
#     def drawElements(self):
#         GL.glDrawElements(self.element_type, self.n_elem, GL.GL_UNSIGNED_INT, None)
#
#     def draw(self):
#         self.bind()
#         self.drawElements()
#         self.unbind()
#
#     def __enter__(self):
#         self.bind()
#         return self
#
#     def __exit__(self, type, value, traceback):
#         self.unbind()
#
#     def delete(self):
#         buffers = list(getattr(self, 'vb_ids', []))
#         if hasattr(self, 'element_id'):
#             buffers.append(self.element_id)
#
#         if buffers:
#             GL.glDeleteBuffers(len(buffers), buffers)
#
#         if hasattr(self, 'id'):
#             GL.glDeleteVertexArrays(1, [self.id])
#             del self.id
#
#     def __del__(self):
#         self.delete()


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
