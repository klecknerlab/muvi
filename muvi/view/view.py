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

'''
This file contains the openGL routines for drawing a volume, but does not
actually handle any of window managment.

The qtview module contains an example of a routine that uses these functions.
'''


# A previous attempt to code this module did not rely on pyopengl, but rather
# used the external module to supply the OpenGL functions -- this proved quite
# problematic!  It turns out you can directly embed pyopengl in a pyqt5 widget
# so long as _all_ of the calls are from pyopengl; mixing calls from the two
# set of opengl function produces errors.
#
# The pyqt opengl is particularly problematic for passing numpy data, which
# makes it useless here!

from .ooopengl import *
from string import Template
import re
import os

_module_path = os.path.split(os.path.abspath(__file__))[0]
def in_module_dir(fn):
    return os.path.join(_module_path, fn)

#----------------------------------------------------------
# Class for writing persepective corrected volume rendering
#----------------------------------------------------------

_VOLUME_GLSL_DEFS = '''
uniform sampler3D vol_texture;
uniform sampler2DRect vol_back_buffer;
uniform vec3 vol_size;
uniform vec3 vol_delta;
uniform float vol_grad_step;
uniform float vol_gamma_correct;

vec3 vol_gradient(in vec3 p);
vec3 vol_texture_space(in vec3 p);

vec3 vol_rgb_function(in vec3 p) {
    return texture3D(vol_texture, vol_texture_space(p)).$cc;
}

float vol_a_function(in vec3 p) {
    return texture3D(vol_texture, vol_texture_space(p)).$oc;
}

vec4 vol_rgba_function(in vec3 p) {
    return texture3D(vol_texture, vol_texture_space(p)).$cc$oc;
}

vec4 vol_output_correct(in vec4 color) {
    return pow(color, vec4(vol_gamma_correct, vol_gamma_correct, vol_gamma_correct, 1.0));
}
'''

_VOLUME_DEFAULT_UNIFORMS = {
    'vol_texture': 0,
    'vol_back_buffer': 1,
    'vol_size': np.ones(3, dtype='f')*256,
    'vol_delta': np.ones(3, dtype='f')/256,
    'vol_grad_step': 1.0,
    'vol_gamma_correct': 1/2.2,
}

class Value(object):
    def __init__(self, val, minval=None, maxval=None, step=1):
        self.val = val
        self.minval = minval
        self.maxval = maxval
        self.step = step

    def inc(self):
        self.val += self.step
        if self.minval is not None:
            self.val = max(self.minval, val)

    def dec(self):
        self.val -= self.step
        if self.minval is not None:
            self.val = min(self.maxval, val)


class LogValue(object):
    def __init__(self, val, minval=None, maxval=None, logbase=2, steps_per_base=2):
        self.val = val
        self.minval = minval
        self.maxval = maxval
        self.logbase = logbase
        self.step = logbase**(1/steps_per_base)

    def inc(self):
        self.val *= self.step
        if self.minval is not None:
            self.val = max(self.minval, val)

    def dec(self):
        self.val /= self.step
        if self.minval is not None:
            self.val = min(self.maxval, val)


_PERSPECTIVE_MODEL_CODE = {
'uncorrected': '''
vec3 vol_texture_space(in vec3 p) {
    return(p * vol_delta);
}

vec3 vol_gradient(in vec3 p) {
    vec3 dx = vol_grad_step * vec3(vol_delta.x, 0.0, 0.0);
    vec3 dy = vol_grad_step * vec3(0.0, vol_delta.y, 0.0);
    vec3 dz = vol_grad_step * vec3(0.0, 0.0, vol_delta.z);
    vec3 ts = vol_texture_space(p);

    return(vec3(
    texture3D(vol_texture, ts + dx).$oc - texture3D(vol_texture, ts - dx).$oc,
    texture3D(vol_texture, ts + dy).$oc - texture3D(vol_texture, ts - dy).$oc,
    texture3D(vol_texture, ts + dz).$oc - texture3D(vol_texture, ts - dz).$oc
    ));
}
''',
}

class VolumeShader(object):
    '''Creates a volume shader from an appropriate GLSL fragment shader.

    The Python code handles inserting the appropriate perspective correction
    functions, textures, and key variables to provide consistent implementation.

    Parameters
    ----------
    source : filename for external fragment shader code, or string with shader
        code.  If the input contains a line ending, it will assume it is code,
        otherwise it is treated as a filename.

    Note that no code will not get compiled until the "compile" method is
        called.
    '''

    def __init__(self, source):
        if '\n' in source:
            self.source = source
        else:
            with open(source) as f:
                self.source = f.read()

        self.uniforms = _VOLUME_DEFAULT_UNIFORMS.copy()

        def tup(*val): return val

        spec_dict = {"logfloat":LogValue, "float":Value, "int":Value,
                     "color":tup, "vector":tup, "norm":tup}
        for t, name, spec in \
            re.findall('\s*uniform\s+(\S+)\s+(\S+)\s*;\s*//\s*VIEW_VAR:\s*(.*)\s+', self.source, flags=re.M):
            # print(spec)
            val = eval(spec, spec_dict)
            # print(name, val)
            self.uniforms[name] = val

        self.cached_shaders = {}

    def compile(self, color_function='rrr', opacity_function='r',
                perspective_model="uncorrected", defines=""):
        '''Compile and return a shader with the options.

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
        '''
        key = (color_function, opacity_function, perspective_model, defines)

        if key not in self.cached_shaders:
            if perspective_model not in _PERSPECTIVE_MODEL_CODE:
                raise ValueError('Specified unknown perspective model "%s"\n(known options:[%s])' %
                    (perspective_model, ', '.join(map(repr, _PERSPECTIVE_MODEL_CODE.keys())) ))

            code = defines + '\n' + _VOLUME_GLSL_DEFS + _PERSPECTIVE_MODEL_CODE[perspective_model]
            code = self.source.replace('<<VOL INIT>>', code).replace('$oc', opacity_function).replace('$cc', color_function)

            self.cached_shaders[key] = ShaderProgram(fragment_shader=code, uniforms=self.uniforms)

        return self.cached_shaders[key]


#--------------------------------------------------------
# Constants for drawing boxes
#--------------------------------------------------------

_UNIT_BOX = np.array([(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype='f')

_BOX_FACES = np.array([
    0, 1, 3, 2, #-X
    4, 6, 7, 5, #+X
    0, 4, 5, 1, #-Y
    2, 3, 7, 6, #+Y
    0, 2, 6, 4, #-Z
    1, 5, 7, 3, #+Z
], dtype='u4')

_BOX_EDGES = [(i, j) for i in range(8) for j in range(8) if ((i < j) and sum((_UNIT_BOX[i] - _UNIT_BOX[j])**2) == 1.)]


#--------------------
# The main view class
#--------------------
class View(object):
    '''An object which represents and renders views of a 3D volumes.

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
    width : float (default: 100), the width of the view
    height : float (default: 100), the height of the view
    '''

    def __init__(self, volume=None, R=np.eye(3), center=None, fov=30,
                 X0=np.zeros(3), X1=None, width=100, height=100,
                 scale=None):

        self.R = normalize_basis(R)
        self.center = center
        self.fov = fov
        self.X0 = X0
        self.X1 = X1
        self.width = width
        self.height = height
        self.scale = scale
        self._oldwidth = -1
        self._oldheight = -1
        self.buttons_pressed = 0
        self.volume_shader = None
        self.render_uniforms = {}

        if volume is not None: self.attach_volume(volume)
        else: self.volume = None

        self.texture_to_color_shader = ShaderProgram(fragment_shader='''
            void main() {
                gl_FragColor = vec4(gl_TexCoord[0].xyz, gl_FragCoord.z);
            }
        ''')

        self.interference_shader = ShaderProgram(fragment_shader = '''
            #extension GL_ARB_texture_rectangle : enable

            uniform sampler2DRect back_buffer;
            uniform float scale;

            void main() {
                float x = scale*length(gl_TexCoord[0].xyz - texture2DRect(back_buffer, gl_FragCoord.st).rgb);
                vec3 int_color = sin(vec3(x*6.15, x*7.55, x*8.51));
                int_color = int_color * int_color;
                gl_FragColor = vec4(int_color, 1.0);
            }
        ''', uniforms=dict(back_buffer=1, scale=2.0))

        # self.volume_add = VolumeShader(in_module_dir('volume_add_shader.glsl'))
        # self.volume_add_shader = self.volume_add.compile('rrr', 'r')
        #
        # self.volume_iso = VolumeShader(in_module_dir('volume_iso_shader.glsl'))
        # self.volume_iso_shader = self.volume_iso.compile('rrr', 'r')
        # self.select_volume_shader(self.volume_iso_shader)

        self.volume_shader_template = VolumeShader(in_module_dir('volume_shader.glsl'))
        self.select_volume_shader()


    def units_per_pixel(self):
        '''Returns the viewport scale in image units per pixel.'''
        return 2.0/self.scale


    def mouse_move(self, x, y, dx, dy):
        '''Handles mouse move events, rotating the volume accordingly.

        Parameters
        ----------
        dx : the x motion since the last event in viewport pixels.
        dy : the y motion since the last event in viewport pixels.

        Note that the buttons_pressed property should also be directly
        updated by the window manager.
        '''

        h = self.height #Should be replaced by current size?
        dx /= h
        dy /= h
        x = (x - self.width / 2.) / h
        y = (y - self.height / 2.) / h

        if abs(x) < 1E-6: x = 1E-6
        if abs(y) < 1E-6: y = 1E-6

        if self.buttons_pressed & 1:
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

            r_hat = np.array([np.cos(phi), np.sin(phi)])
            phi_hat = np.array([-np.sin(phi), np.cos(phi)])

            dr = dot(r_hat, (dx, dy))
            dphi = dot(phi_hat, (dx, dy))

            if r > 0.33:
                dphi /= r
                r = 0.33

            r_xy = r_hat * dr + (1 - r) * dphi * phi_hat
            r_z  = r * dphi

            self.rot_y(3*r_xy[0])
            self.rot_x(3*r_xy[1])
            self.rot_z(-r_z)

        elif self.buttons_pressed & 2:
            self.autoscale()
            self.center += self.units_per_pixel() * (self.R[:, 0] * dx - self.R[:, 1] * dy)

        # display_rot(y = r_xy[0], x = r_xy[1], z = -r_z)


    def attach_volume(self, volume):
        '''Attach a VolumetricMovie to the view.'''
        self.volume = volume
        if hasattr(self, 'volume_texture'):
            self.volume_texture.delete()
        vol = self.volume[0]
        if vol.ndim == 3: vol = vol[..., np.newaxis]
        # print(vol.shape)
        self.volume_texture = texture_from_array(vol, wrap=GL_CLAMP_TO_EDGE)
        vs = np.array(vol.shape[-2::-1], dtype='f')
        self.update_uniforms(vol_size=vs, vol_delta=1./vs)


    def frame(self, frame):
        glActiveTexture(GL_TEXTURE0)
        vol = self.volume[frame % len(self.volume)]
        if vol.ndim == 3: vol = vol[..., np.newaxis]
        # print(vol.shape)
        self.volume_texture.replace(vol)


    def update_uniforms(self, **kwargs):
        '''Function to update uniforms associated with volume rendering shader.

        The variable names and values should be passed as keyword arguments.

        See ``volume_shadre.glsl`` for a list of valid parameters.
        '''
        self.render_uniforms.update(**kwargs)
        # for k, v in kwargs.items(): print(k, v)
        if hasattr(self, "current_volume_shader"):
            self.current_volume_shader.bind()
            self.current_volume_shader.set_uniforms(**kwargs)


    def update_view_settings(self, **kwargs):
        '''Updates the view settings, recompiling the shader if needed.

        Accepts keyword arguments which are passed either to ``update_uniforms``
        or ``select_volume_shader``, as needed.  This is a convenience function
        which treats all view variables the same to ease front end creation.
        '''
        svs = {}

        if 'frame' in kwargs:
            self.frame(kwargs.pop('frame'))

        for k in ("show_isosurface", "color_function", "opacity_function", "perspective_model", "show_grid"):
            if k in kwargs:
                svs[k] = kwargs.pop(k)

        if svs: self.select_volume_shader(**svs)
        if kwargs: self.update_uniforms(**kwargs)


    def select_volume_shader(self, show_isosurface=True, color_function='rrr',
        opacity_function='r', perspective_model='uncorrected', show_grid=False):
        '''Compiles the volume render shader with the desired options.

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
        '''
        if show_isosurface:
            defines = "#define VOL_SHOW_ISOSURFACE 1"
            if show_grid:
                defines = defines + "\n#define VOL_SHOW_GRID 1"
        else:
            defines = ""

        self.current_volume_shader = self.volume_shader_template.compile(
            color_function=color_function, opacity_function=opacity_function,
            perspective_model=perspective_model, defines=defines)
        # print(self.render_uniforms)
        self.current_volume_shader.bind()
        self.current_volume_shader.set_uniforms(**self.render_uniforms)


    def resize(self, width, height):
        '''Convenience command to update the width and height.'''
        self.width = width
        self.height = height


    def autoscale(self):
        '''Sets limits of volume and scale if not already defined.'''
        if self.X1 is None:
            self.X1 = np.array(self.volume.shape[2::-1]) if self.volume is not None else np.ones(3)
        if self.scale is None:
            self.scale = float(2.0/mag(self.X1-self.X0))
        if self.center is None:
            self.center = -0.5*(self.X0+self.X1)


    def fov_correction(self):
        '''Computes the half height of the viewport in OpenGL units.'''
        return np.tan(self.fov*np.pi/360) if self.fov > 0 else 1.


    def draw(self, z0=0.01, z1=10.0):
        '''Draw the volume, creating all objects as required.  Has two
        parameters, but normally the defaults are fine.

        Parameters
        ----------
        z0 : float, the front clip plane (default: 0.01)
        z1 : float, the back clip plane (default: 10.0)
        '''

        # When called from QT or other window managers, we can't assume
        # that the default framebuffer is 0, so check!
        default_fbo = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)

        if self.width != self._oldwidth or self.height != self._oldheight:
            #If the viewport has changed, resize things.
            if not hasattr(self, 'fbo'):
                self.fbo = FrameBufferObject(width=self.width, height=self.height,
                                             target=GL_TEXTURE_RECTANGLE, data_type=GL_FLOAT)
            else:
                self.fbo.resize(self.width, self.height)
            self._oldwidth = self.width
            self._oldheight = self.height

        # Need to define things like scale if not already done
        self.autoscale()

        # Set up the viewport.  We do this on every draw call to be safe; no
        #   telling what the window manager is doing outside this code!
        glViewport(0, 0, self.width, self.height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = self.width/self.height
        if self.fov > 0:
            ymax = z0 * np.tan(self.fov*np.pi/360)
            glFrustum(-ymax*aspect, ymax*aspect, -ymax, ymax, z0, z1)
        else:
            glOrtho(-aspect_ratio, aspect_ratio, -1, 1, z0, z1)

        # Clear the display and set the view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        R4 = np.eye(4, dtype='f')
        R4[:3, :3] = self.R * self.scale * self.fov_correction()

        # These get applied in the opposite order!
        glTranslatef(0, 0, -1) # Move it back to z = -1
        glMultMatrixf(R4) # Rotate about center
        glTranslatef(self.center[0], self.center[1], self.center[2]) # Move the center to the center



        # Draw the back buffer; this is used to find the back of the ray trace
        self.fbo.bind()
        # glDisable(GL_FRAMEBUFFER_SRGB)

        # Here the drawing just writes the value of the texture coordinate on
        #   the back surface.
        self.texture_to_color_shader.bind()

        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Only draw back faces!
        glEnable(GL_CULL_FACE)
        glCullFace(GL_FRONT)

        # Don't do anything fancy here.
        glDisable(GL_BLEND)

        # Draw the cube; handles clipping etc. automatically.
        self._texture_cube()



        # Draw the front buffer; this is where the ray tracing magic happens.
        glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)
        # glEnable(GL_FRAMEBUFFER_SRGB)
        # glClearColor(1.0, 1.0, 1.0, 1.0)
        glClearColor(0.0, 0.0, 0.0, 1.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # We can blend the output of the ray tracer with the background.
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        # We need to pass the information from the back buffer so we know which
        #   ray to trace.
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE1)
        self.fbo.texture.bind()

        if self.volume is not None:
            self.current_volume_shader.bind()
            # self.volume_iso_shader.bind()
        else:
            # If we don't have a volume bound, we can draw something neat!
            self.interference_shader.bind()

        # Draw just the front faces this time.
        glCullFace(GL_BACK)

        # Do it!
        self._texture_cube()


    def rot_x(self, a):
        '''Rotate view around x axis by a given angle (in radians).'''
        self.R = rot_x(a, self.R)


    def rot_y(self, a):
        '''Rotate view around y axis by a given angle (in radians).'''
        self.R = rot_y(a, self.R)


    def rot_z(self, a):
        '''Rotate view around z axis by a given angle (in radians).'''
        self.R = rot_z(a, self.R)


    def _texture_cube(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        ub = (_UNIT_BOX * (self.X1 - self.X0) + self.X0).astype('f')

        glVertexPointer(3, GL_FLOAT, 0, ub)
        glTexCoordPointer(3, GL_FLOAT, 0, ub)
        glDrawElements(GL_QUADS, len(_BOX_FACES), GL_UNSIGNED_INT, list(_BOX_FACES))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
