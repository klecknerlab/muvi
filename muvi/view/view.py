#!/usr/bin/python3
#
# Copyright 2021 Dustin Kleckner
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
import os
from PIL import Image
import warnings
from .params import PARAMS, SHADER_PATH, MAX_CHANNELS, COLORMAPS, \
    SUBSHADER_TYPES, SUBSHADER_SOURCE
# import time

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

_FS_QUAD = np.array([
    (-1, -1, 0),
    (+1, -1, 0),
    (+1, +1, 0),
    (-1, +1, 0),
], dtype='f')
_FS_QUAD_FACES = np.arange(4, dtype='u4')

#--------------------
# The main view class
#--------------------
class View:
    '''An object used to render views of 3D volumes.  This class does not take
    care of window management, just the low-level OpenGL calls and viewport
    and rendering managment.

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
    width : int (default: 1000), the width of the view
    height : int (default: 1000), the height of the view
    os_width : int (default: 1920), the width of the off-screen view (used for screenshots)
    os_height : int (default: 1080), the height of the off-screen view

    Note that X0, X1, and center are in physical units, i.e. the units used
    by Lx, Ly, and Lz in the volume.
    '''

    view_defaults = {k:v.default for k, v in PARAMS.items() if v.vcat=='view'}
    uniform_defaults = {k:v.default for k, v in PARAMS.items() if v.vcat=='uniform'}
    shader_defaults = {k:v.default for k, v in PARAMS.items() if v.vcat=='shader'}

    hidden_uniform_defaults = {
        'vol_texture': 0,
        'back_buffer_texture': 1,
        'colormap1_texture': 2,
        'colormap2_texture': 3,
        'colormap3_texture': 4,
        'vol_size': np.ones(3, dtype='f')*256,
        'vol_L': np.ones(3, dtype='f')*256,
        'grad_step': 1.0,
        # 'gamma_correct': 1/2.2,
    }

    def __init__(self, volume=None, width=1000, height=1000, os_width=1920, os_height=1080, params={}, source_dir=None):

        self.width = width
        self.height = height
        self._old_width = -1
        self._old_height = -1
        self.os_width = os_width
        self.os_height = os_height
        self._old_os_width = -1
        self._old_os_height = -1

        self.buttons_pressed = 0

        self.params = self.view_defaults.copy()
        self.params.update(self.shader_defaults)
        self.params.update(self.uniform_defaults)

        self.hidden_uniforms = self.hidden_uniform_defaults.copy()

        if volume is not None:
            self.attach_volume(volume)
        else:
            self.volume = None

        self.reload_shaders()


    def init_gl(self):
        '''Initiliaze OpenGL components.

        This will be automatically called on the first draw statement, if not
        done manually.
        '''

        if getattr(self, '_init_gl', False):
            return True

        self.texture_to_color_shader = ShaderProgram(fragment_shader='''
            void main() {
                gl_FragColor = vec4(gl_TexCoord[0].xyz, 1.0);
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

        for n in range(1, MAX_CHANNELS + 1):
            self.update_colormap(n)

        self.update_shader()

        self._init_gl = True

        if getattr(self, 'volume', None):
            self.attach_volume(self.volume)


    def reload_shaders(self):
        '''Load shaders from the shader directory.

        Normally not called by the user, but can be used to udpate sources,
        if they are being edited while a program is being run.

        If you are doing this, you should also call the `refresh_shaders`
        function first.  (Note: this is *not* a method of the view class, but
        is in the view module.)
        '''
        with open(os.path.join(SHADER_PATH, 'volume_shader.glsl')) as f:
            self.source_template = f.read()

        self.cached_shaders = {}


    def update_shader(self, distortion_model=None):
        '''Recompile the shader based on the current view options.
        '''

        # See if this shader has already been compiled by making a key
        key = hash(tuple(self.params[key] for key in self.shader_defaults.keys()) + (distortion_model, ))

        # for k in self.shader_defaults.keys():
        #     print(f'{k:20s}: {repr(self.params[k])}')

        uniforms = self.get_uniforms(include_hidden=True)

        if key in self.cached_shaders:
            self.current_volume_shader = self.cached_shaders[key]
            self.current_volume_shader.bind()
            self.current_volume_shader.set_uniforms(**uniforms)

        # If not, lets build a new one.
        else:
            if distortion_model is None:
                distortion_model = '''
                    vec3 distortion_map(in vec3 U) {
                        float exy = 0.25 * (perspective_xfact * (1.0 - 2.0 * U.x) + perspective_yfact * (1.0 - 2.0 * U.y));
                        float ez = 0.25 * (perspective_zfact * (1.0 - 2.0 * U.z));
                        return vec3((U.x + ez) / (1.0 + 2.0*ez), (U.y + ez) / (1.0 + 2.0*ez), (U.z + exy) / (1.0 + 2.0*exy));
                    }

                    mat3 distortion_map_gradient(in vec3 X) {
                        mat3 map_grad;
                        map_grad[0] = vec3(1.0, 0.0, 0.0);
                        map_grad[1] = vec3(0.0, 1.0, 0.0);
                        map_grad[2] = vec3(0.0, 0.0, 1.0);

                        return map_grad;
                    }
                '''

            code = [distortion_model]

            if self.params['gamma2']:
                code.append('#define GAMMA2_ADJUST 1')

            for n in range(1, MAX_CHANNELS + 1):
                if self.params[f'cloud{n}_active']:
                    code.append(f'#define CLOUD{n}_ACTIVE 1')
                if self.params[f'iso{n}_active']:
                    code.append(f'#define ISOSURFACE{n} 1')

            # Find the cloud_color, iso_level, and iso_color sources
            for subshader in SUBSHADER_TYPES:
                # sources = getattr(self, subshader + '_source')
                sources = SUBSHADER_SOURCE[subshader]
                name = self.params[subshader]

                if name not in sources:
                    raise ValueError("Unknown %s shader '%s'" % (subshader, name))

                code.append(sources[name])

            code = self.source_template.replace('<<VOL INIT>>', '\n'.join(code)).replace('<<COLOR_REMAP>>', self.params['color_remap'])

            for n in range(1, MAX_CHANNELS + 1):
                code = code.replace(f'<<ISO{n}_COLOR>>',
                    f'vec4(iso_color.r, iso_color.g, iso_color.b, iso{n}_opacity)')

            # print(code)
            # print(uniforms)
            # for k, v in uniforms.items():
            #     print(f'{k:20s}: {repr(v)}')

            self.current_volume_shader = ShaderProgram(fragment_shader=code, uniforms=uniforms)
            self.cached_shaders[key] = self.current_volume_shader

            self.current_volume_shader.bind()
            self.current_volume_shader.set_uniforms(**uniforms)


    def units_per_pixel(self):
        '''Returns the viewport scale in image units per pixel.'''
        return 2.0/self.params['scale']


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
            # self.autoscale()
            self.params['center'] -= self.units_per_pixel() * (self.params['R'][:, 0] * dx - self.params['R'][:, 1] * dy)

        # display_rot(y = r_xy[0], x = r_xy[1], z = -r_z)


    def get_options(self, var):
        if var in PARAMS and hasattr(PARAMS[var], 'options'):
            return PARAMS[var].options
        # if var == 'cloud_color':
        #     return self.cloud_color_names
        # elif var == 'colormap':
        #     return _colormap_names
        # elif var == 'color_remap':
        #     return _color_remaps
        else:
            raise ValueError("parameter '%s' does not exist or does not have a list of options" % var)


    def update_colormap(self, n=1):
        '''Choose the display colormap.

        Parameters
        ----------
        name : str (default: "viridis")
            The "short name" of the colormap (corresponds to a key in the
            `COLORMAPS` global dictionary.

        Keywords
        --------
        n : int (default: 1)
            The channel index we are selecting the color map for
            (n = 1-MAX_CHANNELS)
        '''

        if not hasattr(self, 'colormap_textures'):
            self.colormap_textures = [
                Texture(size = (256, ), format=GL_RGB, wrap=GL_CLAMP_TO_EDGE,
                         internal_format=GL_SRGB)
                for i in range(MAX_CHANNELS)
            ]

        glActiveTexture(GL_TEXTURE2 + (n-1))

        name = self.params['colormap' + str(n)]
        if name not in COLORMAPS:
            raise ValueError("unknown colormap '%s'" % name)
        self.colormap_textures[n-1].replace(COLORMAPS[name].data)

        glActiveTexture(GL_TEXTURE0)


    def attach_volume(self, volume):
        '''Attach a VolumetricMovie to the view.'''
        self.volume = volume

        if getattr(self, '_init_gl', False):
            if hasattr(self, 'volume_texture'):
                self.volume_texture.delete()

            vol = self.volume[0]
            if vol.ndim == 3: vol = vol[..., np.newaxis]

            glActiveTexture(GL_TEXTURE0)

            self.volume_texture = texture_from_array(vol, wrap=GL_CLAMP_TO_EDGE)

        gamma = self.volume.info.get('gamma', 1.0)
        self.params['gamma2'] = gamma > 1.95 and gamma < 2.05

        self.reset_view()


    def reset_view(self):
        if hasattr(self, 'volume'):
            vs = np.array(self.volume.info.get_list('Nx', 'Ny', 'Nz'), dtype='f')
            L = np.array(self.volume.info.get_list('Lx', 'Ly', 'Lz'), dtype='f')
        else:
            vs = np.ones(3, dtype='f')
            L = 100 * vs

        self.update_params(vol_size=vs, vol_L=L, center=0.5*L,
                           X1=L, X0=np.zeros(3, dtype='f'),
                           scale=float(2.0 / mag(L)))


    def frame(self, frame):
        '''Change the displayed frame.

        Parameters
        ----------
        frame : int
            The new frame number
        '''

        if self.volume is not None:
            glActiveTexture(GL_TEXTURE0)

            vol = self.volume[frame % len(self.volume)]
            if vol.ndim == 3: vol = vol[..., np.newaxis]
            self.volume_texture.replace(vol)


    def get_uniforms(self, include_hidden=False):
        '''Get a dictionary of the uniforms used by the volume shader.

        Keywords
        --------
        include_hidden : bool (default: False)
            If True, also includes hidden uniforms.
        '''

        d = self.hidden_uniforms.copy() if include_hidden else {}
        for k in self.uniform_defaults.keys():
            d[k] = self.params[k]

        return d


    def update_params(self, **kwargs):
        '''Updates the view parameters, performing all necessary updates
        downstream.

        Note: this is the prefered method to update the `params` dictionary,
        as updating it directly will not trigger all the dependent operations
        to be performed.
        '''

        if 'frame' in kwargs:
            self.params['frame'] = kwargs.pop('frame')
            self.frame(self.params['frame'])

        for n in range(1, MAX_CHANNELS+1):
            name = f'colormap{n}'
            if name in kwargs:
                self.params[name] = kwargs.pop(name)
                self.update_colormap(n)

        for k in ("os_width", "os_height", "width", "height"):
            if k in kwargs:
                setattr(self, k, kwargs.pop(k))

        update_shader = False
        update_uniforms = {}
        pop_keys = []

        for k, v in kwargs.items():
            if k in self.shader_defaults:
                update_shader = True
            elif k in self.uniform_defaults:
                update_uniforms[k] = v
            elif k in self.hidden_uniforms:
                update_uniforms[k] = v
                self.hidden_uniforms[k] = v
                # kwargs.pop(k)
                pop_keys.append(k)
            elif k not in self.view_defaults:
                # raise KeyError"unknown view parameter '%s'" % k()
                warnings.warn("unknown view parameter '%s', ignoring" % k)

        for k in pop_keys:
            kwargs.pop(k)
        self.params.update(kwargs)

        if update_shader:
            self.update_shader()

        if update_uniforms and hasattr(self, 'current_volume_shader'):
            self.current_volume_shader.bind()
            self.current_volume_shader.set_uniforms(**update_uniforms)

    def resize(self, width, height):
        '''Convenience command to update the width and height.'''
        self.width = width
        self.height = height


    # def fov_correction(self):
    #     '''Computes the half height of the viewport in OpenGL units.'''
    #     return np.tan(self.params['fov']*np.pi/360) if self.params['fov'] > 1E-6 else 1.


    def draw(self, z0=0.01, z1=10.0, offscreen=False, save_image=False, return_image=True):
        '''Draw the volume, creating all objects as required.  Has two
        parameters, but normally the defaults are fine.

        Parameters
        ----------
        z0 : float, the front clip plane (default: 0.01)
        z1 : float, the back clip plane (default: 10.0)
        offscreen : bool, if True, renders to the offscreen buffer instead of
           the onscreen one.  (Usually used for screenshots/movies.)
        '''

        # Framebuffers:
        #  default_fbo: the display
        #  back_buffer: stores texture coordinate for the volume render
        #      * RGB coordinate is physical coordinates of texture
        #      * Alpha coordinate is 1 if there is a
        #  display_buffer: the buffer we are rendering to (may be default_fbo...)

        # Draw process:
        # 1. If models present:
        #   Render the models to display_buffer
        # 2. If volume + models present:
        #   Render the models to the back_buffer (using texture_to_color_shader)
        # 3. If volume:
        #   Render the back of the texture cube to the back buffer
        #   Render the the front

        # start = time.time()

        if not getattr(self, '_init_gl', False):
            self.init_gl()

        # When called from QT or other window managers, we can't assume
        # that the default framebuffer is 0, so check!
        default_fbo = glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING)

        if offscreen:
            width = self.os_width
            height = self.os_height

            if self.os_width != self._old_os_width or self.os_height != self._old_os_height:
                # If the viewport has changed, resize things.
                if not hasattr(self, 'os_fbo_back'):
                    self.os_fbo_back = FrameBufferObject(width=self.os_width, height=self.os_height,
                                                 target=GL_TEXTURE_RECTANGLE, data_type=GL_FLOAT,
                                                 internal_format=GL_RGBA32F)

                    self.os_fbo_output = FrameBufferObject(width=self.os_width, height=self.os_height,
                                                 data_type=GL_UNSIGNED_BYTE, internal_format=GL_SRGB8_ALPHA8)
                else:
                    self.os_fbo_back.resize(self.os_width, self.os_height)
                    self.os_fbo_output.resize(self.os_width, self.os_height)

                self._old_os_width = self.os_width
                self._old_os_height = self.os_height

            back_buffer = self.os_fbo_back
            display_buffer_id = self.os_fbo_output.id

        else:
            width = self.width
            height = self.height

            if self.width != self._old_width or self.height != self._old_height:
                # If the viewport has changed, resize things.
                if not hasattr(self, 'fbo'):
                    self.fbo_back = FrameBufferObject(width=self.width, height=self.height,
                                                 target=GL_TEXTURE_RECTANGLE, data_type=GL_FLOAT,
                                                 internal_format=GL_RGBA32F)
                    self.fbo_output = FrameBufferObject(width=self.width, height=self.height,
                                                 data_type=GL_UNSIGNED_BYTE, internal_format=GL_SRGB8_ALPHA8)
                                                 # format=GL_SRGB8_ALPHA8)
                else:
                    self.fbo_back.resize(self.width, self.height)
                    self.fbo_output.resize(self.width, self.height)

                self._old_width = self.width
                self._old_height = self.height

            back_buffer = self.fbo_back
            display_buffer_id = self.fbo_output.id

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = width/height
        if self.params['fov'] > 1E-6:
            fov_correction = np.tan(self.params['fov']*np.pi/360)
            ymax = z0 * fov_correction
            glFrustum(-ymax*aspect, ymax*aspect, -ymax, ymax, z0, z1)
            camera_correction = 1

        else:
            fov_correction = 1
            glOrtho(-aspect, aspect, -1, 1, z0, z1)
            camera_correction = 1E3


        # Clear the display and set the view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


        R4 = np.eye(4, dtype='f')
        scale = self.params['scale'] * fov_correction
        R4[:3, :3] = self.params['R'] * scale

        # These get applied in the opposite order!
        glTranslatef(0, 0, -1) # Move it back to z = -1
        glMultMatrixf(R4) # Rotate about center
        glTranslatef(*(-self.params['center'])) # Move the center to the center

        # Draw the back buffer; this is used to find the back of the ray trace
        glBindFramebuffer(GL_FRAMEBUFFER, back_buffer.id)

        # Set up the viewport.  We do this on every draw call to be safe; no
        #   telling what the window manager is doing outside this code!
        glViewport(0, 0, width, height)

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
        glEnable(GL_FRAMEBUFFER_SRGB)
        glBindFramebuffer(GL_FRAMEBUFFER, display_buffer_id)
        glClearColor(*self.params['background_color'])

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # We can blend the output of the ray tracer with the background.
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        # We need to pass the information from the back buffer so we know which
        #   ray to trace.
        glEnable(GL_TEXTURE_2D)
        glActiveTexture(GL_TEXTURE1)
        back_buffer.texture.bind()

        # glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE);
        # glClampColor(GL_CLAMP_VERTEX_COLOR, GL_FALSE);
        # glClampColor(GL_CLAMP_FRAGMENT_COLOR, GL_FALSE)

        if self.volume is not None:
            self.current_volume_shader.bind()
            camera_loc = self.params['center'] + self.params['R'][:3, 2] / scale * camera_correction
            self.current_volume_shader.set_uniforms(camera_loc=camera_loc)

        else:
            # If we don't have a volume bound, we can draw something neat!
            self.interference_shader.bind()

        # Clear all matrices to draw a full screen quad
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glDisable(GL_CULL_FACE)
        self._fs_quad()

        glDisable(GL_FRAMEBUFFER_SRGB)

        if save_image or return_image:
            img = np.frombuffer(glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE), dtype='u1').reshape(height, width, -1)
            if save_image: Image.fromarray(img[::-1]).save(save_image)

        if not offscreen:
            w, h = self.width, self.height
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, default_fbo)
            glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST)

        if display_buffer_id != default_fbo:
            glBindFramebuffer(GL_FRAMEBUFFER, default_fbo)

        # print(time.time() - start)

        if save_image or return_image:
            return img


    #
    # def save_offscreen_image(self, fn):
    #     '''Save an image of the rendered volume to a file.  The size of the
    #     screenshot does not generally match the display, and is rendered
    #     offscreen.  The width and height are normally determined by the
    #     `os_width`, and `os_height` parameters.  If `width` or `height` are
    #     passed, they will update these parameters.
    #
    #     Parameters
    #     ----------
    #     width : int, default: 1000
    #     height: int, default: 1000
    #     offscreen : bool, if True, renders to the offscreen buffer instead of
    #        the onscreen one.  (Usually used for screenshots/movies.)
    #     '''
    #     Image.fromarray(self.offscreen_image).save(fn)


    def rot_x(self, a):
        '''Rotate view around x axis by a given angle (in radians).'''
        self.params['R'] = rot_x(a, self.params['R'])


    def rot_y(self, a):
        '''Rotate view around y axis by a given angle (in radians).'''
        self.params['R'] = rot_y(a, self.params['R'])


    def rot_z(self, a):
        '''Rotate view around z axis by a given angle (in radians).'''
        self.params['R'] = rot_z(a, self.params['R'])


    def _texture_cube(self):
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)

        ub = (_UNIT_BOX * (self.params['X1'] - self.params['X0']) + self.params['X0']).astype('f')

        glVertexPointer(3, GL_FLOAT, 0, ub)
        glTexCoordPointer(3, GL_FLOAT, 0, ub)
        glDrawElements(GL_QUADS, len(_BOX_FACES), GL_UNSIGNED_INT, list(_BOX_FACES))

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    def _fs_quad(self):
        glEnableClientState(GL_VERTEX_ARRAY)

        glVertexPointer(3, GL_FLOAT, 0, _FS_QUAD)
        glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT, list(_FS_QUAD_FACES))

        glDisableClientState(GL_VERTEX_ARRAY)
