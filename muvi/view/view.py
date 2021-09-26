from .ogl import ShaderProgram, GL, VertexArray, FrameBuffer, useProgram, \
    norm, cross, mag, dot, dot1, textureFromArray, Texture, TextRenderer
from scipy.spatial.transform import Rotation
from ..mesh import load_mesh, Mesh
from .. import open_3D_movie, VolumetricMovie
import numpy as np
import sys, os
# from text_render import TextRenderer
from .params import PARAMS, MAX_CHANNELS, COLORMAPS, ASSET_DEFAULTS, \
    ASSET_PARAMS, ALL_ASSET_PARAMS
import re

SHADER_DIR = os.path.join(os.path.split(__file__)[0], 'shaders')


#--------------------------------------------------------
# Rounding to nearest 1/2/5, used for axes
#--------------------------------------------------------


_OTF = np.array([1, 2, 5], 'd')

def ceil125(x):
    round_up = 10**np.ceil(np.log10(x / _OTF)) * _OTF
    return round_up[np.argmin(round_up - x)]

def pow125(i):
    return 10**float(i//3) * _OTF[i%3]

def log125(x):
    l10 = int(np.floor(np.log10(x)))
    wl = np.where(x >= _OTF*10**l10)[0]
    return l10 * 3 + ((wl).max() if len(wl) else 0)


#----------------------------------------------------------------
# ViewAsset class: an object that can be displayed by the viewer
#----------------------------------------------------------------

CUBE_CORNERS = ((np.arange(8).reshape(-1, 1) // 2**np.arange(3)) % 2).astype('f')
CUBE_TRIANGLES = np.array([
    0, 1, 5, 0, 5, 4,
    1, 3, 7, 1, 7, 5,
    2, 7, 3, 2, 6, 7,
    0, 4, 2, 2, 4, 6,
    4, 5, 7, 4, 7, 6,
    0, 2, 3, 0, 3, 1
], dtype='u4')

def copyArray(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        return x.copy()
    else:
        return x

class ViewAsset:
    def __init__(self, data, id=0, parent=None):
        self.filename = None
        self.parent = parent
        self.id = id
        self.info = [f'Id: {self.id}']

        if isinstance(data, str):
            self.abspath = os.path.abspath(data)

            bfn, ext = os.path.splitext(data)
            ext = ext.lower()

            if ext in ('.vti', '.cine'):
                self.filename = data
                data = open_3D_movie(data)
            elif ext in ('.ply'):
                dir, bfn = os.path.split(bfn)
                m = re.match('(.*_)frame([0-9]+)', bfn)
                if m: # This is a sequence of polygon meshes!
                    self.filename = m.group(1) + "[frame]" + ext
                    data = {}
                    regex = re.compile(f'{m.group(1)}frame([0-9]+)' + ext)

                    for fn in os.listdir(dir):
                        m2 = regex.match(fn)
                        if m2:
                            data[int(m2.group(1))] = os.path.join(dir, fn)
                    self.abspath = os.path.abspath(data[min(data.keys())])
                else:
                    self.filename = data
                    data = load_mesh(data)
            else:
                raise ValueError('Supported file types are VTI, CINE, and PLY')


        if self.filename is None:
            self.info.append('Source: Object passed directly to viewer')
            self.filename = '-'
        else:
            dir, self.filename = os.path.split(self.filename)
            self.info.append(f'Directory: {os.path.abspath(dir)}')

        self.isVolume = False
        self.visible = False
        self.vertexArray = None
        self.validFrame = True
        self._frame = None
        self.uniforms = {}
        self.globalUniformNames = set()
        self.globalUniforms = {}

        if isinstance(data, Mesh):
            self.shader = 'mesh'
            self.vertexArray, self.X0, self.X1 = meshToVertexArray(data)
            self.frameRange = None
            self.label = f'Mesh: {self.filename}'

        elif isinstance(data, dict):
            self.shader = 'mesh'
            self.X0 = None
            self.X1 = None
            self.meshSeq = {}

            X0s = []
            X1s = []

            for key, mesh in data.items():
                if isinstance(mesh, str):
                    mesh = load_mesh(mesh)
                va, X0, X1 = meshToVertexArray(mesh)
                self.meshSeq[key] = va
                X0s.append(X0)
                X1s.append(X1)

            self.X0 = np.min(X0s, axis=0)
            self.X1 = np.max(X1s, axis=0)

            keys = self.meshSeq.keys()
            self.frameRange = (min(keys), max(keys))
            self.missingFrames = len(keys) != (self.frameRange[1] - self.frameRange[0] + 1)

            self.label = f'Mesh Sequence: {self.filename}'

        elif isinstance(data, VolumetricMovie):
            self.volume = data
            self.isVolume = True
            self.shader = 'volume'
            L = np.array(self.volume.info.get_list('Lx', 'Ly', 'Lz'), dtype='f')
            self.X0 = -0.5 * L
            self.X1 = 0.5 * L
            self.uniforms = dict(
                _vol_L = L,
                _vol_N = np.array(self.volume.info.get_list('Nx', 'Ny', 'Nz'), dtype='f'),
                distortion_correction_factor = self.volume.distortion.var.get('distortion_correction_factor', np.zeros(3, 'f'))
            )
            self.globalUniformNames.update(self.parent._shaderDep[self.shader])
            self.globalUniformNames.update(self.parent._rebuildDep.keys())

            self.frameRange = (0, len(self.volume) - 1)

            vol = self.volume[0]
            if vol.ndim == 3:
                vol = vol[..., np.newaxis]

            GL.glActiveTexture(GL.GL_TEXTURE1)
            self.volumeTexture = textureFromArray(vol, wrap=GL.GL_CLAMP_TO_EDGE)
            GL.glActiveTexture(GL.GL_TEXTURE0)


            points = np.empty(len(CUBE_CORNERS), _volVertType)
            points['position'] = CUBE_CORNERS
            self.vertexArray = VertexArray(points)
            self.vertexArray.attachElements(CUBE_TRIANGLES)

            self.label = f"Volumes: {self.filename}"

        else:
            raise ValueError('asset data should be a filename, Mesh/VolumetricMovie object, or dictionary of {frameNumber:Mesh}')

        with np.printoptions(precision=4) as opts:
            self.info.append(f'Lower Extent: {self.X0}')
            self.info.append(f'Upper Extent: {self.X1}')

        if self.frameRange is not None:
            self.info.append(f"Frames: {self.frameRange[0]}-{self.frameRange[1]}{' (missing)' if getattr(self, 'missingFrames', False) else ''}")

        for key, val in ASSET_DEFAULTS[self.shader].items():
            if key in self.uniforms or key in self.globalUniforms:
                continue
            self[key] = val

    def paramList(self):
        return ASSET_PARAMS[self.shader]

    def __setitem__(self, key, val):
        if key in self.globalUniformNames:
            if self.visible:
                self.parent[key] = val
            self.globalUniforms[key] = val
        elif key == 'frame':
            self.setFrame(val)
        elif key == 'visible':
            if val and (not self.visible) and hasattr(self, 'globalUniforms'):
                self.parent.update(self.globalUniforms)
            self.visible = val
        else:
            self.uniforms[key] = val

    def update(self, d):
        for k, v in d.items():
            self.__setitem__(k, v)

    def allParams(self, prefix=True, hidden=False):
        if prefix is True:
            prefix = f'#{self.id}_'

        d = {prefix+'visible':self.visible}
        d.update({
            prefix+k:copyArray(v)
            for k, v in self.uniforms.items()
            if (hidden or (not k.startswith('_')))
        })

        if hasattr(self, 'globalUniforms'):
            d.update({
                prefix+k:copyArray(v)
                for k, v in self.globalUniforms.items()
                if (hidden or (not k.startswith('_')))
            })

        return d

    def setFrame(self, frame):
        if frame == self._frame or self.frameRange is None:
            return
        if frame < self.frameRange[0] or frame > self.frameRange[1]:
            self.validFrame = False
            return
        self.validFrame = True

        if self.isVolume:
            if frame != getattr(self, '_frame', None):
                GL.glActiveTexture(GL.GL_TEXTURE1)
                self.volumeTexture.replace(self.volume[frame])

        elif hasattr(self, 'meshSeq'):
            self.vertexArray = self.meshSeq.get(frame, None)
            if self.vertexArray is None:
                self.validFrame = False

        self._frame = frame

    def draw(self):
        if self.validFrame:
            self.vertexArray.draw()

    def delete(self):
        # Explicitly clean up opengl storage.
        # Trusting the garbage collector to do this isn't a good idea, as it
        #   doesn't work well on app shutdown.
        if hasattr(self, 'volumeTexture'):
            self.volumeTexture.delete()
        if hasattr(self, 'meshSeq'):
            for item in self.meshSeq.values():
                item.delete()
            del self.vertexArray
        if hasattr(self, 'vertexArray'):
            self.vertexArray.delete()

_meshVertType = np.dtype([
    ('position', '3float32'),
    ('normal',   '3float32'),
    ('color',    '4float32')
])

_volVertType = np.dtype([
    ('position', '3float32'),
])

def meshToVertexArray(m):
    N = len(m.points)
    vert = np.empty(N, _meshVertType)

    if not hasattr(m, 'normals'):
        raise ValueError('Displayed meshes must include point normals!')

    vert['position'] = m.points
    vert['normal'] = m.normals

    if hasattr(m, 'colors'):
        N, channels = m.colors.shape
        m.ensure_linear_colors()
        vert['color'][:, :channels] = m.colors

        if channels == 3:
            vert['color'][:, 3] = 1.0
    else:
        vert['color'] = 1.0

    X0 = m.points.min(0)
    X1 = m.points.max(0)

    va = VertexArray(vert)
    va.attachElements(m.triangles.astype('u4'))

    return va, X0, X1


#--------------------------------------------------------
# View class: handles low level GL calls, UI agnostic!
#--------------------------------------------------------

class View:
    AXIS_MAX_TICKS = 1000
    AXIS_LABEL_MAX_CHARS = 5000

    _shaderDep = {
        "volume": {"surface_shade", "distortion_model", "cloud_shade",
            "color_remap", "vol_cloud1", "vol_cloud2", "vol_cloud3", "vol_iso1",
            "vol_iso2", "vol_iso3", "gamma2"},
        "mesh": {"surface_shade", "mesh_clip"},
        "text": {},
        "axis": {},
    }

    _rebuildDep = {
        "camera_pos":{"viewMatrix", "visibleAxis"},
        "look_at":{"viewMatrix"},
        "up":{"viewMatrix"},
        "fov":{"perspectiveMatrix", "visibleAxis"},
        "near":{"perspectiveMatrix"},
        "far":{"perspectiveMatrix"},
        "disp_X0":{"axisLine", "visibleAxis", "viewMatrix", "axisLabel"},
        "disp_X1":{"axisLine", "visibleAxis", "viewMatrix", "axisLabel"},
        "axis_major_tick_spacing":{"axisLine", "axisLabel"},
        "axis_minor_ticks":{"axisLine"},
        "axis_major_tick_length_ratio":{"axisLine"},
        "axis_minor_tick_length_ratio":{"axisLine"},
        "mesh_scale":{"meshModelMatrix"},
        "mesh_offset":{"meshModelMatrix"},
        "vol_colormap1":{"colormaps"},
        "vol_colormap2":{"colormaps"},
        "vol_colormap3":{"colormaps"},
        "axis_angle_exclude":{"visibleAxis"},
        "axis_single_label":{"visibleAxis"},
        "axis_label_angle":{"visibleAxis"},
        # "frame":{"frame"},
    }

    # Items that are not included get built in arbitrary order *AFTER* these
    _rebuildOrder = [
        "viewMatrix", "perspectiveMatrix", "axisLine"
    ]

    _allShaderDep = set.union(*_shaderDep.values())

    # Note: Camel case params are generated automatically, underscore versions
    #  correspond to external params.
    _defaults = dict(
        # surface_shade = "camera",
        # cloud_shade = "colormap",
        distortion_model = "simple",
        # mesh_perspective_correction = False,
        fontAtlas = 0,
        # model_matrix = np.eye(4, dtype='f'),
        # camera_pos = np.array([0, 300, 100], dtype='f'),
        # up = np.array([0, 1, 0], dtype='f'),
        # look_at = np.full(3, 50, dtype='f'),
        # fov = 45.0,
        near = 1.0,
        far = 1000.0,
        # disp_X0 = np.full(3, 0, dtype='f'),
        # disp_X1 = np.full(3, 100, dtype='f'),
        depthTexture = 0,
        volumeTextureId = 1,
        colormap1Texture = 2,
        colormap2Texture = 3,
        colormap3Texture = 4,
        # color_remap = "rgb",
        # mesh_clip = True,
        # mesh_scale = 20,
        # mesh_offset = np.full(3, 50, dtype='f'),
        # axis_line_color = np.ones(3, dtype='f'),
        # axis_line_width = 1.,
        display_scaling = 1.0,
        # background_color = np.zeros(3, dtype='f'),
        # axis_major_tick_spacing = 20,
        # axis_minor_ticks = 4,
        # axis_major_tick_length_ratio = 0.15, # Relative to major spacing
        # axis_minor_tick_length_ratio = 0.6, # Relative to major length
        # show_mesh = True,
        # show_volume = True,
        # show_axis = True,
        axis_max_ticks = 9,
    )

    _subShaders = {"surface_shade", "cloud_shade", "distortion_model"}

    def __init__(self, valueCallback=None, rangeCallback=None):
        self.buffers = {}
        self.viewMatrix = []
        # self._callbacks = {}
        # self._rangeUpdates = {}
        self._params = self._defaults.copy()
        for k, v in PARAMS.items():
            self._params[k] = v.default
        self._uniforms = {}
        # self._uniformUpdates = {}
        self._uniformNames = set()
        self._updateView = True
        self._uniformLastUpdate = 0
        self._needsRebuild = set.union(*self._rebuildDep.values())
        self.perspectiveMatrix = {}
        self.shaders = {}
        self.volFrames = 0
        self.visibleAssets = {
            "volume":set(),
            "mesh":set(),
        }
        self.assets = {}

        self._valueCallbacks = (valueCallback, ) if valueCallback is not None else ()
        self._rangeCallbacks = (rangeCallback, ) if rangeCallback is not None else ()

        self._cachedShaders = {}
        self._subShaderCode = {}
        self._colormap = [None] * MAX_CHANNELS
        self.frameRange = None

    #--------------------------------------------------------
    # Shader Compilation
    #--------------------------------------------------------

    def getSubShader(self, subshader):
        target = subshader + "_" + self[subshader]

        if target not in self._subShaderCode:
            with open(os.path.join(SHADER_DIR, target + '.glsl'), 'rt') as f:
                self._subShaderCode[target] = f.read() + '\n'

        return self._subShaderCode[target]

    def buildShader(self, target):
        if target not in self._shaderDep:
            raise ValueError(f'Shader target should be one of {tuple(self._shaderDep.keys())}')

        deps = self._shaderDep[target]

        # Make a unique key for this shader
        key = (target, ) + tuple(self._params[key] for key in deps)

        # See if it's already been compiled...
        if key in self._cachedShaders:
            shader = self._cachedShaders[key]
        # If not, let's compile!
        else:
            code = {}

            prefixCode = []

            for dep in deps:
                if dep in self._subShaders:
                    prefixCode.append(self.getSubShader(dep))
                elif dep != "color_remap" and self[dep]:
                    prefixCode.insert(0, f'#define {dep.upper()} 1')

            prefixCode = '\n'.join(prefixCode)

            for st in ('vertex', 'geometry', 'fragment'):
                fn = os.path.join(SHADER_DIR, f'{target}_{st}.glsl')
                if os.path.exists(fn):
                    with open(fn, 'rt') as f:
                        source = f.read()
                        source = source.replace('//<<INSERT_SHARED_FUNCS>>',
                            prefixCode)
                        source = source.replace('<<COLOR_REMAP>>',
                            self['color_remap'])
                        code[f'{st}Shader'] = source

            shader = ShaderProgram(**code)
            self._uniformNames.update(shader.keys())
            for k in shader.keys():
                if k in self._params:
                    self._uniforms[k] = self._params[k]

            self._cachedShaders[key] = shader

        self.shaders[target] = shader
        shader.update(self._uniforms, ignore=True)

        return shader

    def useShader(self, target):
        if target is None:
            useProgram(0)
        else:
            shader = self.shaders.get(target, None)
            if shader is None:
                shader = self.buildShader(target)
            shader.bind()
            return shader

    #--------------------------------------------------------
    # UI Interaction methods
    #--------------------------------------------------------

    _assetRe = re.compile('\#([0-9]+)_(.*)')

    def __setitem__(self, key, val, callback=False):
        self._params[key] = val

        if key.startswith('#'):
            m = self._assetRe.match(key)
            if not m:
                raise ValueError('Keys starting with # refer to assets, and should have the form "#[number]_[parameter]')
            else:
                id = int(m.group(1))
                asset = self.assets[id]
                assetKey = m.group(2)
                asset[assetKey] = val

                if assetKey == "visible":
                    val = bool(val)
                    if val:
                        self.visibleAssets[asset.shader].add(id)
                        self.resetRange()
                    else:
                        if id in self.visibleAssets[asset.shader]:
                            self.visibleAssets[asset.shader].remove(id)
                        self.resetRange()

        else:
            if key in self._uniformNames:
                # self._uniformUpdates[key] = val
                self._uniforms[key] = val
                for shader in self.shaders.values():
                    if shader is not None:
                        shader.__setitem__(key, val, ignore=True)
            if key in self._rebuildDep:
                self._needsRebuild.update(self._rebuildDep[key])
            if key in self._allShaderDep:
                for shader, dep in self._shaderDep.items():
                    if key in dep:
                        self.shaders[shader] = None
        if callback:
            for func in self._valueCallbacks:
                func(key, val)


    def update(self, d, callback=False):
        for k, v in d.items():
            self.__setitem__(k, v, callback)

    def __getitem__(self, k):
        return self._params[k]

    def updateRange(self, name, minVal, maxVal):
        for func in self._rangeCallbacks:
            func(name, minVal, maxVal)

    def mouseMove(self, x, y, dx, dy, buttonsPressed):
        '''Handles mouse move events, rotating the volume accordingly.

        Parameters
        ----------
        x, y:
            The x/y coordinate of the mouse, scaled to the height of the
            viewport
        dx, dy : int
            The x/y motion since the last event in viewport pixels, scaled to
            the height of the viewport
        buttonsPressed : int
            A bitmap of buttons pressed (button1 = 1, button2 = 2, button3 = 4,
            button(1+3) = 5...)
        '''

        if not buttonsPressed:
            return

        pb = self.buffers[0] # The primary draw buffer
        h = pb.height
        w = pb.width
        dx /= h
        dy /= h
        x = (x - w/2)/h
        y = (y - h/2)/h

        if abs(x) < 1E-6: x = 1E-6
        if abs(y) < 1E-6: y = 1E-6

        F = norm(self['look_at'] - self['camera_pos'])
        R = norm(cross(F, self['up']))
        U = cross(F, R)

        if buttonsPressed & 1:
            r = np.sqrt(x**2 + y**2)
            phi = np.arctan2(y, x)

            r_hat = np.array([np.cos(phi), np.sin(phi)])
            phi_hat = np.array([-np.sin(phi), np.cos(phi)])

            dr = dot(r_hat, (dx, dy))
            dphi = dot(phi_hat, (dx, dy))

            if r > 0.33:
                dphi /= r
                r = 0.33

            r_xy = 3 * (r_hat * dr + np.clip(0.5 - r, 0, 0.5) * dphi * phi_hat)
            r_z  = 3 * r * dphi

            self.rotateCamera((-R*r_xy[1] + U*r_xy[0]) - F * r_z)

        elif buttonsPressed & 2:
            self.moveCamera(-self.viewportHeight() * (R * dx + U * dy))

    def allParams(self, hidden=False):
        d = {k:copyArray(v) for k, v in self._params.items()
                if k in PARAMS and k not in ALL_ASSET_PARAMS
                and (hidden or not k.startswith('_'))}

        for asset in self.assets.values():
            d.update(asset.allParams())

        return d

    def assetSpec(self):
        return {id:getattr(asset, 'abspath', None) for id, asset in self.assets.items()}

    #--------------------------------------------------------
    # Adding/removing Data
    #--------------------------------------------------------

    def openData(self, data, id=None):
        if self.assets:
            if id is None:
                id = max(self.assets.keys()) + 1
            elif id in self.assets:
                self.removeAsset(id)
        else:
            if id is None:
                id = 0
        asset = ViewAsset(data, id, parent=self)
        self.assets[id] = asset
        # self[f'#{id}_visible'] = True # Will automatically trigger resetRange
        # self.resetView()
        return asset

    def openAssets(self, assets):
        newIds = {}

        for id, fn in assets.items():
            if isinstance(id, (str, bytes)):
                id = int(id)

            for id2, asset in self.assets.items():
                if getattr(asset, "abspath", False) == fn:
                    # This asset already exists!  Return id instead of asser
                    #  object to indicate we didn't load something new!
                    newIds[id] = asset.id
                    break
            else:
                if id in self.assets:
                    # We don't want to replace an existing asset
                    newIds[id] = self.openData(fn)
                else:
                    newIds[id] = self.openData(fn, id)


        return newIds

    def removeAsset(self, id):
        asset = self.assets[id]
        self[f'#{asset.id}_visible'] = False
        asset.delete()
        del self.assets[id]

    def resetRange(self):
        X0 = []
        X1 = []
        frameRange = []
        frames = 0

        for asset in self.assets.values():
            if asset.visible:
                X0.append(asset.X0)
                X1.append(asset.X1)
                if asset.frameRange is not None:
                    frameRange.append(asset.frameRange)

        if not len(X0):
            return

        X0 = np.min(X0, axis=0)
        X1 = np.max(X1, axis=0)

        if frameRange:
            frameRange = np.array(frameRange)
            frameRange = (frameRange[:, 0].min(), frameRange[:, 1].max())
        else:
            frameRange = None

        self.frameRange = frameRange

        D = max(X1 - X0) * np.ones(3, 'f')

        # Build the tick spacing, which is the nearest 125 increment that
        #  has less than the specified number of (major) ticks
        majorTick = ceil125(D / self['axis_max_ticks'])
        # print(X0, X1, D, majorTick)
        lc = log125(majorTick)
        # Minor ticks: 5 if the first digit is 1 or 5, or 4 if the first digit is 2
        #  ... or just go down two "powers" of the 125 sequence
        minorTicks = int(np.round(pow125(lc) / pow125(lc-2)))
        minorTick = majorTick / minorTicks

        co = self['camera_pos']
        la = self['look_at']

        # Allow user to set range past limits, to next *major* tick
        X0r = np.floor(X0 / majorTick) * majorTick
        X1r = np.ceil(X1 / majorTick) * majorTick
        self.updateRange('disp_X0', X0r, X1r)
        self.updateRange('disp_X1', X0r, X1r)
        self.updateRange('camera_pos', -10 * D, 10 * D)
        self.updateRange('look_at', X0r, X1r)
        if frameRange is not None:
            self.updateRange('frame', frameRange[0], frameRange[1])

        if self['_autoupdate_limits']:
            self.update({
                # Default limits expanded to nearest *minor* tick
                "disp_X0": np.floor(X0 / minorTick) * minorTick,
                "disp_X1": np.ceil(X1 / minorTick) * minorTick,
                "axis_major_tick_spacing": majorTick,
                "axis_minor_ticks": minorTicks,
            }, callback=True)

            self.resetView(co-la)

    #--------------------------------------------------------
    # Methods for manipulating the viewport
    #--------------------------------------------------------

    def rotateCamera(self, vec):
        R = Rotation.from_rotvec(vec).as_matrix()
        lookAt = self['look_at']
        self.update({
            'camera_pos': lookAt + (R * (self['camera_pos'] - lookAt)).sum(-1),
            'up': (R * norm(self['up'])).sum(-1)
        }, callback=True)

    def moveCamera(self, vec):
        self.update({
            'camera_pos': vec + self['camera_pos'],
            'look_at': vec + self['look_at']
        }, callback=True)

    def zoomCamera(self, zoom):
        lookAt = self['look_at']
        self.update({
            'camera_pos': lookAt + zoom * (self['camera_pos'] - lookAt)
        }, callback=True)

    def viewportHeight(self):
        '''Gives the height of the viewport in world units at the 'lookAt'
        location'''
        dist = mag(self['look_at'] - self['camera_pos'])
        fov = self['fov']

        if fov > 1E-3:
            return 2 * dist * np.tan(self['fov'] * np.pi / 360)
        else:
            return 2 * dist * np.tan(30 * np.pi / 360)

    #-----------------------------------------------------------
    # Methods for rebuilding automatically determined structures
    #-----------------------------------------------------------

    def build_viewMatrix(self):
        cp = self['camera_pos']

        forward = norm(self['look_at'] - cp)
        right = norm(cross(forward, self['up']))
        up = cross(right, forward)

        # print(cp, self['look_at'], self['up'], forward, right, up)

        viewMatrix = np.eye(4, dtype='f')
        viewMatrix[:3, 0] = right
        viewMatrix[:3, 1] = up
        viewMatrix[:3, 2] = -forward
        viewMatrix[3, 0] = -dot(right, cp)
        viewMatrix[3, 1] = -dot(up, cp)
        viewMatrix[3, 2] = +dot(forward, cp)

        X0 = self['disp_X0']
        X1 = self['disp_X1']

        center = np.ones(4, 'f')
        center[:3] = 0.5 * (X0 + X1)

        r = 0.5 * mag(X1 - X0)

        midDepth = -(center * viewMatrix[:, 2]).sum()
        self['near'] = max(midDepth - r, r/1000)
        self['far'] = midDepth + r
        # print(self['near'], self['far'])
        self['viewMatrix'] = viewMatrix

    def build_perspectiveMatrix(self):
        fov = self['fov']
        far = self['far']
        near = self['near']

        if fov > 1E-3:
            tanHD = np.tan(fov * np.pi / 360)

            for id, buffer in self.buffers.items():
                perspective = np.zeros((4, 4), dtype='f')
                perspective[0, 0] = 1.0 / (buffer.aspect * tanHD)
                perspective[1, 1] = 1.0 / tanHD
                perspective[2, 2] = -(far + near) / (far - near)
                perspective[2, 3] = -1.0
                perspective[3, 2] = -(2 * far * near) / (far - near)
                self.perspectiveMatrix[id] = perspective

        else:
            height = mag(self['camera_pos'] - self['look_at']) * np.tan(30 * np.pi / 360)

            for id, buffer in self.buffers.items():
                perspective = np.zeros((4, 4), dtype='f')
                perspective[0, 0] = 1.0 / (buffer.aspect * height)
                perspective[1, 1] = 1.0 / height
                perspective[2, 2] = -2.0 / (far - near)
                perspective[3, 2] = -(far + near) / (far - near)
                perspective[3, 3] = 1.0
                self.perspectiveMatrix[id] = perspective


    def build_visibleAxis(self):
        axisEnd = self.axisLineVert['position'][self.axisEdge[:12]]
        axisCenter = axisEnd.mean(1)
        X0 = self['disp_X0']
        X1 = self['disp_X1']
        middle = 0.5 * (X0 + X1)
        cameraPos = self['camera_pos']

        if self['fov'] > 1E-3:
            vaf = ((cameraPos > X0) * (1,  2,  4)).sum() + \
                  ((cameraPos < X1) * (8, 16, 32)).sum()
            cameraVec = norm(axisCenter - cameraPos)
        else:
            lookAt = self['look_at']
            vaf = ((cameraPos > lookAt) * (1,  2,  4)).sum() + \
                  ((cameraPos < lookAt) * (8, 16, 32)).sum()
            cameraVec = norm(lookAt - cameraPos)

        ev = (vaf & self.axisEdgeFaceMask)
        # Find edges where one face is visible and one is not
        ev = ((ev & (ev - 1)) == 0) * (ev != 0) # True iff 1 bit is nonzero

        dp = dot(cameraVec, norm(axisEnd[:, 1] - axisEnd[:, 0]))

        # Exclude axes which are oriented close to the camera vector
        ev *= np.arccos(abs(dp)) * 180/np.pi > self['axis_angle_exclude']

        if self['axis_single_label']:
            axisOffset = axisCenter - middle
            dir = norm((axisOffset[..., np.newaxis] * self['viewMatrix'][:3, 0:2]).sum(1))
            angle = self['axis_label_angle'] * np.pi / 180
            priority = dot(dir, (np.sin(angle), np.cos(angle)))
            priority[np.where(ev == 0)] = -10

            val = 0
            for axis in range(3):
                i0 = axis * 4
                best = np.argmax(priority[i0:i0+4]) + i0
                if priority[best] >= -1:
                    val += 1 << best
            self['visibleAxisLabels'] = val

        else:
            self['visibleAxisLabels'] = np.sum(1 << np.arange(12)[np.where(ev)])

        self['visibleAxisFaces'] = vaf

    def build_meshModelMatrix(self):
        scale = self['mesh_scale']
        matrix = np.diag((scale, scale, scale, 1)).astype('f')
        matrix[3, :3] = self['mesh_offset']
        self['meshModelMatrix'] = matrix

    def build_axisLabel(self):
        X0 = self['disp_X0']
        X1 = self['disp_X1']
        spacing = self['axis_major_tick_spacing']
        # Let's make sure the axes have actually changed...
        key = (tuple(X0), tuple(X1), spacing)
        if getattr(self, '_lastaxisLabel', None) == key:
            return
        else:
            self._lastaxisLabel = key

        i0 = np.ceil(X0 / spacing).astype('i4')
        i1 = np.floor(X1 / spacing).astype('i4')

        start = 0
        for axis in range(3):
            offset = X0.copy()
            offset[axis] = 0
            a2 = (axis + 1) % 3
            a3 = (axis + 2) % 3

            baseline = np.zeros(3, 'f')
            baseline[axis] = 1.0

            for n in range(4):
                up = np.zeros(3, 'f')
                if n%2:
                    offset[a2] = X1[a2]
                    up[a2] = +1
                else:
                    offset[a2] = X0[a2]
                    up[a2] = -1

                if n//2:
                    offset[a3] = X1[a3]
                    up[a3] = +1
                else:
                    offset[a3] = X0[a3]
                    up[a3] = -1

                visFlag = 1 << (4*axis + n + 8)

                for x in np.arange(i0[axis], i1[axis]+1) * spacing:
                    offset[axis] = x
                    start = self.textRender.write(self.axisLabel, offset,
                        f'{x}', flags=48 + visFlag, padding=0.5,
                        start=start, baseline=baseline, up=up)

                offset[axis] = 0.5 * (X0[axis] + X1[axis])
                start = self.textRender.write(self.axisLabel, offset,
                    f'{chr(ord("X")+axis)}', flags=48 + visFlag, padding=3.5,
                    start=start, baseline=baseline, up=up)

        self.axisLabelChars = start
        self.axisLabelVertexArray.update(self.axisLabel[:self.axisLabelChars])

    def build_axisLine(self):
        # Ouch! Place the tick lines...
        X0 = self['disp_X0']
        X1 = self['disp_X1']

        MS = self['axis_major_tick_spacing']
        mS = MS / self['axis_minor_ticks']
        ML = MS * self['axis_major_tick_length_ratio']
        mL = ML * self['axis_minor_tick_length_ratio']

        # Let's make sure the axes have actually changed...
        key = (tuple(X0), tuple(X1), MS, mS, ML, mL)
        if getattr(self, '_lastAxis', None) == key:
            return

        self._lastAxis = key

        start = 0

        for axis in range(3):
            i = np.arange(int(np.ceil(X0[axis] / mS)),
                          int(np.ceil(X1[axis] / mS)))
            n = len(i)
            if start + n > self.AXIS_MAX_TICKS:
                n = self.AXIS_MAX_TICKS - start
                i = i[:n]

            i = i.reshape(-1, 1)

            tl = np.full((n, 1), mL, 'f')
            tl[np.where(i % self['axis_minor_ticks'] == 0)] = ML

            y0 = X0[(axis + 1) % 3]
            y1 = X1[(axis + 1) % 3]
            z0 = X0[(axis + 2) % 3]
            z1 = X1[(axis + 2) % 3]

            points = np.zeros((n, 12, 3), 'd')
            points[:, :, 0] = i * mS
            points[:, 0:3,  1:] = (y0, z0)
            points[:, 3:6,  1:] = (y0, z1)
            points[:, 6:9,  1:] = (y1, z1)
            points[:, 9:12, 1:] = (y1, z0)
            points[:, (2,  4), 1] += tl
            points[:, (8, 10), 1] -= tl
            points[:, (1, 11), 2] += tl
            points[:, (5,  7), 2] -= tl

            points = np.roll(points.reshape(12*n, 3), axis, -1)

            f1 = 1 << ((axis + 1) % 3)
            f2 = 8 << ((axis + 2) % 3)
            f3 = 8 << ((axis + 1) % 3)
            f4 = 1 << ((axis + 2) % 3)

            faces = np.array([
                f1 + f4, f1, f4,
                f1 + f2, f2, f1,
                f2 + f3, f3, f2,
                f3 + f4, f4, f3
            ], dtype='u4')

            end = start + n

            self.axisLineVert['position'][(8 + start*12):(8 + end*12)] = points
            self.axisLineVert['faceMask'][(8 + start*12):(8 + end*12)] = np.tile(faces, n)

            start = end
            if start >= self.AXIS_MAX_TICKS:
                break

        self.axisLineVert['position'][:8] = X0 + CUBE_CORNERS * (X1 - X0)
        # end = 0
        self.axisLine.update(self.axisLineVert[:(8 + end*12)])
        # self.axisLine.update(self.axisLineVert[:100])
        # print(self.axisLineVert[:8])
        # self.totalAxisPoints = self.axisEdge.size
        self.totalAxisPoints = self.axisEdge.size + 16 * start

    def build_colormaps(self):
        if not hasattr(self, 'colormapTextures'):
            self.colormapTextures = [
                Texture(size = (256, ), format=GL.GL_RGB,
                    wrap=GL.GL_CLAMP_TO_EDGE, internalFormat=GL.GL_SRGB)
                for i in range(MAX_CHANNELS)]

        for i in range(3):
            name = self[f'vol_colormap{i+1}']
            if name != self._colormap[i]:
                GL.glActiveTexture(GL.GL_TEXTURE0 + self[f'colormap{i+1}Texture'])
                if name not in COLORMAPS:
                    raise ValueError("unknown colormap '%s'" % name)
                self.colormapTextures[i].replace(COLORMAPS[name].data)
                self._colormap[i] = name

        GL.glActiveTexture(GL.GL_TEXTURE0)


    #--------------------------------------------------------
    # Volume Management
    #--------------------------------------------------------

    def resetView(self, direction=None, up=None):
        if direction is None:
            direction = self['camera_pos'] - self['look_at']
        if up is None:
            up = self['up']

        direction = np.asarray(direction, 'f')
        up = np.asarray(up, 'f')

        dir = norm(direction)
        up = norm(up - dot1(up, dir) * dir)
        X0 = self['disp_X0']
        X1 = self['disp_X1']
        L = mag(X1 - X0)
        la = 0.5 * (X1 + X0)

        # print(direction, up, la)

        fov = self['fov']
        tanHD = np.tan((fov if fov > 1E-3 else 30) * np.pi / 360)

        self.update({
            "look_at": la,
            "camera_pos": la + 1.1 * dir * L / (2 * tanHD),
            "up": up
        }, callback=True)

    #--------------------------------------------------------
    # OpenGL setup
    #--------------------------------------------------------

    def setup(self, width=100, height=100):

        vertexType = np.dtype([('position', '3float32'), ('faceMask', 'uint32')])
        self.axisLineVert = np.empty(8 + 12 * self.AXIS_MAX_TICKS, vertexType)
        self.axisLine = VertexArray(vertexType, len(self.axisLineVert))

        faceMask = ((1, 2, 4) << ((CUBE_CORNERS > 0.5) * 3)).sum(-1)

        self.axisEdge = np.array([
            (0, 1), (2, 3), (4, 5), (6, 7), # x-edges
            (0, 2), (4, 6), (1, 3), (5, 7), # y-edges
            (0, 4), (1, 5), (2, 6), (3, 7), # z-edges
            # Corners are handled as a repeated point
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)
        ], dtype='u4')
        e1 = 8 + np.array([(0, 1), (0, 2), (3, 4), (3, 5), (6, 7), (6, 8),
            (9, 10), (9, 11)], dtype='u4')
        edges = np.vstack([
            self.axisEdge,
            (e1 + 12 * np.arange(self.AXIS_MAX_TICKS, dtype='u4').reshape(-1, 1, 1)).reshape(-1, 2)
        ])
        self.axisLine.attachElements(edges, GL.GL_LINES)

        # Get the faces used by each of the 12 axis edges -- needed for axis visiblity
        self.axisEdgeFaceMask = faceMask[self.axisEdge[:12, 0]] & faceMask[self.axisEdge[:12, 1]]

        self.axisLineVert['position'][:8] = self['disp_X0'] + CUBE_CORNERS * (self['disp_X1'] - self['disp_X0'])
        self.axisLineVert['faceMask'][:8] = faceMask
        # self.axisLine.update(self.axisLineVert[:8])

        self.textRender = TextRenderer(os.path.join(os.path.split(__file__)[0], 'fonts', 'Inter-Regular'))
        self['pixelRange'] = self.textRender.pixelRange

        self.axisLabel = np.empty(self.AXIS_LABEL_MAX_CHARS, self.textRender.vertexType)
        self.axisLabelVertexArray = VertexArray(self.textRender.vertexType, self.AXIS_LABEL_MAX_CHARS)

        GL.glClearDepth(1.0)
        GL.glDepthFunc(GL.GL_LESS)
        GL.glClearColor(0.0, 0.0, 0.0, 1.0)

        self.primaryBuffer = self.addBuffer(width, height)

        self.resetRange()
        self.resetView()

    #--------------------------------------------------------
    # Buffer management
    #--------------------------------------------------------

    def addBuffer(self, width, height):
        if self.buffers:
            id = max(self.buffers.keys()) + 1
        else:
            id = 0

        self.buffers[id] = FrameBuffer(width, height,
            internalFormat=GL.GL_SRGB8_ALPHA8, target=GL.GL_TEXTURE_RECTANGLE,
            depthTexture=True)
        # self.perspectiveMatrix.append(None)
        self._needsRebuild.add('perspectiveMatrix')
        return id

    def resizeBuffer(self, id, width, height):
        self.buffers[id].resize(width, height)
        self._needsRebuild.add('perspectiveMatrix')

    #--------------------------------------------------------
    # Main draw method
    #--------------------------------------------------------

    def draw(self, bufferId=None, blitToDefault=False, scaleHeight=None):
        if bufferId is None:
            bufferId = self.primaryBuffer

        # ---- Rebuild anything that is needed ----
        # for n in range(2):
        #     # Sometimes rebuilding one item triggers rebuilding a second!
        #     nr = self._needsRebuild
        #     self._needsRebuild = set()
        #     for param in nr:
        #         getattr(self, 'build_' + param)()

        if self._needsRebuild:
            # Rebuild items that have a specific order
            for item in self._rebuildOrder:
                if item in self._needsRebuild:
                    getattr(self, 'build_' + item)()
                    self._needsRebuild.remove(item)

            # Rebuild unsorted items
            for item in self._needsRebuild:
                getattr(self, 'build_' + item)()

            self._needsRebuild = set()

        defaultFB = GL.glGetIntegerv(GL.GL_DRAW_FRAMEBUFFER_BINDING)

        # ---- Select the draw buffer, and set up for drawing ----
        buffer = self.buffers[bufferId]

        buffer.bind()
        width, height = buffer.width, buffer.height
        viewportSize = np.array([width, height], dtype='f')

        if scaleHeight is None:
            axisScaling = self['display_scaling']
        else:
            axisScaling = height / float(scaleHeight)

        GL.glViewport(0, 0, width, height)
        GL.glEnable(GL.GL_FRAMEBUFFER_SRGB)
        c = self['background_color']
        GL.glClearColor(c[0], c[1], c[2], 1.0)
        GL.glDepthMask(GL.GL_TRUE) # Needs to be before the clear!
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        frame = self['frame']

        # ---- Draw the Mesh ----
        if self.visibleAssets['mesh']:
            GL.glEnable(GL.GL_DEPTH_TEST)
            # Don't draw the back side of meshes... unless we are clipping!
            if not self["mesh_clip"]:
                GL.glEnable(GL.GL_CULL_FACE)
            else:
                GL.glDisable(GL.GL_CULL_FACE)

            GL.glCullFace(GL.GL_BACK)
            GL.glDisable(GL.GL_BLEND)

            shader = self.useShader('mesh')
            # Note: perspective matrix is per-buffer, so we need to update!
            shader['perspectiveMatrix'] = self.perspectiveMatrix[bufferId]

            for id in self.visibleAssets['mesh']:
                asset = self.assets[id]
                asset.setFrame(frame)
                if asset.validFrame:
                    shader.update(asset.uniforms, ignore=True)
                    asset.draw()


        # ---- All subsequent draws allow transparency, and don't write to depth ----
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDepthMask(GL.GL_FALSE)


        # ---- Draw Text ----
        if self['show_axis_labels']:
            shader = self.useShader('text')
            shader['perspectiveMatrix'] = self.perspectiveMatrix[bufferId]
            shader['viewportSize'] = viewportSize
            shader['axis_scaling'] = axisScaling
            shader['font_size'] = self['axis_label_size']
            shader['font_color'] = self['axis_color']

            GL.glActiveTexture(GL.GL_TEXTURE0)
            self.textRender.texture.bind()

            self.axisLabelVertexArray.drawArrays(GL.GL_POINTS, 0, self.axisLabelChars)

        # ---- Draw Axis ----
        if self['show_axis']:
            shader = self.useShader('axis')
            shader['perspectiveMatrix'] = self.perspectiveMatrix[bufferId]
            shader['viewportSize'] = viewportSize
            shader['axis_scaling'] = axisScaling

            self.axisLine.draw(self.totalAxisPoints)


        # ---- Draw Volume ----
        if self.visibleAssets['volume']:
            for id in self.visibleAssets['volume']:
                asset = self.assets[id]
                asset.setFrame(frame)
                if asset.validFrame:
                    # We will draw only the back faces, irrespective if there is
                    #   something in front -- ordering is handled by the renderer
                    #   rather than the usual method!
                    GL.glDisable(GL.GL_DEPTH_TEST)
                    GL.glEnable(GL.GL_CULL_FACE)
                    GL.glCullFace(GL.GL_FRONT)

                    if hasattr(asset, 'volumeTexture'):
                        # Shouldn't need to check, but lets be sure!
                        GL.glActiveTexture(GL.GL_TEXTURE1)
                        asset.volumeTexture.bind()

                    # Get the depth channel from the previous steps
                    GL.glFlush()
                    GL.glActiveTexture(GL.GL_TEXTURE0)
                    buffer.depthTexture.bind()

                    shader = self.useShader('volume')
                    shader['perspectiveMatrix'] = self.perspectiveMatrix[bufferId]
                    shader.update(asset.uniforms, ignore=True)

                    # print('yo!')
                    asset.draw()

                    break; #Only draw one volume!


        # ---- Cleanup and draw to the screen (if required) ----
        self.useShader(None)

        GL.glDisable(GL.GL_FRAMEBUFFER_SRGB)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        if blitToDefault:
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, defaultFB)
            GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, defaultFB)


    #--------------------------------------------------------
    # Cleanup
    #--------------------------------------------------------

    def cleanup(self):
        # pass
        for attr in ('mesh', 'volumeBox', 'textRenderer', 'axisLine', 'volumeTexture', 'depthTexture'):
            item = getattr(self, attr, None)
            if item is not None:
                item.delete()
        for shader in self._cachedShaders.values():
            shader.delete()
        for buffer in self.buffers.values():
            buffer.delete()
