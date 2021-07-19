# Class Description and Dependencies for MUVI Modules

## Main Module Directory: `muvi/`

### File: `__init__.py`

#### `class VolumeProperties`
  - A dictionary like-object that stores the properties of the volumes (and enforces types).  Includes methods to save to a standard XML format, which is also embedded in VTI files.
  - *Used by:* `VolumetricMovie` and descendants


#### `class VolumetricMovie`
  - The base class for volumetric movies.  Includes methods for saving to a standard format (VTI).  Generally not called directly; use the `open_3d_movie` function instead.
  - *Uses:* `VolumeProperties`
  - *Used by:* `VolumetricMovieFrom2D`


#### `VolumetricMoveFrom2D(VolumetricMovie)`
  - A class used for volumetric movies derived from 2D images.  
  - *Uses:* `VolumeProperties`, `VolumetricMovie`


## 2D File Readers: `muvi/readers/`

### File: `cine.py`

#### `Cine`
  - A reader for Cine image files
  - *Used by:* `VolumetricMovieFrom2D`

### File: `seq.py`

#### `Seq`
  - A reader for SEQ files (experimental!)
  - *Used by:* `VolumetricMovieFrom2D`

### File: `vti.py`

#### `VTIMovie`
  - A reader for VTI files.  This reader is very restricted, and will probably not work on files which were not created by this library.
  - *Uses:* `VolumeProperties`


## 3D Viewer: `muvi/view/`

### File: `ooopengl.py`

#### `Texture`
  - OpenGL texture object
  - *Used by:*

#### `FrameBufferObject`
  - OpenGL framebuffer object
  - *Used by:*

#### `ShaderProgram`
  - OpenGL shader program
  - *Used by:*


### File: `qtview.py`

#### `[Bool/Linear/Int/List/Log/Options/Color]ViewSetting`
  - GUI interfaces for various types of view settings.
  - *Used by:* `ViewerApp`

#### `ViewWidget`
  - OpenGL widget which handles main display
  - *Used by:* `ViewerApp`

#### `CollapseableVBox`
  - Pane to contain the view settings, which can be collapsed
  - *Used by:* `ViewerApp`

#### `PlayButton`
  - Button to control start/stop of playback
  - *Used by:* `ViewerApp`

#### `ExportWindow`
  - Window to export images
  - *Used by:* `ViewerApp`

#### `ViewerApp`
  - Main QT app for viewer.  Low level drawing operations handled by the `View` object.
  - *Uses:* `[---]ViewSetting`, `ViewWidget`, `CollapsableVBox`, `PlayButton`, `ExportWindow`, `View`, `VolumetricMovie` or descendants


### File: `view.py`

#### `ColorMap`
  - A colormap, loaded from an XML
  - *Used by:* `View`

#### `View`
  - An OpenGL view object.  Handles the low level drawing.
  - *Used by:* `ViewerApp`
  - *Uses:* `Texture`, `FrameBufferObject`, `ShaderProgram`, `VolumetricMovie` or descendants
