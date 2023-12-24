# MUVI GEOMETRY: PointSequence
# Note: the first line must be identical to the above so that the MUVI library
#   recognizes this as a plugin!

# This script is meant to be loaded using muvi.geometry.loadgeometry
# It dynamicallty generates data using the python script below
# The output is _identical_ to loop.py, but generated on the fly

import numpy as np
import time

# Regular variables are basically ignored -- they exist only in the local, sandboxed, namespace.
N = 500
Nt = 500

# This special variable indicates the data length -- it will give an error if this is not defined!
# This should be a set or castable to one.
_valid_frames = range(Nt)

# This special variable holds the same data as the "display" attribute in a Geometry VTK file
_display = dict(
    render_as = 'loop', # Display as a closed loop rather than discrete points
    size = 't', # The field used to determine diameter of the rendered 3D tube; can vary with position!
    scale = 1, # Scale factor for the size
    color = 'c', # The field used to determine the (varying) color of the line
    X0 = [-2, -2, -1], # The display limits; if not specified these are automatically determined.
    X1 = [2, 2, 1], # The display limits; if not specified these are automatically determined.
    colormap = 'twilight', # The colormap used to convert the color value ("c" field) to an actual tube color.
)

# This special variable is used to define metadata -- this can be used to store arbitrary user data.
# You can leave this undefined if you wish, but is here defined as an example.
# Note that the "__file__" special variable is defined in the local namespace -- you can use this to
#   open files if needed!
_metadata = dict(
    info = f'Dyanamically generated on {time.strftime("%Y/%m/%d")}',
    source = __file__
)

# Another local variable -- used below.
ϕ = np.linspace(0, 2*np.pi, N, False)

# This special function is used to retrive a frame
def _get(i):
    global np, Nt, ϕ, N
    # Why did we do this?  Because of the way this code executes, you need to explicitly specify
    #   which variables will be used in the _get function.
    # This is _not_ required for anything else, as the _get function is the only thing  that will get
    #   executed later.

    θ = 2*np.pi*i/Nt
    aspect = 0.3 * (1.5 + np.cos(θ))

    r = 1 + aspect * np.sin(3*ϕ)

    X = np.zeros((N, 3))
    X[:, 0] = r * np.cos(2*ϕ)
    X[:, 1] = r * np.sin(2*ϕ)
    X[:, 2] = aspect * np.cos(3*ϕ)
    thickness = 0.05 * (1.2 + np.sin(19*ϕ + 2*θ))
    color = np.sin(9*ϕ -θ)

    return X, dict(t = thickness, c = color)
