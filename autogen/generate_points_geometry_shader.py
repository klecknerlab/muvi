from pyffi.utils.tristrip import stripify
from muvi.mesh import get_glyph
import random
# import re

fn = '../muvi/view/shaders/points_geometry.glsl'

with open(fn, 'rt') as f:
    shader = f.read()

START = '// <<START GLYPH VERTICES>>'
END = '// <<END GLYPH VERTICES>>'

start = shader.find(START)
end = shader.find(END)

if start < 0 or end < 0:
    raise ValueError('missing glyph start and/or end... aborting!')

with open(fn, 'wt') as f:
    f.write(shader[:start])
    f.write(START + '\n')

    f.write('    switch (gIn[0].glyphType) {\n')

    for i, glyph in enumerate(['sphere', 'arrow', 'tick', 'cylinder']):
        mesh = get_glyph(glyph)
        tris = mesh.triangles.tolist()
        strips = stripify(tris)

        f.write(f'        case {i}u: //{glyph}\n')

        for strip in strips:
            for point in strip:
                X = mesh.points[point]
                N = mesh.normals[point]
                f.write(f'            pos = p + vec4({X[0]}, {X[1]}, {X[2]}, 1.0);\n')
                f.write(f'            gl_Position = Mp * pos;\n')
                f.write(f'            vOut.worldPos = (Mwp * pos).xyz;\n')
                f.write(f'            vOut.worldNormal = Mwn * vec3({N[0]}, {N[1]}, {N[2]});\n')
                f.write(f'            EmitVertex();\n\n')
            f.write(f'            EndPrimitive();\n\n')

        f.write('            break;\n')

    f.write('    }\n')
    f.write('    ' + shader[end:])
