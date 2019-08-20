/*
Copyright 2018 Dustin Kleckner

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This shader is used to render volumes by "view.py".
The $oc and $cc macros will automatically get updated by the python code to
  reflect the color components and opacity components (in one channel
  volumes this is often "rrr" and "r", respectively.)
Additionally a block of code defining uniforms and functions gets inserted at
  the "VOL INIT" marker.
*/

#version 120
#extension GL_ARB_texture_rectangle : enable

/*
The following uniforms and functions are defined by the python code,
    reproduced here for clarity.
uniform sampler3D vol_texture = 0;
uniform sampler2DRect vol_back_buffer = 1;
uniform vec3 vol_size = {256., 256., 256.};
uniform vec3 vol_delta = {1./256., 1./256., 1./256.};
uniform float vol_grad_step = 1.0;
vec3 vol_gradient(in vec3 p);
vec3 vol_texture_space(in vec3 p);
*/

// !! The following line is used to insert code from Python, do not remove !!
<<VOL INIT>>

// Any variables that should be adjustable by the user should be specified with
//   a comment in the format "VIEW_VAR: [value spec]".  This specifies both
//   defaults and how it changes.
// "value spec" is in one of several formats:
//   * float([val], [min], [max], [step (optional)])
//   * int([val], [min], [max], [step (optional)])
//   * logfloat([val], [min], [max], [log base], [step]) (actual step is log_base^1/step)
//   * color([r], [g], [b], [a])

uniform float opacity; // VIEW_VAR: logfloat(0.05, 1E-4, 1.0, 2, 2)
uniform float step_size; // VIEW_VAR: logfloat(1.0, 0.125, 1.0, 2, 2)
uniform float iso_level; // VIEW_VAR: float(0.25, 0.0, 1.0, 0.05)
uniform vec4 surface_color; // VIEW_VAR: color(1.0, 0.0, 0.0, 0.5)
uniform vec3 tint; // VIEW_VAR: color(0.0, 0.3, 0.3)
uniform float shine; // VIEW_VAR: float(0.2, 0.0, 1.0, 0.05)
uniform float grid_thickness; // VIEW_VAR: logfloat(0.5, 0.1, 10.0, 10, 5)
uniform float grid_spacing; // VIEW_VAR: logfloat(8.0, 1.0, 1024.0, 2, 1)
uniform vec4 grid_color; // VIEW_VAR: color(0.0, 0.0, 0.0, 1.0)

#ifdef VOL_SHOW_ISOSURFACE
vec4 add_surface_color(in vec3 p0, in float l0, in vec3 p1, in float l1, in vec3 light_norm, in vec4 color) {
    float z = (iso_level - l0)/(l1 - l0);
    vec3 p = (1.0-z) * p0 + z * p1;
    vec3 grad = normalize(vol_gradient(p));
    float dp = dot(grad, light_norm);
    #ifdef VOL_SHOW_GRID
        //vol_texture_space(p) * vol_size
        vec3 tc = mod(p + 0.5 * grid_thickness, vec3(grid_spacing, grid_spacing, grid_spacing));
        vec4 surf = (min(min(tc.x, tc.y), tc.z) < grid_thickness) ? grid_color : surface_color;
    #else
        vec4 surf = surface_color;
    #endif
    surf.a += (surf.a*(1.0-abs(dp)))*(1.0 - surf.a);
    //surf.rgb = grad;
    surf = min(surf, 1.0);
    surf.rgb *= (dp > 0.0) ? dp : -0.5*dp;
    surf.rgb *= surf.a;
    surf.rgb += shine*pow(abs(dp), 30.0);
    return(color + surf * (1-color.a));
}
#endif

void main() {
    vec3 p0 = gl_TexCoord[0].xyz;
    vec3 p1 = texture2DRect(vol_back_buffer, gl_FragCoord.st).xyz;

    // Current location and step.
    float l = length(p0 - p1);
    vec3 s = (p1-p0)/l * step_size;
    vec3 p = p0 + 0.5 * s;
    vec4 voxel_color;
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    float mod_opacity = step_size * opacity;
    vec3 light_norm = normalize(s);

    #ifdef VOL_SHOW_ISOSURFACE
    vec3 last_p = p0;
    float last_opa = texture3D(vol_texture, vol_texture_space(last_p)).$oc;
    int last_level = (last_opa > iso_level) ? 1 : 0;
    int level;
    #endif

    // Check to make sure that our ray makes sense; if not abort to
    //    avoid ending up in a super long loop.
    if (l < 1.1*length(vol_size)) {
        for (float ll = 0.5 * step_size; ll < l; ll += step_size) {
            voxel_color = vol_rgba_function(p);

            #ifdef VOL_SHOW_ISOSURFACE
            level = (voxel_color.a > iso_level) ? 1 : 0;
            if (last_level != level) {
                color = add_surface_color(last_p, last_opa, p, voxel_color.a, light_norm, color);
            }
            last_level = level;
            last_opa = voxel_color.a;
            last_p = p;
            #endif

            voxel_color.a *= mod_opacity * (1.0-color.a);
            voxel_color.rgb *= tint * voxel_color.a;
            color += voxel_color;
            p += s;
        }

        #ifdef VOL_SHOW_ISOSURFACE
        p = p1;
        float voxel_opa = vol_a_function(p);

        level = (voxel_opa > iso_level) ? 1 : 0;
        if (last_level != level) {
            color = add_surface_color(last_p, last_opa, p, voxel_opa, light_norm, color);
        }
        #endif
    }

    gl_FragColor = vol_output_correct(color);
}
