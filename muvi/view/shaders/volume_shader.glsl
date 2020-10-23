/*
Copyright 2020 Dustin Kleckner

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

// Any variables that should be adjustable by the user should be specified with
//   a comment in the format "VIEW_VAR: [value spec]".  This specifies both
//   defaults and how it changes.
// "value spec" is in one of several formats:
//   * float([val], [min], [max], [step (optional)])
//   * int([val], [min], [max], [step (optional)])
//   * logfloat([val], [min], [max], [log base], [step]) (actual step is log_base^1/step)
//   * color([r], [g], [b], [a])

uniform sampler3D vol_texture;
uniform sampler2DRect back_buffer_texture;
uniform sampler1D colormap_texture;
uniform vec3 vol_size = vec3(256.0, 256.0, 256.0);
uniform vec3 vol_delta = vec3(1.0/256.0, 1.0/256.0, 1.0/256.0);
uniform float grad_step = 1.0;
uniform float gamma_correct = 1.0/2.2;
uniform float exposure = 0.0;

uniform float opacity = 0.1; // VIEW_VAR: logfloat(0.05, 1E-4, 1.0, 2, 2)
uniform float step_size = 1.0; // VIEW_VAR: logfloat(1.0, 0.125, 1.0, 2, 2)
uniform float iso_offset = 0.5;
// uniform float iso_level = 0.5; // VIEW_VAR: float(0.25, 0.0, 1.0, 0.05)
// uniform vec4 surface_color = vec4(1.0, 0.0, 0.0, 0.5); // VIEW_VAR: color(1.0, 0.0, 0.0, 0.5)
// uniform vec3 tint; // VIEW_VAR: color(0.0, 0.3, 0.3)
// uniform float shine = 0.2; // VIEW_VAR: float(0.2, 0.0, 1.0, 0.05)
// uniform float grid_thickness; // VIEW_VAR: logfloat(0.5, 0.1, 10.0, 10, 5)
// uniform float grid_spacing; // VIEW_VAR: logfloat(8.0, 1.0, 1024.0, 2, 1)
// uniform vec4 grid_color; // VIEW_VAR: color(0.0, 0.0, 0.0, 1.0)

uniform float perspective_xfact = 0.0;
uniform float perspective_zfact = 0.0;


vec3 distortion_map(in vec3 U);
mat4x3 distortion_map_gradient(in vec3 U);
vec4 iso_color(in vec4 voxel_color, in mat4x3 grad, in int level);
int iso_level(in vec4 color);
vec4 cloud_color(in vec4 color, in vec3 X);


// !! The following line is used to insert code from Python, do not remove !!
<<VOL INIT>>

// #ifdef VOL_SHOW_ISOSURFACE
//     vec3 vol_gradient(in vec3 p) {
//         return ( vec3(
//             vol_iso_level(p + vec3(1.0, 0.0, 0.0)) - vol_iso_level(p - vec3(1.0, 0.0, 0.0)),
//             vol_iso_level(p + vec3(0.0, 1.0, 0.0)) - vol_iso_level(p - vec3(0.0, 1.0, 0.0)),
//             vol_iso_level(p + vec3(0.0, 0.0, 1.0)) - vol_iso_level(p - vec3(0.0, 0.0, 1.0))
//         ));
//     }
//
//     vec4 add_surface_color(in vec3 p0, in float l0, in vec3 p1, in float l1, in vec3 light_norm, in vec4 color) {
//         float z = (iso_level - l0)/(l1 - l0);
//         vec3 p = (1.0-z) * p0 + z * p1;
//         vec3 grad = normalize(vol_gradient(p));
//         float dp = dot(grad, light_norm);
//         #ifdef VOL_SHOW_GRID
//             //vol_texture_space(p) * vol_size
//             vec3 tc = mod(p + 0.5 * grid_thickness, vec3(grid_spacing, grid_spacing, grid_spacing));
//             vec4 surf = (min(min(tc.x, tc.y), tc.z) < grid_thickness) ? grid_color : surface_color;
//         #else
//             vec4 surf = surface_color;
//         #endif
//         surf.a += (surf.a*(1.0-abs(dp)))*(1.0 - surf.a);
//         //surf.rgb = grad;
//         surf = min(surf, 1.0);
//         surf.rgb *= (dp > 0.0) ? dp : -0.5*dp;
//         surf.rgb *= surf.a;
//         surf.rgb += shine*pow(abs(dp), 30.0);
//         return(color + surf * (1-color.a));
//     }
// #endif

#ifdef VOL_SHOW_ISOSURFACE
    // vec4 surface_color(in int last_level, in int level, vec3 voxel_color, ):
    //     level = vol_iso_lecel(voxel_color);
    //     if (level != last_level) {
    //         int level_dir = (level > last_level) ? 1 : -1;
    //         int level_add = (level_dir > 0) ? 0 : -1;
    //         for (int j = last_level; j != level; j += level_dir) {
    //             vol_iso_color(X, )
    //         }
    //     }
    //     last_level = level;
#endif

void main() {
    // Get the front of the ray from the plane just drawn -- units here are
    //   pixel coordinates in the volume, not yet normalized!
    vec3 P0 = gl_TexCoord[0].xyz;
    // The back of the ray is determined from a pre-drawn buffer.
    vec3 P1 = texture2DRect(back_buffer_texture, gl_FragCoord.st).xyz;

    // Length of a ray in pixels
    float l = length(P0 - P1);

    // The size of a step, converted into normalized coordinates
    vec3 dP = (P1 - P0) / l * step_size;
    vec3 dX = dP * vol_delta;
    // The initial position, also in normalized coordinates
    vec3 X = P0 * vol_delta;
    // Position after map
    vec3 Xm;

    // The overall appearance shouldn't change with step size -- this adjust
    //   the opacity accordingly.
    float mod_opacity = step_size * opacity;

    vec4 voxel_color;
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    vec3 light_norm = normalize(dX);

    // #ifdef VOL_SHOW_ISOSURFACE
    //     vec3 last_P = P0;
    //     float last_opa = texture3D(vol_texture, vol_texture_space(last_p)).$oc;
    //     int last_level = (last_opa > iso_level) ? 1 : 0;
    //     int level;
    // #endif

    // Check if we are starting below or above the isosurface
    // #ifdef VOL_SHOW_ISOSURFACE
    //     Xm = distortion_map(X);
    //     voxel_color = Texture3D(vol_texture, Xm);
    //     int last_level = vol_iso_level(voxel_color);
    //     int level;
    // #endif

    // Round up the number of steps... sort of.  If needed, the last step can
    //  be 20% bigger to cover the volume.
    int num_steps = int(l / step_size + 0.8);
    float last_step_size = l - (num_steps - 1) * step_size;

    float color_mult = pow(2.0, exposure);

    // Start 1/2 a step in
    X += + 0.5 * dX;

    // Check to make sure that our ray makes sense; if not abort to
    //    avoid ending up in a super long loop.
    if (l < 1.1*length(vol_size)) {
        for (int i = num_steps; i > 0; i --) {
            // Since the last step is a little differently sized, adjust
            //    the opacity accordingly.
            if (i == 1) {mod_opacity = last_step_size * opacity;}

            // Map the coordinates, and get the texture at the current location
            Xm = distortion_map(X);
            voxel_color = color_mult * texture3D(vol_texture, Xm);
            // voxel_color.a = clamp(voxel_color.a, 0.0, 1.0);

            // #ifdef VOL_SHOW_ISOSURFACE
            //
            // #endif

            vec4 cc = cloud_color(voxel_color, X);
            cc.a *= mod_opacity * (1.0 - clamp(color.a, 0.0, 1.0));
            cc.rgb *= cc.a;
            color += cc;

            X += dX;
        }
    }

    //     for (float ll = 0.5 * step_size; ll < l; ll += step_size) {
    //         voxel_color = vol_rgba_function(p);
    //
    //         #ifdef VOL_SHOW_ISOSURFACE
    //
    //             level = (voxel_color.a > iso_level) ? 1 : 0;
    //             if (last_level != level) {
    //                 color = add_surface_color(last_p, last_opa, p, voxel_color.a, light_norm, color);
    //             }
    //             last_level = level;
    //             last_opa = voxel_color.a;
    //             last_p = p;
    //         #endif
    //
    //         voxel_color.a *= mod_opacity * (1.0-color.a);
    //         voxel_color.rgb *= tint * voxel_color.a;
    //         color += voxel_color;
    //         p += s;
    //     }
    //
    //     #ifdef VOL_SHOW_ISOSURFACE
    //         p = p1;
    //         float voxel_opa = vol_iso_level(p);
    //
    //         level = (voxel_opa > iso_level) ? 1 : 0;
    //         if (last_level != level) {
    //             color = add_surface_color(last_p, last_opa, p, voxel_opa, light_norm, color);
    //         }
    //     #endif
    // }

    gl_FragColor = pow(color, vec4(gamma_correct, gamma_correct, gamma_correct, 1.0));
    // gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
