/*
Copyright 2021 Dustin Kleckner

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
Additionally a block of code defining uniforms and functions gets inserted at
  the "VOL INIT" marker.
*/

#version 120
#extension GL_ARB_texture_rectangle : enable
// #extension GL_EXT_gpu_shader4 : enable

#define CROP_CUTOFF 0.0000001

uniform sampler3D vol_texture;
uniform sampler2DRect back_buffer_texture;
uniform sampler1D colormap1_texture;
uniform sampler1D colormap2_texture;
uniform sampler1D colormap3_texture;

uniform vec3 vol_L = vec3(100.0, 100.0, 100.0);
uniform vec3 X0 = vec3(0.0, 0.0, 0.0);
uniform vec3 X1 = vec3(100.0, 100.0, 100.0);
uniform vec3 vol_size = vec3(100.0, 100.0, 100.0);

uniform vec3 camera_loc = vec3(0.0, 0.0, 0.0);

uniform float gamma_correct = 1.0/2.2;
uniform float exposure1 = 0.0;
uniform float exposure2 = 0.0;
uniform float exposure3 = 0.0;
uniform float density = 0.1;
uniform float glow = 1.0;
uniform float step_size = 1.0;

uniform float iso1_level = 0.5;
uniform float iso2_level = 0.5;
uniform float iso3_level = 0.5;
uniform vec3 iso1_color = vec3(1.0, 0.0, 0.0);
uniform vec3 iso2_color = vec3(0.0, 1.0, 0.0);
uniform vec3 iso3_color = vec3(0.0, 0.0, 1.0);
uniform float iso1_opacity = 0.5;
uniform float iso2_opacity = 0.5;
uniform float iso3_opacity = 0.5;
uniform float isosurface_shine = 0.2;

uniform float perspective_xfact = 0.0;
uniform float perspective_yfact = 0.0;
uniform float perspective_zfact = 0.0;



vec3 distortion_map(in vec3 U);
mat3 distortion_map_gradient(in vec3 U);
vec4 cloud_color(in vec4 color, in vec3 X);

vec4 accumulate_isosurface(vec4 color, vec4 surf_color, vec3 N, vec3 Ng) {
    float dp = dot(N, Ng);
    vec4 sc = surf_color;

    // Increase the opacity at glancing angles
    // Mimics what happens for dielectric interfaces, and looks more physical
    sc.a += (sc.a * 2 * (1.0 - abs(dp))) * (1.0 - sc.a);

    // Clamp the brightness
    sc = min(sc, 1.0);

    // Darken the back side of the surface
    sc.rgb *= (dp > 0.0) ? dp : -0.5 * dp;

    // Add specular reflection
    sc.rgb += isosurface_shine * pow(abs(dp), 30.0);

    // Modify the opacity based on what is already in front of this color
    sc.a *= 1.0 - clamp(color.a, 0.0, 1.0);
    sc.rgb *= sc.a;

    // Add in the new color
    return color + sc;
}

// #define ISO_ABOVE(C) ivec3((((ISOSURFACE1 && (C.x > iso1_level)) ? 1 : 0)), 0, 0)
//+ (((ISOLEVEL & 2) && (C.y > iso2_level)) ? 2 : 0) + (((ISOLEVEL & 4) && (C.z > iso3_level)) ? 4 : 0))

// #define GAMMA2_ADJUST 1

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

// #ifdef VOL_SHOW_ISOSURFACE
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
// #endif



    //
    //
    // // Get the front of the ray from the plane just drawn -- units here are
    // //   pixel coordinates in the volume, which we convert to 0-1 normalized
    // vec3 P0 = gl_TexCoord[0].xyz / vol_L;
    // // The back of the ray is determined from a pre-drawn buffer.
    // vec3 P1 = texture2DRect(back_buffer_texture, gl_FragCoord.st).xyz / vol_L;
    //
    // // Length of a ray
    // float l = length((P0 - P1)*vol_size);
    //
    // // The size of a step, converted into normalized coordinates
    // vec3 dP = (P1 - P0)*vol_size / l * step_size;
    // vec3 dX = dP * vol_delta;
    // // The initial position, also in normalized coordinates
    // vec3 X = P0;
    // // Position after map
    // vec3 Xm;
    //
    // // The overall appearance shouldn't change with step size -- this adjusts
    // //   the opacity accordingly.
    // float mod_opacity = step_size * density / glow;
    //
    // vec4 voxel_color;
    // vec4 color = vec4(0.0, 0.0, 0.0, 0.0);
    // vec3 light_norm = normalize(dX);

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

void main() {
    // Build the start and end of the ray, as well as distances along the ray
    vec4 back_color = texture2DRect(back_buffer_texture, gl_FragCoord.st);

    // Make sure we are even in a region where there is something to draw...
    if (back_color.a < 1E-3) {
        gl_FragColor = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // vec3 Xf = gl_TexCoord[0].xyz;
    vec3 Xf = camera_loc;
    vec3 Xb = back_color.xyz;
    vec3 delta = Xb - Xf;
    float d0 = 0;
    float d1 = length(delta);
    vec3 N = normalize(delta);

    // Crop the ray segment to the active volume
    if (abs(N.x) > CROP_CUTOFF) {
    	float a0 = (X0.x - Xf.x) / N.x;
    	float a1 = (X1.x - Xf.x) / N.x;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(X0.x, X1.x);
    	float M = max(X0.x, X1.x);
    	if (Xf.x <= m || Xf.x >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    if (abs(N.y) > CROP_CUTOFF) {
    	float a0 = (X0.y - Xf.y) / N.y;
    	float a1 = (X1.y - Xf.y) / N.y;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(X0.y, X1.y);
    	float M = max(X0.y, X1.y);
    	if (Xf.y<= m || Xf.y >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    if (abs(N.z) > CROP_CUTOFF) {
    	float a0 = (X0.z - Xf.z) / N.z;
    	float a1 = (X1.z - Xf.z) / N.z;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(X0.z, X1.z);
    	float M = max(X0.z, X1.z);
    	if (Xf.z <= m || Xf.z >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    // Find the location of the segment ends in texture space
    vec3 U0 = (Xf + d0 * N) / vol_L;
    vec3 U1 = (Xf + d1 * N) / vol_L;
    delta = U1 - U0;

    // Length of the ray in voxels
    float Lv = length(delta * vol_size);
    delta *= step_size / Lv;

    // Round up the number of steps... sort of.  If needed, the last step can
    //  be 20% bigger to cover the volume.
    int num_steps = int(Lv / step_size + 0.8);
    float last_step_size = Lv - (num_steps - 1) * step_size;

    // Exposure adjustment
    vec4 color_mult = vec4(
      pow(2, exposure1),
      pow(2, exposure2),
      pow(2, exposure3),
      1.0);

    // Start 1/2 a step in, and with no color
    vec3 U = U0 + 0.5 * delta;
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);

    // Keep track if we are above or below the isosurface(s)
    // ivec3 last_level = vec3(-1, -1, -1);
    #ifdef ISOSURFACE1
        bool last_above1 = false;
    #endif
    #ifdef ISOSURFACE2
        bool last_above2 = false;
    #endif
    #ifdef ISOSURFACE3
        bool last_above3 = false;
    #endif

    // Modified opacity
    float mod_opacity = step_size * density / glow;

    // Shortcut render for testing the ray clipping
    // float x = Lv / 100;
    // vec3 int_color = sin(vec3(x*6.15, x*7.55, x*8.51));
    // int_color = int_color * int_color;
    // gl_FragColor = vec4(int_color, 1.0);
    // return;

    // Check to make sure that our ray makes sense; if not abort to
    //    avoid ending up in a super long loop.
    if (Lv < 1.1*length(vol_size)) {
        for (int i = num_steps; i > 0; i --) {
            // Since the last step is a little differently sized, adjust
            //    the opacity accordingly.
            if (i == 1) {mod_opacity = last_step_size * density / glow;}

            // Map the coordinates, and get the texture at the current location
            vec3 Uc = distortion_map(U);
            vec4 voxel_color = texture3D(vol_texture, Uc).<<COLOR_REMAP>>;

            // Often the volumes are given with gamma=2 for optimal storage
            #ifdef GAMMA2_ADJUST
                voxel_color *= voxel_color;
            #endif

            // Exposure adjustment
            voxel_color *= color_mult;

            // Compute cloud color at this locations
            vec4 cc = cloud_color(voxel_color, U);

            // Accumulate the cloud color
            cc.a *= mod_opacity * (1.0 - clamp(color.a, 0.0, 1.0));
            cc.rgb *= cc.a;
            color += cc;

            // Are we also testing isosurfaces?
            #if defined ISOSURFACE1 || defined ISOSURFACE2 || defined ISOSURFACE3
                // Check each axis to see if we went from above to below.
                bool flipped = false;
                bool above;

                #ifdef ISOSURFACE1
                    above = voxel_color.x > iso1_level;
                    bool flipped1 = above != last_above1;
                    flipped = flipped || flipped1;
                    last_above1 = above;
                #endif

                #ifdef ISOSURFACE2
                    above = voxel_color.y > iso2_level;
                    bool flipped2 = above != last_above2;
                    flipped = flipped || flipped2;
                    last_above2 = above;
                #endif

                #ifdef ISOSURFACE3
                    above = voxel_color.z > iso3_level;
                    bool flipped3 = above != last_above3;
                    flipped = flipped || flipped3;
                    last_above3 = above;
                #endif

                // Only compute isosurfaces if this is not the first point
                //    AND at least one point flipped.
                // Note: the gradient calculation is expensive, so don't do
                //    it unless needed!
                if ((i < num_steps) && flipped) {
                    // We found an isosurface!
                    // Let's compute the gradient first.
                    mat3 mg = distortion_map_gradient(U);
                    mat3 gradient;

                    // Derivative in each direction is a 2-point stencil
                    // Do all colors at once.
                    // Note: since we will normalize the gradient, we can ignore
                    // exposure, gamma, etc.
                    for (int j = 0; j < 3; j++) {
                        mg[j] /= vol_size;
                        // mg[j] *= 0.5;
                        gradient[j] = texture3D(vol_texture, Uc + mg[j]).<<COLOR_REMAP>>.rgb -
                            texture3D(vol_texture, Uc - mg[j]).<<COLOR_REMAP>>.rgb;
                    }
                    gradient = transpose(gradient);

                    #ifdef ISOSURFACE1
                        if (flipped1) {
                            color = accumulate_isosurface(
                                        color, vec4(iso1_color, iso1_opacity), N,
                                        normalize(gradient[0])
                                    );
                        }
                    #endif

                    #ifdef ISOSURFACE2
                        if (flipped2) {
                            color = accumulate_isosurface(
                                        color, vec4(iso2_color, iso2_opacity), N,
                                        normalize(gradient[1])
                                    );
                        }
                    #endif

                    #ifdef ISOSURFACE3
                        if (flipped3) {
                            color = accumulate_isosurface(
                                        color, vec4(iso3_color, iso3_opacity), N,
                                        normalize(gradient[2])
                                    );
                        }
                    #endif
                }
            #endif

            // Step forward
            U += delta;
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
}
