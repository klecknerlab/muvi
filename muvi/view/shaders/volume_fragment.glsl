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

#version 330

#define CROP_CUTOFF 0.0000001

uniform mat4 perspectiveMatrix;
uniform mat4 viewMatrix;
uniform vec3 camera_pos;
uniform float fov;
uniform vec3 look_at;

uniform sampler3D volumeTextureId;
uniform sampler2DRect depthTexture;

uniform vec3 _vol_L = vec3(100.0);
uniform vec3 disp_X0;
uniform vec3 disp_X1;
uniform vec3 _vol_N = vec3(100.0);

uniform float vol_exposure1 = 0.0;
uniform float vol_exposure2 = 0.0;
uniform float vol_exposure3 = 0.0;
uniform float vol_density = 0.01;
uniform float vol_glow = 1.0;
uniform float vol_step_size = 1.0;

uniform float vol_iso1_level = 0.25;
uniform float vol_iso2_level = 0.5;
uniform float vol_iso3_level = 0.5;
uniform vec3 vol_iso1_color = vec3(1.0, 0.0, 0.0);
uniform vec3 vol_iso2_color = vec3(0.0, 1.0, 0.0);
uniform vec3 vol_iso3_color = vec3(0.0, 0.0, 1.0);
uniform float vol_iso1_opacity = 0.9;
uniform float vol_iso2_opacity = 0.5;
uniform float vol_iso3_opacity = 0.5;

uniform vec4 vol_background_color = vec4(0.0, 0.0, 0.0, 1.0);

vec3 distortion_map(in vec3 U);
mat3 distortion_map_gradient(in vec3 U);
vec4 cloud_color(in vec4 color, in vec3 X);

/*
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
*/

// !! The following line is used to insert code from Python, do not remove !!
//<<INSERT_SHARED_FUNCS>>

vec4 accumulate_isosurface(vec4 color, vec4 surf_color, vec3 X, vec3 N)
{
    vec4 sc = shadeSurface(surf_color, X, N);
    sc.a *= 1.0 - clamp(color.a, 0.0, 1.0);
    sc.rgb *= sc.a;

    return color + sc;
}

in VertexData {
    vec3 worldPos;
} vIn;

out vec4 fragColor;

float z_from_depth(float depth)
{
    float d = 1.0 - 2.0 * depth;
    return -(perspectiveMatrix[3][2] + perspectiveMatrix[3][3] * d)/
            (perspectiveMatrix[2][2] + perspectiveMatrix[2][3] * d);
}

void main()
{
    // Build the start and end of the ray, as well as distances along the ray
    vec3 Xf = camera_pos;
    vec3 Xb = vIn.worldPos;
    if (fov < 1E-3)
    {
        Xf = Xb + camera_pos - look_at;
    }

    vec3 delta = Xb - Xf;
    vec3 clip0 = max(-0.5*_vol_L, disp_X0);
    vec3 clip1 = min(0.5*_vol_L, disp_X1);

    float d0 = 0;
    float d1 = length(delta);
    vec3 N = normalize(delta);

    vec4 back_color = vec4(0.0);

    // Check if there is an object between us and the back
    float z_depth = z_from_depth(texture(depthTexture, gl_FragCoord.st).r);
    float z_back = z_from_depth(gl_FragCoord.z);
    if (z_depth > z_back) {
        // We need to shorten the ray
        d1 *= z_depth / z_back;
    }

    // Crop the ray segment to the active volume
    if (abs(N.x) > CROP_CUTOFF) {
    	float a0 = (clip0.x - Xf.x) / N.x;
    	float a1 = (clip1.x - Xf.x) / N.x;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(clip0.x, clip1.x);
    	float M = max(clip0.x, clip1.x);
    	if (Xf.x <= m || Xf.x >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    if (abs(N.y) > CROP_CUTOFF) {
    	float a0 = (clip0.y - Xf.y) / N.y;
    	float a1 = (clip1.y - Xf.y) / N.y;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(clip0.y, clip1.y);
    	float M = max(clip0.y, clip1.y);
    	if (Xf.y<= m || Xf.y >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    if (abs(N.z) > CROP_CUTOFF) {
    	float a0 = (clip0.z - Xf.z) / N.z;
    	float a1 = (clip1.z - Xf.z) / N.z;
    	float b0 = min(a0, a1);
    	float b1 = max(a0, a1);
    	d0 = clamp(d0, b0, b1);
    	d1 = clamp(d1, b0, b1);
    } else {
    	float m = min(clip0.z, clip1.z);
    	float M = max(clip0.z, clip1.z);
    	if (Xf.z <= m || Xf.z >= M) {
    		d0 = 0.0;
    		d1 = 0.0;
    	}
    }

    if (d1 <= d0) {
        discard;
    }

    // Find the location of the segment ends in texture space
    vec3 U0 = (Xf + d0 * N) / _vol_L + 0.5;
    vec3 U1 = (Xf + d1 * N) / _vol_L + 0.5;
    delta = U1 - U0;

    // Length of the ray in voxels
    float Lv = length(delta * _vol_N);
    delta *= vol_step_size / Lv;

    // Round up the number of steps... sort of.  If needed, the last step can
    //  be 20% bigger to cover the volume.
    int num_steps = int(Lv / vol_step_size + 0.8);
    float last_vol_step_size = Lv - (num_steps - 1) * vol_step_size;

    // vol_exposure adjustment
    vec4 color_mult = vec4(pow(2, vol_exposure1), pow(2, vol_exposure2),
        pow(2, vol_exposure3), 1.0);

    // Start 1/2 a step in, and with no color
    vec3 U = U0 + 0.5 * delta;
    vec4 color = vec4(0.0, 0.0, 0.0, 0.0);

    // Keep track if we are above or below the isosurface(s)
    // ivec3 last_level = vec3(-1, -1, -1);
    #ifdef VOL_ISO1
        bool last_above1 = false;
    #endif
    #ifdef VOL_ISO2
        bool last_above2 = false;
    #endif
    #ifdef VOL_ISO3
        bool last_above3 = false;
    #endif

    // Modified opacity
    float mod_opacity = vol_step_size * vol_density / vol_glow;

    // Shortcut render for testing the ray clipping
    // float x = Lv / 100.0;
    // vec3 int_color = sin(vec3(x*6.15, x*7.55, x*8.51));
    // int_color *= int_color;
    // fragColor = vec4(int_color*0.5, 0.5);
    // return;

    // Check to make sure that our ray makes sense; if not abort to
    //    avoid ending up in a super long loop.
    if (Lv < 1.1*length(_vol_N)) {
        for (int i = num_steps; i > 0; i --) {
            // Since the last step is a little differently sized, adjust
            //    the opacity accordingly.
            if (i == 1) {mod_opacity = last_vol_step_size * vol_density / vol_glow;}

            // Map the coordinates, and get the texture at the current location
            vec3 Uc = distortion_map(U);
            vec4 voxel_color = texture(volumeTextureId, Uc).<<COLOR_REMAP>>a;

            // Often the volumes are given with gamma=2 for optimal storage
            #ifdef GAMMA2_ADJUST
                voxel_color *= voxel_color;
            #endif

            // vol_exposure adjustment
            voxel_color *= color_mult;

            // Compute cloud color at this locations
            vec4 cc = cloud_color(voxel_color, U);

            // Accumulate the cloud color
            cc.a *= mod_opacity * (1.0 - clamp(color.a, 0.0, 1.0));
            cc.rgb *= cc.a * vol_glow;
            color += cc;

            // Are we also testing isosurfaces?
            #if defined VOL_ISO1 || defined VOL_ISO2 || defined VOL_ISO3
                // Check each axis to see if we went from above to below.
                bool flipped = false;
                bool above;

                #ifdef VOL_ISO1
                    above = voxel_color.r > vol_iso1_level;
                    bool flipped1 = above != last_above1;
                    flipped = flipped || flipped1;
                    last_above1 = above;
                #endif

                #ifdef VOL_ISO2
                    above = voxel_color.g > vol_iso2_level;
                    bool flipped2 = above != last_above2;
                    flipped = flipped || flipped2;
                    last_above2 = above;
                #endif

                #ifdef VOL_ISO3
                    above = voxel_color.b > vol_iso3_level;
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

                    vec3 X = U * _vol_L;

                    // Derivative in each direction is a 2-point stencil
                    // Do all colors at once.
                    // Note: since we will normalize the gradient, we can ignore
                    // vol_exposure, gamma, etc.
                    for (int j = 0; j < 3; j++) {
                        mg[j] /= _vol_N;
                        // mg[j] *= 0.5;
                        gradient[j] = texture(volumeTextureId, Uc + mg[j]).<<COLOR_REMAP>> -
                            texture(volumeTextureId, Uc - mg[j]).<<COLOR_REMAP>>;
                    }
                    gradient = transpose(gradient);

                    #ifdef VOL_ISO1
                        if (flipped1) {
                            color = accumulate_isosurface(color,
                                vec4(vol_iso1_color, vol_iso1_opacity), X, -normalize(gradient[0]));
                        }
                    #endif

                    #ifdef VOL_ISO2
                        if (flipped2) {
                            color = accumulate_isosurface(color,
                                vec4(vol_iso2_color, vol_iso2_opacity), X, -normalize(gradient[1]));
                        }
                    #endif

                    #ifdef VOL_ISO3
                        if (flipped3) {
                            color = accumulate_isosurface(color,
                                vec4(vol_iso3_color, vol_iso3_opacity), X, -normalize(gradient[2]));
                        }
                    #endif
                }
            #endif

            // Step forward
            U += delta;
        }
    } else {
        color = vec4(1.0, 0.0, 0.0, 0.5);
    }

    // Accumulate the background color
    vec4 c = vol_background_color;
    c.a *= 1.0 - clamp(color.a, 0.0, 1.0);
    // c.rgb *= c.a * vol_glow;
    color += c;

    fragColor = color;
}
