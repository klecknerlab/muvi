/*
Copyright 2023 Dustin Kleckner

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

uniform mat4 viewMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform vec3 disp_X0;
uniform vec3 disp_X1;
uniform sampler2DRect colormapTextureId;
uniform float colormapOffset = 0.5;
uniform float geometry_shade = 0.0;
uniform vec3 geometry_shade_color = vec3(1.0);

//<<INSERT_SHARED_FUNCS>>

in VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    float c;
} vIn;

out vec4 fragColor;

void main()
{
    #ifdef MESH_CLIP
    vec3 p = vIn.worldPos;
    if (p.x < disp_X0.x || p.x > disp_X1.x ||
        p.y < disp_X0.y || p.y > disp_X1.y ||
        p.z < disp_X0.z || p.z > disp_X1.z )
    {
        discard;
    }
    #endif

    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb = texture(colormapTextureId, vec2(vIn.c * 255.0 + 0.5, colormapOffset)).rgb * (1.0 - geometry_shade)
        + geometry_shade_color * geometry_shade;
    fragColor = shadeSurface(color, vIn.worldPos, normalize(vIn.worldNormal));
}
