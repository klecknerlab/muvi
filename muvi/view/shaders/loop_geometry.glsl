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

uniform vec3 disp_X0;
uniform vec3 disp_X1;
uniform mat4 viewMatrix = mat4(1.0);
uniform mat4 perspectiveMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform int points_skip = 1;
uniform float loop_angle = 0.0;

//<<INSERT_SHARED_FUNCS>>

layout (lines_adjacency) in;
layout (triangle_strip, max_vertices=50) out;
// Assuming 1024 max ouptut attributes: 93 * (4+3+3+1) = 1023
// In practice the arrow glyph has 88 points, so this just barely works!
// Note that computing the color here does *not* work, as 1024 // (4+3+3+3) = 78

in GlyphData {
    vec3 position;
    vec3 normal;
    float size;
    float color;
    uint glyphType;
} gIn[4];

out VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    float c;
} vOut;

void main()
{
    vec3 T01 = gIn[1].position - gIn[0].position;
    float L01 = length(T01);
    T01 /= L01;
    float m01 = (gIn[1].size - gIn[0].size) / L01;

    vec3 T12 = gIn[2].position - gIn[1].position;
    float L12 = length(T12);
    T12 /= L12;
    float m12 = (gIn[2].size - gIn[1].size) / L12;

    vec3 T23 = gIn[3].position - gIn[2].position;
    float L23 = length(T23);
    T23 /= L23;
    float m23 = (gIn[3].size - gIn[2].size) / L23;

    vec3 Ti = normalize(T01 + T12);
    vec3 Tf = normalize(T12 + T23);
    float mi = 0.5 * (m01 + m12);
    float mf = 0.5 * (m12 + m23);

    vec3 Ni0 = gIn[1].normal;
    Ni0 = normalize(Ni0 - Ti * dot(Ti, Ni0));
    vec3 Nf0 = gIn[2].normal;
    Nf0 = normalize(Nf0 - Tf * dot(Tf, Nf0));

    vec3 Bi0 = cross(Ti, Ni0);
    vec3 Bf0 = cross(Tf, Nf0);

    float c = cos(radians(loop_angle));
    float s = sin(radians(loop_angle));

    vec3 Ni = c * Ni0 - s * Bi0;
    vec3 Bi = c * Bi0 + s * Ni0;
    vec3 Nf = c * Nf0 - s * Bf0;
    vec3 Bf = c * Bf0 + s * Nf0;

    vec3 Pi = gIn[1].position;
    vec3 Pf = gIn[2].position;
    float ri = gIn[1].size;
    float rf = gIn[2].size;
    float ci = gIn[1].color;
    float cf = gIn[2].color;

    mat4 Mp = perspectiveMatrix * viewMatrix;

    for (int i = 0; i < 21; i++) {
        float a = radians(18.0 * i);

        vec3 X = cos(a) * Ni - sin(a) * Bi;
        vOut.worldPos = Pi + ri * X;
        vOut.worldNormal = normalize(X - mi * Ti);
        vOut.c = ci;
        gl_Position = Mp * vec4(vOut.worldPos, 1.0);
        EmitVertex();

        X = cos(a) * Nf - sin(a) * Bf;
        vOut.worldPos = Pf + rf * X;
        vOut.worldNormal = normalize(X - mf * Tf);
        vOut.c = cf;
        gl_Position = Mp * vec4(vOut.worldPos, 1.0);
        EmitVertex();
    }
}
