/*
Copyright 2022 Dustin Kleckner

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

//<<INSERT_SHARED_FUNCS>>

layout (points) in;
layout (triangle_strip, max_vertices=93) out;
// Assuming 1024 max ouptut attributes: 93 * (4+3+3+1) = 1023
// In practice the arrow glyph has 88 points, so this just barely works!
// Note that computing the color here does *not* work, as 1024 // (4+3+3+3) = 78

in GlyphData {
    vec3 position;
    vec3 normal;
    vec3 size;
    float color;
    uint glyphType;
} gIn[1];

out VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    float c;
} vOut;

void main()
{
    if (gl_PrimitiveIDIn % points_skip == 0) {

        #ifdef MESH_CLIP
        vec3 wp = gIn[0].position;
        if (wp.x < disp_X0.x || wp.x > disp_X1.x ||
            wp.y < disp_X0.y || wp.y > disp_X1.y ||
            wp.z < disp_X0.z || wp.z > disp_X1.z )
        {
            return;
        }
        #endif

        vOut.c = gIn[0].color;

        vec3 X = normalize(gIn[0].normal);
        vec3 Y;
        if (abs(X.x) > 0.9) {Y = vec3(0.0, 1.0, 0.0);}
        else {Y = vec3(1.0, 0.0, 0.0);}

        // if (length(gIn[0].Y) == 0.0) {
        //     if (abs(X.x) > 0.9) {Y = vec3(0.0, 1.0, 0.0);}
        //     else {Y = vec3(1.0, 0.0, 0.0);}
        // } else {
        //     Y = gIn[0].Y;
        // }

        Y = normalize(Y - X * dot(X, Y));
        vec3 Z = cross(X, Y);

        mat4 Mwp = mat4(1.0);
        Mwp[0].xyz = X * gIn[0].size.x;
        Mwp[1].xyz = Y * gIn[0].size.y;
        Mwp[2].xyz = Z * gIn[0].size.z;
        // Mwp = transpose(Mwp);
        Mwp[3].xyz = gIn[0].position;

        mat3 Mwn = mat3(1.0);
        Mwn[0] = X;
        Mwn[1] = Y;
        Mwn[2] = Z;
        // Mwn = transpose(Mwn);

        mat4 Mp = perspectiveMatrix * viewMatrix * Mwp;
        vec4 pos;
        vec4 p = vec4(0.0, 0.0, 0.0, 0.0);
        // vec4(gIn[0].position, 0.0);

        // Everything below this line is auto-generated -- do not edit!
        // <<START GLYPH VERTICES>>
        switch (gIn[0].glyphType) {
            case 0u: //sphere
                pos = p + vec4(-0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                pos = p + vec4(-0.2628655433654785, -0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.525731086730957, -0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(-0.4253253936767578, 0.0, 0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, 0.525731086730957);
                EmitVertex();

                pos = p + vec4(0.0, -0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, 0.8506507873535156);
                EmitVertex();

                pos = p + vec4(0.0, 0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.525731086730957, 0.8506507873535156);
                EmitVertex();

                pos = p + vec4(0.4253253936767578, 0.0, 0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.8506507873535156, 0.0, 0.525731086730957);
                EmitVertex();

                pos = p + vec4(0.2628655433654785, 0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.525731086730957, 0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                pos = p + vec4(0.0, 0.2628655433654785, -0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.525731086730957, -0.8506507873535156);
                EmitVertex();

                pos = p + vec4(0.0, -0.2628655433654785, -0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, -0.8506507873535156);
                EmitVertex();

                pos = p + vec4(-0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                pos = p + vec4(-0.2628655433654785, -0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.525731086730957, -0.8506507873535156, 0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.2628655433654785, -0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, -0.8506507873535156);
                EmitVertex();

                pos = p + vec4(0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                pos = p + vec4(0.2628655433654785, -0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.525731086730957, -0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.4253253936767578, 0.0, 0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.8506507873535156, 0.0, 0.525731086730957);
                EmitVertex();

                pos = p + vec4(0.0, -0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, 0.8506507873535156);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, 0.8506507873535156);
                EmitVertex();

                pos = p + vec4(-0.2628655433654785, -0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.525731086730957, -0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.2628655433654785, -0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.525731086730957, -0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.2628655433654785, -0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.525731086730957, -0.8506507873535156);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.525731086730957, 0.8506507873535156);
                EmitVertex();

                pos = p + vec4(0.2628655433654785, 0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.525731086730957, 0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(-0.2628655433654785, 0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.525731086730957, 0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.2628655433654785, -0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.525731086730957, -0.8506507873535156);
                EmitVertex();

                pos = p + vec4(-0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.4253253936767578, 0.0, -0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, -0.525731086730957);
                EmitVertex();

                pos = p + vec4(-0.4253253936767578, 0.0, 0.2628655433654785, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.8506507873535156, 0.0, 0.525731086730957);
                EmitVertex();

                pos = p + vec4(-0.2628655433654785, 0.4253253936767578, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-0.525731086730957, 0.8506507873535156, 0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.2628655433654785, 0.4253253936767578, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.525731086730957, 0.8506507873535156);
                EmitVertex();

                EndPrimitive();

                break;
            case 1u: //arrow
                pos = p + vec4(0.0, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, 9.184851394388759e-18, -0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.0, -0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, -0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.0, -0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, -0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.0, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.0, 1.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.0, 1.0);
                EmitVertex();

                pos = p + vec4(0.0, 0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, 0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.0, 0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, 0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.0, 0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, 0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.0, 0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, 0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.0, 9.184851394388759e-18, -0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 1.2246468525851679e-16, -1.0);
                EmitVertex();

                pos = p + vec4(-0.5, 9.184851394388759e-18, -0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 1.2246468525851679e-16, -1.0);
                EmitVertex();

                pos = p + vec4(0.0, -0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, -0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.0, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.0, -0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 9.184851394388759e-18, -0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.04408389329910278, -0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.07132923603057861, -0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.07500000298023224, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.07132923603057861, 0.023176275193691254, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.04408389329910278, 0.060676272958517075, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.5629961490631104, 0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.0, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.9109469056129456, 0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.7748977541923523, 0.5629961490631104);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.9109469056129456, 0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.0, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.9109469056129456, -0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.9578262567520142, -1.7594983955906073e-16);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.9109469056129456, -0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.0, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.5629961490631104, -0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.7748977541923523, -0.5629961490631104);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.5629961490631104, -0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.0, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 1.172998886277439e-16, -0.9578262567520142);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.2959845960140228, -0.9109469056129456);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 1.172998886277439e-16, -0.9578262567520142);
                EmitVertex();

                pos = p + vec4(0.0, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.5629961490631104, -0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.2959845960140228, -0.9109469056129456);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.5629961490631104, -0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.0, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.9109469056129456, -0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.7748977541923523, -0.5629961490631104);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.9109469056129456, -0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.0, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.9109469056129456, 0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.9578262567520142, 5.864994431387194e-17);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.9109469056129456, 0.2959845960140228);
                EmitVertex();

                pos = p + vec4(0.0, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.5629961490631104, 0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.7748977541923523, 0.5629961490631104);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.0, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.5629961490631104, 0.7748977541923523);
                EmitVertex();

                pos = p + vec4(0.0, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.0, 0.9578262567520142);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.2959845960140228, 0.9109469056129456);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.2959845960140228, 0.9109469056129456);
                EmitVertex();

                pos = p + vec4(0.0, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, 0.0, 0.9578262567520142);
                EmitVertex();

                pos = p + vec4(0.0, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.2873478829860687, -0.5629961490631104, 0.7748977541923523);
                EmitVertex();

                EndPrimitive();

                break;
            case 2u: //tick
                pos = p + vec4(-0.5, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.5812821984291077, 0.8000662922859192);
                EmitVertex();

                pos = p + vec4(-0.5, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.9405343532562256, 0.3055981397628784);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.8000662922859192, 0.5812821984291077);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.9405343532562256, 0.3055981397628784);
                EmitVertex();

                pos = p + vec4(-0.5, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.9405343532562256, -0.3055981397628784);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.9889363646507263, -1.816646649750671e-16);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.9405343532562256, -0.3055981397628784);
                EmitVertex();

                pos = p + vec4(-0.5, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.5812821984291077, -0.8000662922859192);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.8000662922859192, -0.5812821984291077);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.5812821984291077, -0.8000662922859192);
                EmitVertex();

                pos = p + vec4(-0.5, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 1.2110977665004473e-16, -0.9889363646507263);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.3055981397628784, -0.9405343532562256);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 1.8369702788777518e-17, -0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 1.2110977665004473e-16, -0.9889363646507263);
                EmitVertex();

                pos = p + vec4(-0.5, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.5812821984291077, -0.8000662922859192);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.3055981397628784, -0.9405343532562256);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.08816778659820557, -0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.5812821984291077, -0.8000662922859192);
                EmitVertex();

                pos = p + vec4(-0.5, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.9405343532562256, -0.3055981397628784);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.8000662922859192, -0.5812821984291077);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.14265847206115723, -0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.9405343532562256, -0.3055981397628784);
                EmitVertex();

                pos = p + vec4(-0.5, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.9405343532562256, 0.3055981397628784);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.9889363646507263, 6.055488832502236e-17);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.14265847206115723, 0.04635255038738251, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.9405343532562256, 0.3055981397628784);
                EmitVertex();

                pos = p + vec4(-0.5, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.5812821984291077, 0.8000662922859192);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.8000662922859192, 0.5812821984291077);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.5812821984291077, 0.8000662922859192);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.0, 0.9889363646507263);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.3055981397628784, 0.9405343532562256);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.5, 0.0, 0.0, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.3055981397628784, 0.9405343532562256);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.15000000596046448, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, 0.0, 0.9889363646507263);
                EmitVertex();

                pos = p + vec4(-0.5, -0.08816778659820557, 0.12135254591703415, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.14834044873714447, -0.5812821984291077, 0.8000662922859192);
                EmitVertex();

                EndPrimitive();

                break;
            case 3u: //cylinder
                pos = p + vec4(0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.0, 1.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.5, -0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, -0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.5, -0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, -0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.5, -0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, 1.2246467698671066e-17, -0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 1.2246468525851679e-16, -1.0);
                EmitVertex();

                pos = p + vec4(0.5, 1.2246467698671066e-17, -0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 1.2246468525851679e-16, -1.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.5, 0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, -0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, -0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, 0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(0.5, 0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.9510565400123596, 0.30901700258255005);
                EmitVertex();

                pos = p + vec4(-0.5, 0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(0.5, 0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.5877852439880371, 0.80901700258255);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.0, 1.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, 0.0, 1.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(0.0, -0.5877852439880371, 0.80901700258255);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.5, -0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, -0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, -0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 1.2246467698671066e-17, -0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 1.2246467698671066e-17, -0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, -0.05877852439880371, -0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(0.5, 0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                pos = p + vec4(0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(1.0, 0.0, 0.0);
                EmitVertex();

                EndPrimitive();

                pos = p + vec4(-0.5, 0.09510564804077148, -0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.0, 0.10000000149011612, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.09510564804077148, 0.030901700258255005, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                pos = p + vec4(-0.5, 0.05877852439880371, 0.08090169727802277, 1.0);
                gl_Position = Mp * pos;
                vOut.worldPos = (Mwp * pos).xyz;
                vOut.worldNormal = Mwn * vec3(-1.0, -0.0, -0.0);
                EmitVertex();

                EndPrimitive();

                break;
        }
    // <<END GLYPH VERTICES>>
    }
}
