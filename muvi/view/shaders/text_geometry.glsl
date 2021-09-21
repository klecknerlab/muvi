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

uniform float font_size = 15.0;
uniform float pixelRange = 2.0;
uniform float pixelsPerEm;
uniform vec2 viewportSize = vec2(100.0, 100.0);
uniform float axis_scaling = 2.0;
uniform uint visibleAxisLabels = 12u;

layout (points) in;
layout (triangle_strip, max_vertices=4) out;

in GlyphData {
    vec2 offset;
    vec2 right;
    vec2 up;
    vec4 atlas;
    uint visible;
} gIn[1];

out VertexData {
    vec2 texCoord;
    float screenPxRange;
} vOut;

void main()
{
    float fontSize = axis_scaling * font_size;
    vec2 scale = 2.0 * fontSize * gl_in[0].gl_Position.w / viewportSize;
    vec4 pos = gl_in[0].gl_Position;
    pos.xy += gIn[0].offset * scale;
    vec2 right = scale * gIn[0].right;
    vec2 up = scale * gIn[0].up;

    vOut.screenPxRange = fontSize / pixelsPerEm * pixelRange;

    if ((gIn[0].visible == 0u) || ((gIn[0].visible & visibleAxisLabels) > 0u))
    {
        gl_Position = pos;
        vOut.texCoord = gIn[0].atlas.xy;
        EmitVertex();

        gl_Position.xy += right;
        vOut.texCoord = gIn[0].atlas.zy;
        EmitVertex();

        gl_Position = pos;
        gl_Position.xy += up;
        vOut.texCoord = gIn[0].atlas.xw;
        EmitVertex();

        gl_Position.xy += right;
        vOut.texCoord = gIn[0].atlas.zw;
        EmitVertex();
        EndPrimitive();
    }
}
