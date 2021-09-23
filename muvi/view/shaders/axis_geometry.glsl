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

uniform float axis_line_width = 1.0;
uniform float axis_scaling = 2.0;
uniform vec2 viewportSize = vec2(100.0, 100.0);
uniform uint visibleAxisFaces;

layout (lines) in;
layout (triangle_strip, max_vertices=8) out;

in VertexData {
    uint faceMask;
    int id;
} vIn[2];

noperspective out vec2 coord;
flat out int isPoint; // This should be a bool, but opengl doesn't allow!
flat out float lineLength;

vec2 screen(vec4 pos)
{
    return vec2(pos) / pos.w * viewportSize;
}

void main()
{
    float line_width = axis_line_width * axis_scaling;
    if ((vIn[0].faceMask & vIn[1].faceMask & visibleAxisFaces) != uint(0)) {
        isPoint = vIn[0].id == vIn[1].id ? 1 : 0;
        vec4 start = gl_in[0].gl_Position;
        vec4 end = gl_in[1].gl_Position;
        vec2 onePix = 2.0 * start.w / viewportSize;
        float d = line_width + 1.0;

        if (isPoint != 0) {
            gl_Position.zw = start.zw;

            coord = vec2(-d, -d);
            gl_Position.xy = start.xy + coord * onePix;
            EmitVertex();

            coord = vec2(-d, +d);
            gl_Position.xy = start.xy + coord * onePix;
            EmitVertex();

            coord = vec2(+d, -d);
            gl_Position.xy = start.xy + coord * onePix;
            EmitVertex();

            coord = vec2(+d, +d);
            gl_Position.xy = start.xy + coord * onePix;
            EmitVertex();

        } else {
            vec2 screenN = screen(end) - screen(start);
            lineLength = length(screenN);
            screenN = normalize(screenN);

            // Start cap
            vec2 right = screenN * onePix;
            vec2 up = vec2(screenN.y, -screenN.x) * onePix;
            gl_Position.zw = start.zw;

            coord = vec2(-0.5, -d);
            gl_Position.xy = start.xy - right - d * up;
            EmitVertex();

            coord = vec2(-0.5, +d);
            gl_Position.xy = start.xy - right + d * up;
            EmitVertex();

            coord = vec2(+0.5, -d);
            gl_Position.xy = start.xy - d * up;
            EmitVertex();

            coord = vec2(+0.5, +d);
            gl_Position.xy = start.xy + d * up;
            EmitVertex();

            // End cap
            onePix = 2.0 * end.w / viewportSize;
            right = screenN * onePix;
            up = vec2(screenN.y, -screenN.x) * onePix;
            gl_Position.zw = end.zw;

            coord = vec2(lineLength + 0.5, -d);
            gl_Position.xy = end.xy - d * up;
            EmitVertex();

            coord = vec2(lineLength + 0.5, +d);
            gl_Position.xy = end.xy + d * up;
            EmitVertex();

            coord = vec2(lineLength + 1.5, -d);
            gl_Position.xy = end.xy + right - d * up;
            EmitVertex();

            coord = vec2(lineLength + 1.5, +d);
            gl_Position.xy = end.xy + right + d * up;
            EmitVertex();
        }
    }
}
