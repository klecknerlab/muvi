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

uniform mat4 modelMatrix = mat4(1.0);
uniform mat4 viewMatrix = mat4(1.0);
uniform mat4 perspectiveMatrix = mat4(1.0);
uniform float pixelsPerEm = 32.0;

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 offset;
layout (location = 2) in vec4 atlas;

out GlyphData {
    vec2 offset;
    vec2 right;
    vec2 up;
    vec4 atlas;
} gOut;

void main()
{
    gOut.atlas = atlas;
    float c = cos(offset.z);
    float s = sin(offset.z);
    vec2 emSize = (atlas.zy - atlas.xw) / pixelsPerEm;
    gOut.offset = vec2(offset.x * c - offset.y * s, offset.x * s + offset.y * c);
    gOut.right = emSize[0] * vec2(c, s);
    gOut.up = emSize[1] * vec2(-s, c);
    gl_Position = perspectiveMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}
