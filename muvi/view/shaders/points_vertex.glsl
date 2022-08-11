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

uniform float geometry_c0 = 0.0, geometry_c1 = 1.0;
uniform float geometry_scale = 1.0;
uniform mat4 viewMatrix = mat4(1.0);

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in float size;
layout (location = 3) in float color;
layout (location = 4) in uint glyphType;

out GlyphData {
    vec3 position;
    vec3 normal;
    vec3 size;
    float color;
    uint glyphType;
} gOut;

void main()
{
    gOut.position = position;
    gOut.normal = normal;
    gOut.size = vec3(size, size, size) * geometry_scale;
    gOut.color = clamp((color - geometry_c0) / (geometry_c1 - geometry_c0), 0.0, 1.0);
    gOut.glyphType = glyphType;
}
