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

uniform mat4 viewMatrix;
uniform mat4 perspectiveMatrix;
uniform vec3 camera_pos;
uniform vec3 disp_X0;
uniform vec3 disp_X1;

layout (location = 0) in vec3 position;

out VertexData {
    vec3 worldPos;
} vOut;

void main()
{
    vOut.worldPos = disp_X0 + (disp_X1 - disp_X0) * position;
    gl_Position = perspectiveMatrix * viewMatrix * vec4(vOut.worldPos, 1.0);
}
