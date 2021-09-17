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

uniform vec3 axis_line_color = vec3(0.0);
uniform float axis_line_width;
uniform float axis_scaling;

noperspective in vec2 coord;
flat in int isPoint;
flat in float lineLength;

out vec4 fragColor;

void main() {
    float line_width = axis_line_width * axis_scaling;
    float a;
    if (isPoint != 0)
    {
        a = 1.0 - clamp(length(coord) - 0.5 * (line_width - 1.0), 0.0, 1.0);
    } else {
        a = clamp(min(coord.x, lineLength - coord.x), 0.0, 1.0) *
            (1.0 - clamp(abs(coord.y) - 0.5 * (line_width - 1.0), 0.0, 1.0));
    }
    if (a < 1E-6) {discard;}

    fragColor = vec4(axis_line_color * a, a);
}
