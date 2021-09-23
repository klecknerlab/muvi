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

uniform sampler2DRect fontAtlas;
uniform vec3 font_color = vec3(1.0, 1.0, 1.0);
uniform float font_opacity = 1.0;

in VertexData {
    vec2 texCoord;
    float screenPxRange;
} vIn;

out vec4 fragColor;

float median(vec3 c) {
    return max(min(c.r, c.g), min(max(c.r, c.g), c.b));
}

void main() {
    vec3 msd = texture(fontAtlas, vIn.texCoord).rgb;
    float sd = median(msd.rgb);
    float dist = vIn.screenPxRange * (sd - 0.5);
    float opacity = clamp(dist + 0.5, 0.0, 1.0) * font_opacity;
    if (opacity < 1E-3) {
        discard;
    }
    fragColor = vec4(font_color.rgb * opacity, opacity);

    // // Red background for testing
    // fragColor = vec4(text_color.rgb * opacity + vec3(1.0, 0.0, 0.0) * (1-opacity), 1.0);
}
