// NAME: Camera Oriented Lights

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

uniform vec3 light_N1 = vec3(0.0, -0.25, -1.0);
uniform vec3 light_C1 = vec3(1.0, 1.0, 1.0);
uniform vec3 light_N2 = vec3(0.0);
uniform vec3 light_C2 = vec3(0.1, 0.05, 0.05);
uniform vec3 light_N3 = vec3(-2, +1, 2.0);
uniform vec3 light_C3 = vec3(0.0);
uniform vec3 light_N4 = vec3(1);
uniform vec3 light_C4 = vec3(0);
uniform float specular_brightness = 0.75;
uniform float specular_power = 100.0;
uniform vec3 edge_color = vec3(0.1, 0.2, 0.4)*1E-1;
uniform float edge_power = 50;

vec3 diffuseSpecular(vec4 color, vec3 N, vec3 C, vec3 L, vec3 lightColor)
{
    vec3 H = normalize(C + L);
    float highlight = specular_brightness * pow(clamp(dot(H, -N), 0.0, 1.0), specular_power);
    vec3 outColor = lightColor * vec3(color) * clamp(-dot(L, N), 0.0, 1.0);
    return outColor * color.a + highlight * lightColor;
}

vec4 shadeSurface(vec4 color, vec3 pos, vec3 N)
{
    pos = vec3(viewMatrix * vec4(pos, 1.0));
    N = mat3(viewMatrix) * N;
    vec3 C = normalize(pos);

    // If we're looking at the back side, dark it.
    if (dot(C, N) > 0)
    {
        N *= -1.0;
        color.rgb *= vec3(0.1) + color.rgb * 0.1;
        // color.r = 1.0;
    }

    float edge_highlight = pow(1.0 - pow(dot(C, N), 2.0), 5.0);

    vec3 c = edge_highlight * edge_color;
    c += diffuseSpecular(color, N, C, normalize(light_N1), light_C1);
    c += diffuseSpecular(color, N, C, normalize(light_N2), light_C2);
    c += diffuseSpecular(color, N, C, normalize(light_N3), light_C3);
    c += diffuseSpecular(color, N, C, normalize(light_N4), light_C4);

    return vec4(c, color.a);
}
