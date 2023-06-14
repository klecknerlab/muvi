#version 330

uniform mat4 viewMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform vec3 disp_X0;
uniform vec3 disp_X1;
uniform sampler2DRect colormapTextureId;
uniform float colormapOffset = 0.5;
uniform float geometry_shade = 0.0;
uniform vec3 geometry_shade_color = vec3(1.0);

//<<INSERT_SHARED_FUNCS>>

in VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    float c;
} vIn;

out vec4 fragColor;

void main()
{
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb = texture(colormapTextureId, vec2(vIn.c * 255.0 + 0.5, colormapOffset)).rgb * (1.0 - geometry_shade)
        + geometry_shade_color * geometry_shade;
    fragColor = shadeSurface(color, vIn.worldPos, normalize(vIn.worldNormal));
}
