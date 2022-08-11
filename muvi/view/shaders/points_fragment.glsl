#version 330

uniform mat4 viewMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform vec3 disp_X0;
uniform vec3 disp_X1;
uniform sampler2DRect colormapTextureId;
uniform float colormapOffset = 0.5;

//<<INSERT_SHARED_FUNCS>>

in VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    float c;
} vIn;

out vec4 fragColor;

void main()
{
    #ifdef MESH_CLIP
    vec3 pos = vIn.worldPos;
    if (pos.x < disp_X0.x || pos.x > disp_X1.x ||
        pos.y < disp_X0.y || pos.y > disp_X1.y ||
        pos.z < disp_X0.z || pos.z > disp_X1.z )
    {
        discard;
    }
    #endif
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    color.rgb = texture(colormapTextureId, vec2(vIn. c * 255.0 + 0.5, colormapOffset)).rgb;
    fragColor = shadeSurface(color, vIn.worldPos, normalize(vIn.worldNormal));
}