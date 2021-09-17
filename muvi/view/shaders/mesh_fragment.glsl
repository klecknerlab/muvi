#version 330

uniform mat4 viewMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform vec3 disp_X0;
uniform vec3 disp_X1;

//<<INSERT_SHARED_FUNCS>>

in VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    vec4 color;
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
    fragColor = shadeSurface(vIn.color, vIn.worldPos, normalize(vIn.worldNormal));
}
