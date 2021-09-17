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
