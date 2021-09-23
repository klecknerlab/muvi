#version 330

uniform mat4 modelMatrix = mat4(1.0);
uniform mat4 viewMatrix = mat4(1.0);
uniform mat4 perspectiveMatrix = mat4(1.0);

layout (location = 0) in vec3 position;
layout (location = 1) in uint faceMask;

out VertexData {
    uint faceMask;
    int id;
} vOut;

void main()
{
    vOut.id = gl_VertexID;
    vOut.faceMask = faceMask;
    gl_Position = perspectiveMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
}
