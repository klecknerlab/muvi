#version 330

// uniform mat4 meshModelMatrix = mat4(1.0);
uniform mat4 viewMatrix = mat4(1.0);
uniform mat4 perspectiveMatrix = mat4(1.0);
uniform vec3 camera_pos = vec3(1.0);
uniform float mesh_scale = 1.0;
uniform vec3 mesh_offset = vec3(0.0);

//<<INSERT_SHARED_FUNCS>>

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec4 color;

out VertexData {
    vec3 worldPos;
    vec3 worldNormal;
    vec4 color;
} vOut;

void main()
{
    // vec4 pos = meshModelMatrix * vec4(position, 1.0);
    vOut.worldPos = mesh_offset + mesh_scale * position;
    vOut.color = color;
    vOut.worldNormal = normalize(normal);
    gl_Position = perspectiveMatrix * viewMatrix * vec4(vOut.worldPos, 1.0);
}
