// NAME: Single Isolevel

vec4 cloud_color(in vec4 color, in vec3 X) {
    return vec4(color.r, color.g, color.b, length(color.rgb));
}
