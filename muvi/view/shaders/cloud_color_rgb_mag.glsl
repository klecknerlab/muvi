// NAME: RGB w/ Mag. Opacity

vec4 cloud_color(in vec4 color, in vec3 X) {
    return vec4(color.r*glow, color.g*glow, color.b*glow, length(color.rgb));
}
