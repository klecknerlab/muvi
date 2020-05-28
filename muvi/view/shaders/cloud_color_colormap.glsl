// NAME: Color Map

vec4 cloud_color(in vec4 color, in vec3 X) {
    vec4 cm = texture1D(colormap_texture, color.r);
    // cm.r = 1;
    cm.a = color.r;

    return cm;
}
