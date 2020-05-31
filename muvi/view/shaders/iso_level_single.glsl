// NAME: Single Isolevel

int iso_level(in vec4 color)
    return color.a > iso_offset ? 0 : 1;
}
