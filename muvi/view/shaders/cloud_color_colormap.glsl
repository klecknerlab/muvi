// NAME: Color Map

vec4 cloud_color(in vec4 color, in vec3 X) {
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);
    float a = 0;

    #ifdef CLOUD1_ACTIVE
      c += texture1D(colormap1_texture, color.r);
      a += color.r;
    #endif

    #ifdef CLOUD2_ACTIVE
      c += texture1D(colormap2_texture, color.g);
      a += color.g;
    #endif

    #ifdef CLOUD3_ACTIVE
      c += texture1D(colormap3_texture, color.b);
      a += color.b;
    #endif

    c.rgb *= glow;
    c.a = a;

    return c;
}
