// NAME: Color Map

vec4 cloud_color(in vec4 color, in vec3 X) {
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);
    float a = 0;

    #ifdef CLOUD1_ACTIVE
      c.rgb += texture1D(colormap1_texture, color.r).rgb * color.r;
      c.a += color.r;
    #endif

    #ifdef CLOUD2_ACTIVE
      c.rgb += texture1D(colormap2_texture, color.g).rgb * color.g;
      c.a += color.g;
    #endif

    #ifdef CLOUD3_ACTIVE
      c.rgb += texture1D(colormap3_texture, color.b).rgb * color.b;
      c.a += color.b;
    #endif

    c.rgb *= glow;
    c.a /= glow;

    return c;
}
