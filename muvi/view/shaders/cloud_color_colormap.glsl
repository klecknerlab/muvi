// NAME: Color Map

vec4 cloud_color(in vec4 color, in vec3 X) {
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);
    float a = 0;
    float val;


    #ifdef CLOUD1_ACTIVE
      val = sqrt(color.r);
      c.rgb += texture1D(colormap1_texture, val).rgb * val;
      c.a += val;
    #endif

    #ifdef CLOUD2_ACTIVE
      val = sqrt(color.g);
      c.rgb += texture1D(colormap2_texture, val).rgb * val;
      c.a += val;
    #endif

    #ifdef CLOUD3_ACTIVE
      val = sqrt(color.b);
      c.rgb += texture1D(colormap3_texture, val).rgb * val;
      c.a += val;
    #endif

    c.rgb;
    c.a;

    return c;
}
