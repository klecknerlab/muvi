// NAME: Color Map

uniform sampler1D colormap1Texture, colormap2Texture, colormap3Texture;

vec4 cloud_color(in vec4 color, in vec3 pos) {
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);
    float a = 0;
    float val;

    #ifdef VOL_CLOUD1
      val = sqrt(color.r);
      c.rgb += texture(colormap1Texture, val).rgb * val;
      c.a += val;
    #endif

    #ifdef VOL_CLOUD2
      val = sqrt(color.g);
      c.rgb += texture(colormap2Texture, val).rgb * val;
      c.a += val;
    #endif

    #ifdef VOL_CLOUD3
      val = sqrt(color.b);
      c.rgb += texture(colormap3Texture, val).rgb * val;
      c.a += val;
    #endif

    c.rgb;
    c.a;

    return c;
}
