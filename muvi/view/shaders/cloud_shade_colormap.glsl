// NAME: Color Map

// uniform sampler1D colormap1Texture, colormap2Texture, colormap3Texture;
uniform sampler2DRect colormapTextureId;
uniform float colormap1Offset=0.5, colormap2Offset=0.5, colormap3Offset=0.5;

vec4 cloud_color(in vec4 color, in vec3 pos) {
    vec4 c = vec4(0.0, 0.0, 0.0, 0.0);
    float a = 0;
    float val;

    #ifdef VOL_CLOUD1
      val = sqrt(color.r);
      // c.rgb += texture(colormap1Texture, val).rgb * val;
      c.rgb += texture(colormapTextureId, vec2(val*255.0 + 0.5, colormap1Offset)).rgb * val;
      c.a += val;
    #endif

    #ifdef VOL_CLOUD2
      val = sqrt(color.g);
      // c.rgb += texture(colormap2Texture, val).rgb * val;
      c.rgb += texture(colormapTextureId, vec2(val*255.0 + 0.5, colormap2Offset)).rgb * val;
      c.a += val;
    #endif

    #ifdef VOL_CLOUD3
      val = sqrt(color.b);
      // c.rgb += texture(colormap3Texture, val).rgb * val;
      c.rgb += texture(colormapTextureId, vec2(val*255.0 + 0.5, colormap3Offset)).rgb * val;
      c.a += val;
    #endif

    c.rgb;
    c.a;

    return c;
}
