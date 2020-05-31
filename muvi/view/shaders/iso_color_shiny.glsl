// NAME: Single Isolevel

vec4 iso_color(in vec4 voxel_color, in vec3 grad, in int level) {
    //     vec4 add_surface_color(in vec3 p0, in float l0, in vec3 p1, in float l1, in vec3 light_norm, in vec4 color) {
    //         float z = (iso_level - l0)/(l1 - l0);
    //         vec3 p = (1.0-z) * p0 + z * p1;
    //         vec3 grad = normalize(vol_gradient(p));
    //         float dp = dot(grad, light_norm);
    //         #ifdef VOL_SHOW_GRID
    //             //vol_texture_space(p) * vol_size
    //             vec3 tc = mod(p + 0.5 * grid_thickness, vec3(grid_spacing, grid_spacing, grid_spacing));
    //             vec4 surf = (min(min(tc.x, tc.y), tc.z) < grid_thickness) ? grid_color : surface_color;
    //         #else
    //             vec4 surf = surface_color;
    //         #endif
    //         surf.a += (surf.a*(1.0-abs(dp)))*(1.0 - surf.a);
    //         //surf.rgb = grad;
    //         surf = min(surf, 1.0);
    //         surf.rgb *= (dp > 0.0) ? dp : -0.5*dp;
    //         surf.rgb *= surf.a;
    //         surf.rgb += shine*pow(abs(dp), 30.0);
    //         return(color + surf * (1-color.a));
}
