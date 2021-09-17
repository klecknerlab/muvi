// NAME: Simple

uniform vec3 vol_perspective_correction_factor = vec3(0.0);

vec3 distortion_map(in vec3 U) {
    float exy = 0.25 * (vol_perspective_correction_factor.x * (1.0 - 2.0 * U.x) + vol_perspective_correction_factor.y * (1.0 - 2.0 * U.y));
    float ez = 0.25 * (vol_perspective_correction_factor.z * (1.0 - 2.0 * U.z));
    return vec3((U.x + ez) / (1.0 + 2.0*ez), (U.y + ez) / (1.0 + 2.0*ez), (U.z + exy) / (1.0 + 2.0*exy));
}

mat3 distortion_map_gradient(in vec3 X) {
    mat3 map_grad;

    map_grad[0] = vec3(1.0, 0.0, 0.0);
    map_grad[1] = vec3(0.0, 1.0, 0.0);
    map_grad[2] = vec3(0.0, 0.0, 1.0);

    return map_grad;
}
