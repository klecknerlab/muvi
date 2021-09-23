// NAME: Simple

uniform vec3 distortion_correction_factor = vec3(0.0);

vec3 distortion_map(in vec3 U) {
    vec3 eps = 0.25 * distortion_correction_factor * (1.0 - 2.0 * U);
    float exy = eps.x + eps.y;
    float ez = eps.z;

    return vec3((U.x + ez) / (1.0 + 2.0*ez), (U.y + ez) / (1.0 + 2.0*ez), (U.z + exy) / (1.0 + 2.0*exy));
}

mat3 distortion_map_gradient(in vec3 X) {
    mat3 map_grad;

    map_grad[0] = vec3(1.0, 0.0, 0.0);
    map_grad[1] = vec3(0.0, 1.0, 0.0);
    map_grad[2] = vec3(0.0, 0.0, 1.0);

    return map_grad;
}
