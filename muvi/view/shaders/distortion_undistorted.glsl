vec3 distortion_map(in vec3 U) {
    return U;
}

mat4x3 distortion_map_gradient(in vec3 U){
    mat4x3 map_grad;
    map_grad[0] = U;
    map_grad[1] = vec3(1.0, 0.0, 0.0);
    map_grad[2] = vec3(0.0, 1.0, 0.0));
    map_grad[3] = vec3(0.0, 0.0, 1.0));

    return map_grad
}
