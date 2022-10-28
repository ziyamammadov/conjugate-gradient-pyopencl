
__kernel void sub(__global const float *a,
                  __global const float *b,
                  __global float *result,
                  const int size) {
    const int __idx = get_global_id(0);
    if (__idx < size)
        for (int r = 0; r < N_RHS; r++)
            result[__idx + r * size] =
                    a[__idx + r * size] -
                    b[__idx + r * size];
}
